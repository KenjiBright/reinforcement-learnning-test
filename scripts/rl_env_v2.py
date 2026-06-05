"""
rl_env_v2.py  –  Clean 4WS Robot Gymnasium Environment (v2)
============================================================
Design goals
  - Single file covers all three maze levels (1=corridor, 2=L-shape, 3=U-shape)
  - Reward in a small scale (~[-5, +5] per step) — compatible with VecNormalize
  - Single smooth wall-penalty (no multi-tier heuristics)
  - Robust reset: 6 retries, stop + drain before each attempt
  - Clean observation: 24 LiDAR (norm) + goal_dist + sin/cos(heading_err) + speed + steer = 29 D

Usage
    env = Gazebo4WSEnvV2(level=1)   # 1 / 2 / 3
"""

import time
import threading
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from rclpy.qos import qos_profile_sensor_data

# ─────────────────────────────────────────────
# Level configurations
# ─────────────────────────────────────────────
LEVELS = {
    1: {
        "name"      : "Corridor (Lv1)",
        "waypoints" : [[5.0, 0.0], [10.0, 0.0], [16.0, 0.0]],
        "goal_idx"  : 2,
        "spawn_x"   : (-0.5,  1.5),
        "spawn_y"   : (-0.5,  0.5),
        "oob"       : (-2.0, 18.0, -3.0, 3.0),   # xmin xmax ymin ymax
        "timeout"   : 4000,
    },
    2: {
        "name"      : "L-Shape (Lv2)",
        "waypoints" : [[4.0, 0.0], [7.5, 0.0], [9.5, 0.5], [10.0, 3.0], [10.0, 6.0], [10.0, 9.5]],
        "goal_idx"  : 5,
        "spawn_x"   : (-2.0,  3.0),
        "spawn_y"   : (-1.5,  1.5),
        "oob"       : (-3.0, 13.0, -3.0, 13.0),
        "timeout"   : 6000,
    },
    3: {
        "name"      : "U-Shape (Lv3)",
        "waypoints" : [[3.0, 0.0], [7.0, 0.0], [10.0, 2.5],
                       [10.0, 5.0], [10.0, 8.5], [9.0, 10.0],
                       [4.0, 10.0], [0.0, 10.0]],
        "goal_idx"  : 7,
        "spawn_x"   : (-2.0,  3.5),
        "spawn_y"   : (-1.5,  1.5),
        "oob"       : (-3.0, 13.0, -3.0, 13.0),
        "timeout"   : 8000,
    },
}


class Gazebo4WSEnvV2(gym.Env):
    metadata = {"render_modes": []}

    # ──────────────── init ────────────────
    def __init__(self, level: int = 3):
        super().__init__()
        assert level in LEVELS, f"level must be 1, 2 or 3 — got {level}"
        self.cfg   = LEVELS[level]
        self.level = level

        # Action: [speed_cmd, steer_cmd] ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Obs: 24 lidar + goal_dist + sin(err) + cos(err) + speed + steer = 29
        self.observation_space = spaces.Box(
            low=-np.ones(29, dtype=np.float32),
            high= np.ones(29, dtype=np.float32),
            dtype=np.float32,
        )

        # ── ROS2 ──
        self._node = rclpy.create_node(f"rl_env_v2_lv{level}")
        self._lock = threading.Lock()

        self._steer_pub = self._node.create_publisher(
            Float64MultiArray, "/steering_controller/commands", 10
        )
        self._wheel_pub = self._node.create_publisher(
            Float64MultiArray, "/wheel_controller/commands", 10
        )
        self._node.create_subscription(
            LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data
        )
        self._node.create_subscription(
            Odometry, "/ground_truth", self._odom_cb, 10
        )
        self._reset_cli = self._node.create_client(Empty, "/reset_world")

        # ── State ──
        self._pos   = np.zeros(2)
        self._yaw   = 0.0
        self._speed = 0.0
        self._lidar = np.full(24, 10.0)
        self._got_pose = False

        self._step      = 0
        self._steer     = 0.0
        self._cmd_speed = 0.0          # rad/s, smoothed wheel velocity
        self._wp_idx    = 0
        self._prev_dist = 0.0

        # ── Episode stats ──
        self._episode    = 0
        self._ep_reward  = 0.0

        threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        ).start()

    # ──────────────── ROS callbacks ────────────────
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # Discard any message that contains nan/inf (physics explosion)
        if not all(np.isfinite([p.x, p.y, q.w, q.x, q.y, q.z])):
            return
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        fwd = vx * np.cos(yaw) + vy * np.sin(yaw)
        with self._lock:
            self._pos[:] = [p.x, p.y]
            self._yaw    = yaw
            self._speed  = fwd if np.isfinite(fwd) else 0.0
            self._got_pose = True

    def _scan_cb(self, msg):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, 10.0)
        raw = np.clip(raw, 0.0, 10.0)
        if len(raw) != 24:
            idx = np.round(np.linspace(0, len(raw) - 1, 24)).astype(int)
            raw = raw[idx]
        with self._lock:
            self._lidar = raw

    # ──────────────── Observation ────────────────
    def _get_obs(self) -> np.ndarray:
        with self._lock:
            lidar = self._lidar.copy()
            pos   = self._pos.copy()
            yaw   = self._yaw
            speed = self._speed

        target = np.array(self.cfg["waypoints"][self._wp_idx])
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        dist   = np.hypot(dx, dy)
        angle  = np.arctan2(dy, dx)
        herr   = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))

        obs = np.concatenate([
            np.clip(lidar / 10.0, 0.0, 1.0),            # 24  lidar (0→1)
            [np.clip(dist / 20.0, 0.0, 1.0)],           #  1  goal distance
            [np.sin(herr)],                               #  1  heading error sin
            [np.cos(herr)],                               #  1  heading error cos
            [np.clip(speed / 0.15, -1.0, 1.0)],         #  1  forward speed
            [np.clip(self._steer / 0.785, -1.0, 1.0)],  #  1  steer
        ]).astype(np.float32)
        # Safety: ensure no nan/inf reaches the policy network
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ──────────────── Action helpers ────────────────
    def _publish_stop(self):
        stop = Float64MultiArray()
        stop.data = [0.0] * 4
        self._steer_pub.publish(stop)
        self._wheel_pub.publish(stop)

    # ──────────────── Step ────────────────
    def step(self, action):
        # Map actions to physical commands
        # action[0] ∈ [-1, 1] → target speed ∈ [-1, 3] m/s → wheel rad/s
        # wheel_radius = 0.05 m → max 20 rad/s = 1.0 m/s linear
        target_spd = float(action[0]) * 2.0 + 1.0           # m/s: [-1.0, +3.0]
        target_str = float(action[1]) * 0.75                 # rad: [-0.75, +0.75]

        # Smooth steer: max 0.05 rad/step (≈1.5 rad/s at 100 Hz — matches URDF limit)
        self._steer += np.clip(target_str - self._steer, -0.05, 0.05)

        # Convert m/s → rad/s, then smooth at 3.0 rad/s per step
        target_wv = np.clip(target_spd / 0.05, -20.0, 20.0)
        self._cmd_speed += np.clip(target_wv - self._cmd_speed, -3.0, 3.0)
        wv = self._cmd_speed

        w_msg = Float64MultiArray(); w_msg.data = [wv] * 4
        # 4WS counter-steer: front +δ, rear −δ  (same as rl_env_u_shape.py)
        s_msg = Float64MultiArray()
        s_msg.data = [self._steer, self._steer, -self._steer, -self._steer]
        self._wheel_pub.publish(w_msg)
        self._steer_pub.publish(s_msg)

        time.sleep(0.002)
        self._step += 1

        # Snapshot
        with self._lock:
            lidar = self._lidar.copy()
            pos   = self._pos.copy()
            yaw   = self._yaw
        min_d = float(np.min(lidar))

        target = np.array(self.cfg["waypoints"][self._wp_idx])
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        dist   = float(np.hypot(dx, dy))
        angle  = np.arctan2(dy, dx)
        herr   = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))

        # ── Reward ──
        reward = 0.0
        done   = False
        trunc  = False
        info   = {"wp": self._wp_idx, "end_reason": None}

        # 1. Progress toward current WP — only reward forward-aligned progress
        align = float(np.cos(herr))
        reward += (self._prev_dist - dist) * 5.0 * max(0.0, align)

        # 1b. Penalise reverse driving
        if self._speed < -0.05:
            reward -= abs(self._speed) * 4.0

        # 2. Time penalty
        reward -= 0.05

        # 3. Smooth wall proximity (single signal, no tiers)
        if min_d < 1.0:
            reward -= (1.0 - min_d) * 1.5

        # 4. Collision (immunity for first 30 steps)
        if self._step > 30 and min_d < 0.35:
            reward -= 50.0
            done = True
            info["end_reason"] = "collision"
            print(
                f"[Ep {self._episode:4d}|step {self._step:5d}]  CRASH   "
                f"lidar={min_d:.2f}m  pos=({pos[0]:.1f},{pos[1]:.1f})  "
                f"wp={self._wp_idx}/{self.cfg['goal_idx']}  R={self._ep_reward+reward:.0f}",
                flush=True,
            )

        # 5. Out-of-bounds
        if not done:
            xmn, xmx, ymn, ymx = self.cfg["oob"]
            if pos[0] < xmn or pos[0] > xmx or pos[1] < ymn or pos[1] > ymx:
                reward -= 50.0
                done = True
                info["end_reason"] = "oob"
                print(
                    f"[Ep {self._episode:4d}|step {self._step:5d}]  OOB     "
                    f"pos=({pos[0]:.1f},{pos[1]:.1f})  "
                    f"wp={self._wp_idx}/{self.cfg['goal_idx']}  R={self._ep_reward+reward:.0f}",
                    flush=True,
                )

        # 6. Waypoint / Goal
        if not done:
            accept_r = 0.8 if self._wp_idx == self.cfg["goal_idx"] else 1.2
            if dist < accept_r:
                if self._wp_idx < self.cfg["goal_idx"]:
                    reward += 10.0
                    self._wp_idx += 1
                    info["wp"] = self._wp_idx
                    print(
                        f"[Ep {self._episode:4d}|step {self._step:5d}]  WP      "
                        f"{self._wp_idx-1}->{self._wp_idx}/{self.cfg['goal_idx']}  "
                        f"pos=({pos[0]:.1f},{pos[1]:.1f})",
                        flush=True,
                    )
                    new_tgt = np.array(self.cfg["waypoints"][self._wp_idx])
                    dist = float(np.hypot(new_tgt[0] - pos[0], new_tgt[1] - pos[1]))
                else:
                    reward += 200.0
                    done = True
                    info["end_reason"] = "goal"
                    print(
                        f"[Ep {self._episode:4d}|step {self._step:5d}]  GOAL    "
                        f"pos=({pos[0]:.1f},{pos[1]:.1f})  "
                        f"R={self._ep_reward+reward:.0f}  *** LV{self.level} COMPLETE ***",
                        flush=True,
                    )

        # 7. Timeout
        if not done and self._step >= self.cfg["timeout"]:
            trunc = True
            info["end_reason"] = "timeout"
            print(
                f"[Ep {self._episode:4d}|step {self._step:5d}]  TIMEOUT "
                f"wp={self._wp_idx}/{self.cfg['goal_idx']}  "
                f"pos=({pos[0]:.1f},{pos[1]:.1f})  R={self._ep_reward+reward:.0f}",
                flush=True,
            )

        self._ep_reward += reward
        self._prev_dist  = dist

        return self._get_obs(), reward, done, trunc, info

    # ──────────────── Reset ────────────────
    def reset(self, seed=None, options=None):
        self._publish_stop()
        time.sleep(0.2)                  # let velocities drain

        for attempt in range(6):
            self._publish_stop()         # clear any residual velocity commands
            with self._lock:
                self._got_pose = False
            self._reset_cli.call_async(Empty.Request())
            time.sleep(0.15)             # drain old pose messages

            # Wait for fresh pose (up to 1.0 s)
            t0 = time.time()
            while time.time() - t0 < 1.0:
                with self._lock:
                    got = self._got_pose
                if got:
                    break
                time.sleep(0.05)

            time.sleep(0.1)              # brief settle

            with self._lock:
                sx, sy = self._pos[0], self._pos[1]

            xlo, xhi = self.cfg["spawn_x"]
            ylo, yhi = self.cfg["spawn_y"]
            if xlo <= sx <= xhi and ylo <= sy <= yhi:
                break
            print(
                f"[RESET] Physics explosion ({sx:.1f}, {sy:.1f}), "
                f"retry {attempt + 1}/6",
                flush=True,
            )
            time.sleep(0.3)

        # ── Re-initialise episode state ──
        self._episode   += 1
        self._step       = 0
        self._ep_reward  = 0.0
        self._steer      = 0.0
        self._cmd_speed  = 0.0
        self._wp_idx     = 0
        with self._lock:
            self._lidar = np.full(24, 10.0)
            target = np.array(self.cfg["waypoints"][0])
            self._prev_dist = float(np.linalg.norm(self._pos - target))

        return self._get_obs(), {}

    # ──────────────── Close ────────────────
    def close(self):
        self._publish_stop()
        self._node.destroy_node()
