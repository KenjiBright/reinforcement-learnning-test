"""
rl_env_4wd.py  –  4WD Differential-Drive Robot Gymnasium Environment
=====================================================================
Robot: robot_4wd  (4 driven wheels, no steer joints, 360° LiDAR)
Control: skid-steer  →  action[0]=linear, action[1]=angular
  v_left  = (v_lin - v_ang * 0.12) / 0.05   [rad/s]
  v_right = (v_lin + v_ang * 0.12) / 0.05   [rad/s]

Observation (29-D):
  24  LiDAR ranges (normalised 0→1, 10 m max)
   1  distance to current waypoint  (/ 20 m)
   1  sin(heading error)
   1  cos(heading error)
   1  forward speed                 (/ 1.0 m/s)
   1  angular velocity              (/ 2.0 rad/s)

Usage
    env = Gazebo4WDEnv(level=1)    # 1=corridor  2=L-shape  3=U-shape
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
# Level configurations  (same mazes as 4WS)
# ─────────────────────────────────────────────
LEVELS = {
    1: {
        "name"      : "Corridor (Lv1)",
        "waypoints" : [[5.0, 0.0], [10.0, 0.0], [16.0, 0.0]],
        "goal_idx"  : 2,
        "spawn_x"   : (-0.5,  1.5),
        "spawn_y"   : (-0.5,  0.5),
        "oob"       : (-2.0, 18.0, -3.0, 3.0),
        "timeout"   : 4000,
    },
    2: {
        "name"      : "L-Shape (Lv2)",
        "waypoints" : [[4.0, 0.0], [8.0, 0.0], [10.0, 2.5], [10.0, 6.0], [10.0, 9.5]],
        "goal_idx"  : 4,
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

# Robot geometry
WHEEL_RADIUS  = 0.05   # m
TRACK_HALF    = 0.12   # m  (half of wheel centre-to-centre distance in Y)
MAX_LIN       = 1.0    # m/s   (action[0] * MAX_LIN)
MAX_ANG       = 2.0    # rad/s (action[1] * MAX_ANG)
MAX_WHEEL_VEL = 25.0   # rad/s (hardware limit)
SMOOTH_STEP   = 3.0    # rad/s per env-step (wheel velocity ramp limit)


class Gazebo4WDEnv(gym.Env):
    metadata = {"render_modes": []}

    # ──────────────── init ────────────────
    def __init__(self, level: int = 1):
        super().__init__()
        assert level in LEVELS, f"level must be 1, 2 or 3 — got {level}"
        self.cfg   = LEVELS[level]
        self.level = level

        # Action: [linear ∈ [-1,1], angular ∈ [-1,1]]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Obs: 24 lidar + goal_dist + sin(err) + cos(err) + speed + ang_vel = 29
        self.observation_space = spaces.Box(
            low=-np.ones(29, dtype=np.float32),
            high= np.ones(29, dtype=np.float32),
            dtype=np.float32,
        )

        # ── ROS2 ──
        self._node = rclpy.create_node(f"rl_env_4wd_lv{level}")
        self._lock = threading.Lock()

        # 4WD has only one controller topic
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
        self._pos     = np.zeros(2)
        self._yaw     = 0.0
        self._speed   = 0.0    # forward  m/s
        self._ang_vel = 0.0    # angular  rad/s
        self._lidar   = np.full(24, 10.0)
        self._got_pose = False

        self._step      = 0
        self._cmd_left  = 0.0   # rad/s, smoothed left-wheel velocity
        self._cmd_right = 0.0   # rad/s, smoothed right-wheel velocity
        self._wp_idx    = 0
        self._prev_dist = 0.0

        # Episode stats
        self._episode   = 0
        self._ep_reward = 0.0

        threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        ).start()

    # ──────────────── ROS callbacks ────────────────
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if not all(np.isfinite([p.x, p.y, q.w, q.x, q.y, q.z])):
            return
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        fwd = vx * np.cos(yaw) + vy * np.sin(yaw)
        wz  = msg.twist.twist.angular.z
        with self._lock:
            self._pos[:]  = [p.x, p.y]
            self._yaw     = yaw
            self._speed   = fwd if np.isfinite(fwd) else 0.0
            self._ang_vel = wz  if np.isfinite(wz)  else 0.0
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
            lidar   = self._lidar.copy()
            pos     = self._pos.copy()
            yaw     = self._yaw
            speed   = self._speed
            ang_vel = self._ang_vel

        target = np.array(self.cfg["waypoints"][self._wp_idx])
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        dist   = np.hypot(dx, dy)
        angle  = np.arctan2(dy, dx)
        herr   = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))

        obs = np.concatenate([
            np.clip(lidar / 10.0, 0.0, 1.0),               # 24  lidar (0→1)
            [np.clip(dist / 20.0, 0.0, 1.0)],              #  1  goal distance
            [np.sin(herr)],                                  #  1  heading error sin
            [np.cos(herr)],                                  #  1  heading error cos
            [np.clip(speed   / MAX_LIN, -1.0, 1.0)],       #  1  forward speed
            [np.clip(ang_vel / MAX_ANG, -1.0, 1.0)],       #  1  angular velocity
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ──────────────── Action helpers ────────────────
    def _publish_stop(self):
        stop = Float64MultiArray()
        stop.data = [0.0, 0.0, 0.0, 0.0]
        self._wheel_pub.publish(stop)

    # ──────────────── Step ────────────────
    def step(self, action):
        # ── Differential-drive kinematics ──
        # action[0] ∈ [-1, 1] → v_lin ∈ [-1.0, +1.0] m/s
        # action[1] ∈ [-1, 1] → v_ang ∈ [-2.0, +2.0] rad/s
        v_lin = float(action[0]) * MAX_LIN
        v_ang = float(action[1]) * MAX_ANG

        v_left_tgt  = np.clip((v_lin - v_ang * TRACK_HALF) / WHEEL_RADIUS,
                               -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
        v_right_tgt = np.clip((v_lin + v_ang * TRACK_HALF) / WHEEL_RADIUS,
                               -MAX_WHEEL_VEL, MAX_WHEEL_VEL)

        # Smooth wheel velocities
        self._cmd_left  += np.clip(v_left_tgt  - self._cmd_left,  -SMOOTH_STEP, SMOOTH_STEP)
        self._cmd_right += np.clip(v_right_tgt - self._cmd_right, -SMOOTH_STEP, SMOOTH_STEP)

        # Publish [fl, fr, rl, rr]  (left=fl,rl  right=fr,rr)
        w_msg = Float64MultiArray()
        w_msg.data = [self._cmd_left, self._cmd_right,
                      self._cmd_left, self._cmd_right]
        self._wheel_pub.publish(w_msg)

        time.sleep(0.002)
        self._step += 1

        # Snapshot
        with self._lock:
            lidar = self._lidar.copy()
            pos   = self._pos.copy()
        min_d = float(np.min(lidar))

        target = np.array(self.cfg["waypoints"][self._wp_idx])
        dist   = float(np.linalg.norm(pos - target))

        # ── Reward ──
        reward = 0.0
        done   = False
        trunc  = False
        info   = {"wp": self._wp_idx, "end_reason": None}

        # 1. Progress toward current waypoint
        reward += (self._prev_dist - dist) * 5.0

        # 2. Time penalty
        reward -= 0.05

        # 3. Wall proximity penalty
        if min_d < 1.0:
            reward -= (1.0 - min_d) * 0.3

        # 4. Collision (immunity for first 30 steps after reset)
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

        # 6. Waypoint / Goal reached
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
                    dist = float(np.linalg.norm(pos - new_tgt))
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
        time.sleep(0.2)

        for attempt in range(6):
            self._publish_stop()
            with self._lock:
                self._got_pose = False
            self._reset_cli.call_async(Empty.Request())
            time.sleep(0.15)

            t0 = time.time()
            while time.time() - t0 < 1.0:
                with self._lock:
                    got = self._got_pose
                if got:
                    break
                time.sleep(0.05)

            time.sleep(0.1)

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

        # Re-initialise episode state
        self._episode   += 1
        self._step       = 0
        self._ep_reward  = 0.0
        self._cmd_left   = 0.0
        self._cmd_right  = 0.0
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
