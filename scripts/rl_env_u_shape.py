import gymnasium as gym
import numpy as np
import rclpy
import threading
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from gymnasium import spaces
from std_srvs.srv import Empty
from rclpy.qos import qos_profile_sensor_data
from robot_logger import RobotLogger

class Gazebo4WSEnvUShape(gym.Env):
    def __init__(self):
        super(Gazebo4WSEnvUShape, self).__init__()
        self.node = rclpy.create_node('rl_env_u_shape')
        self._data_lock = threading.Lock()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 24 tia Lidar + 1 Ga + 2 Tọa độ local đến WP = 27
        self.observation_space = spaces.Box(
            low=np.array([0.0]*24 + [-1.0, -30.0, -30.0], dtype=np.float32),
            high=np.array([10.0]*24 + [ 1.0,  30.0,  30.0], dtype=np.float32),
            dtype=np.float32
        )

        self.steer_pub = self.node.create_publisher(Float64MultiArray, '/steering_controller/commands', 10)
        self.wheel_pub = self.node.create_publisher(Float64MultiArray, '/wheel_controller/commands', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.got_pose = False
        self.odom_sub = self.node.create_subscription(Odometry, '/ground_truth', self.odom_callback, 10)
        self.reset_client = self.node.create_client(Empty, '/reset_world')

        # --- WAYPOINTS CHỮ U (8 trạm + 2 điểm dẫn qua góc cua) ---
        # Hành lang dưới: Y=0, X 0→~8.5
        # Hành lang phải: X=10, Y 1.5→8.5
        # Hành lang trên: Y=10, X 8.5→0
        self.waypoints = [
            [ 3.0,  0.0],   # WP0: Hành lang dưới
            [ 7.0,  0.0],   # WP1: Tiếp cận góc cua 1
            [10.0,  2.5],   # WP2: Qua góc cua 1, vào hành lang phải
            [10.0,  5.0],   # WP3: Giữa hành lang phải
            [10.0,  8.5],   # WP4: Tiếp cận góc cua 2
            [ 9.0, 10.0],   # WP5: Qua góc cua 2, vào hành lang trên
            [ 4.0, 10.0],   # WP6: Giữa hành lang trên
            [ 0.0, 10.0],   # WP7: VẠCH ĐÍCH
        ]
        self.current_wp_index = 0

        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.prev_dist = np.linalg.norm(self.robot_pos - np.array(self.waypoints[0]))

        self.laser_data = np.ones(24) * 10.0
        self.current_action = [0.0, 0.0]
        self.current_step = 0
        self.current_steer_angle = 0.0
        self.goal_reached_time = None

        # --- Tracking cho logger ---
        self._episode_reward = 0.0
        self._episode_end_reason = "timeout"
        self._logger = RobotLogger(run_name="u_shape")

        self._ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._data_lock:
            self.robot_pos[0] = x
            self.robot_pos[1] = y
            self.robot_yaw = yaw
            self.got_pose = True

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        laser = np.array([np.min(ranges[i:i+15]) for i in range(0, 360, 15)])
        with self._data_lock:
            self.laser_data = laser

    def step(self, action):
        self.current_action = action
        self.current_step += 1

        # TỐC ĐỘ: [-1,1] → [-1.0, 3.0] m/s — cho phép lùi nhẹ thoát tường
        forward_speed = float(action[0]) * 2.0 + 1.0

        # LÁI CÓ LÀM MỊN: target ±0.75 rad (trong giới hạn URDF ±0.785 rad)
        # Lỗi cũ: target ±1.6 rad bị cắt vật lý → agent mất kiểm soát góc lái thực tế
        # Tốc độ xoay tối đa: 0.05 rad/step → khớp với URDF velocity=1.5 rad/s
        target_steer = float(action[1]) * 0.75
        steer_diff = target_steer - self.current_steer_angle
        self.current_steer_angle += np.clip(steer_diff, -0.05, 0.05)

        steer_msg = Float64MultiArray()
        steer_msg.data = [
            float(self.current_steer_angle),
            float(self.current_steer_angle),
            float(-self.current_steer_angle),
            float(-self.current_steer_angle)
        ]
        self.steer_pub.publish(steer_msg)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(forward_speed)] * 4
        self.wheel_pub.publish(wheel_msg)

        time.sleep(0.002)
        obs = self._get_obs()

        with self._data_lock:
            laser_snapshot = self.laser_data.copy()
            pos_snapshot = self.robot_pos.copy()
            yaw_snapshot = self.robot_yaw

        min_lidar_dist = np.min(laser_snapshot)
        done = False
        reward = 0.0

        active_target = np.array(self.waypoints[self.current_wp_index])
        current_dist = np.linalg.norm(pos_snapshot - active_target)

        if self.current_step <= 1:
            self.prev_dist = current_dist

        # --- 1. TIẾN ĐỘ & BẢO HIỂM LÙI ---
        progress = self.prev_dist - current_dist
        if min_lidar_dist < 0.6 and forward_speed < 0.0 and progress < 0:
            reward += abs(progress) * 100.0   # Miễn phạt lùi tránh tường
        else:
            reward += progress * 100.0
        self.prev_dist = current_dist

        # --- 2. PHẠT THỜI GIAN ---
        reward -= 0.05

        # --- 3. HƯỚNG NHÌN VỀ WP HIỆN TẠI ---
        angle_to_target = np.arctan2(
            active_target[1] - pos_snapshot[1],
            active_target[0] - pos_snapshot[0]
        )
        heading_error = abs(np.arctan2(
            np.sin(angle_to_target - yaw_snapshot),
            np.cos(angle_to_target - yaw_snapshot)
        ))
        reward += (np.pi - heading_error) * 0.05

        # --- 4. PHẠT LỆCH LÀN (chỉ ở đoạn thẳng, không ép ở góc cua) ---
        if self.current_wp_index in [0, 1]:            # Hành lang dưới: bám Y=0
            reward -= abs(pos_snapshot[1]) * 1.0       # 2.0→1.0: giảm phương sai return (value_loss spike)
        elif self.current_wp_index in [2, 3, 4]:       # Hành lang phải: bám X=10
            reward -= abs(pos_snapshot[0] - 10.0) * 0.75  # 1.5→0.75: giảm phương sai return
        elif self.current_wp_index in [6, 7]:          # Hành lang trên: bám Y=10
            reward -= abs(pos_snapshot[1] - 10.0) * 1.0   # 2.0→1.0: giảm phương sai return

        # --- 5. PHẠT TƯỜNG BA CẤP (sớm, mạnh, tuyến tính) ---
        # Đã giảm hệ số: 10→5, 40→20, 20→8 để cân bằng với phần thưởng tiến độ
        r_wall_soft  = 0.0
        r_wall_danger = 0.0
        r_speed_wall  = 0.0
        if min_lidar_dist < 1.2:
            r_wall_soft = -(1.2 - min_lidar_dist) * 5.0
            reward += r_wall_soft
        if min_lidar_dist < 0.7:
            r_wall_danger = -(0.7 - min_lidar_dist) * 20.0
            reward += r_wall_danger
        if min_lidar_dist < 0.5 and forward_speed > 0.5:
            r_speed_wall = -(forward_speed - 0.5) * 8.0
            reward += r_speed_wall

        # --- 6. VA CHẠM & RANH GIỚI MAP ---
        end_reason = None
        if self.current_step > 40:
            if min_lidar_dist < 0.38:
                reward -= 200.0  # -500→-200: giảm variance return, tránh value_loss spike
                end_reason = "collision"
                done = True

        if pos_snapshot[0] < -3.0 or pos_snapshot[1] < -3.0 or \
           pos_snapshot[0] > 13.0 or pos_snapshot[1] > 13.0:
            reward -= 200.0  # -500→-200: giảm variance return
            end_reason = "oob"
            done = True

        if end_reason in ("collision", "oob"):
            self._logger.log_crash(
                step=self.current_step,
                x=pos_snapshot[0], y=pos_snapshot[1], yaw=yaw_snapshot,
                speed=forward_speed, steer=self.current_steer_angle,
                laser_array=laser_snapshot,
                wp_index=self.current_wp_index
            )

        # --- 7. CHECKPOINTS ---
        r_checkpoint = 0.0
        acceptance_radius = 0.8 if self.current_wp_index == len(self.waypoints) - 1 else 1.2

        if current_dist < acceptance_radius:
            if self.current_wp_index < len(self.waypoints) - 1:
                r_checkpoint = 100.0
                reward += r_checkpoint
                self.current_wp_index += 1
                if self.current_wp_index in [2, 5, 7]:
                    print(f"[WP] Trạm {self.current_wp_index}/{len(self.waypoints)-1}", flush=True)
                active_target = np.array(self.waypoints[self.current_wp_index])
                self.prev_dist = np.linalg.norm(pos_snapshot - active_target)
            else:
                # Đích cuối
                if self.goal_reached_time is not None:
                    stop_msg = Float64MultiArray()
                    stop_msg.data = [0.0] * 4
                    self.wheel_pub.publish(stop_msg)
                    self.steer_pub.publish(stop_msg)
                    reward += 10.0
                    elapsed = time.time() - self.goal_reached_time
                    remaining = max(0.0, 5.0 - elapsed)
                    print(f"\r[ĐÍCH] Đang dừng... còn {remaining:.1f}s  ", end="", flush=True)
                    if elapsed >= 5.0:
                        reward += 2000.0 + (0.8 - current_dist) * 1000.0
                        print(f"\n>>> HOÀN THÀNH! (Sai số: {current_dist:.2f}m) <<<")
                        done = True
                elif abs(forward_speed) < 0.5:
                    self.goal_reached_time = time.time()
                    reward += 500.0
                    stop_msg = Float64MultiArray()
                    stop_msg.data = [0.0] * 4
                    self.wheel_pub.publish(stop_msg)
                    self.steer_pub.publish(stop_msg)
                    print(f"\n[ĐÍCH] Robot đã dừng! Chờ 5 giây...")
                else:
                    reward -= abs(forward_speed) * 3.0
                    reward += (0.8 - current_dist) * 50.0

        # --- 8. TIMEOUT ---
        if self.current_step >= 8000:
            end_reason = end_reason or "timeout"
            done = True

        if done:
            end_reason = end_reason or "goal"
            self._episode_end_reason = end_reason
            stop_msg = Float64MultiArray()
            stop_msg.data = [0.0] * 4
            self.wheel_pub.publish(stop_msg)
            self.steer_pub.publish(stop_msg)

        # --- LOG REWARD BREAKDOWN (mỗi 100 bước) ---
        self._episode_reward += reward
        self._logger.log_reward_step(
            step=self.current_step,
            total_reward=self._episode_reward,
            r_progress=progress * 100.0,
            r_heading=(np.pi - heading_error) * 0.05,
            r_lane=reward - progress * 100.0,   # approximation logged separately
            r_wall_soft=r_wall_soft,
            r_wall_danger=r_wall_danger,
            r_speed_wall=r_speed_wall,
            r_time_penalty=-0.05,
            r_checkpoint=r_checkpoint,
            min_lidar=min_lidar_dist,
            wp_index=self.current_wp_index,
            x=pos_snapshot[0], y=pos_snapshot[1],
            speed=forward_speed, steer=self.current_steer_angle
        )

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        # --- Lưu episode vừa kết thúc trước khi reset ---
        if self.current_step > 0:
            with self._data_lock:
                end_x, end_y = self.robot_pos[0], self.robot_pos[1]
            self._logger.log_episode(
                total_reward=self._episode_reward,
                steps=self.current_step,
                max_wp=self.current_wp_index,
                wp_total=len(self.waypoints) - 1,
                end_reason=self._episode_end_reason,
                end_x=end_x,
                end_y=end_y
            )

        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 4
        self.steer_pub.publish(stop_msg)
        self.wheel_pub.publish(stop_msg)

        # --- Reset + xác nhận spawn đúng vị trí (tối đa 3 lần thử) ---
        for _attempt in range(3):
            with self._data_lock:
                self.got_pose = False
            self.reset_client.call_async(Empty.Request())
            time.sleep(0.05)  # để message queue cũ drain trước khi nhận pose mới

            wait_time = 0.0
            while wait_time < 0.6:
                with self._data_lock:
                    if self.got_pose:
                        break
                time.sleep(0.05)
                wait_time += 0.05

            time.sleep(0.05)

            # Kiểm tra spawn hợp lệ: chỉ vị trí (reset_world không khôi phục yaw)
            with self._data_lock:
                spawn_x = self.robot_pos[0]
                spawn_y = self.robot_pos[1]

            spawn_ok = (-2.5 <= spawn_x <= 4.0 and -2.5 <= spawn_y <= 2.5)
            if spawn_ok:
                break
            print(f"[RESET] Physics explosion (x={spawn_x:.1f}, y={spawn_y:.1f}), thử lại...", flush=True)

        self.current_step = 0
        self.current_action = [0.0, 0.0]
        self.current_steer_angle = 0.0
        self.goal_reached_time = None
        self.laser_data = np.ones(24) * 10.0
        self._episode_reward = 0.0
        self._episode_end_reason = "timeout"

        self.current_wp_index = 0
        active_target = np.array(self.waypoints[self.current_wp_index])
        with self._data_lock:
            self.prev_dist = np.linalg.norm(self.robot_pos - active_target)

        return self._get_obs(), {}

    def _get_obs(self):
        with self._data_lock:
            laser = self.laser_data.copy()
            pos = self.robot_pos.copy()
            yaw = self.robot_yaw

        active_target = np.array(self.waypoints[self.current_wp_index])
        dx = active_target[0] - pos[0]
        dy = active_target[1] - pos[1]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        local_dx = cos_y * dx + sin_y * dy
        local_dy = -sin_y * dx + cos_y * dy

        return np.append(laser, [self.current_action[0], local_dx, local_dy]).astype(np.float32)

    def close(self):
        self._logger.close()
        self.node.destroy_node()