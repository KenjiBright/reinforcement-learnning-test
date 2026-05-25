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

class Gazebo4WSEnvLvl2(gym.Env):
    def __init__(self):
        super(Gazebo4WSEnvLvl2, self).__init__()
        self.node = rclpy.create_node('rl_env_lvl2')
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Mở rộng 24 tia Lidar + 3 biến trạng thái = 27 đầu vào
        self.observation_space = spaces.Box(low=np.array([0.0]*24 + [-1.0, -30.0, -30.0]), 
                                            high=np.array([10.0]*24 + [ 1.0,  30.0,  30.0]), dtype=np.float32)
        
        self.steer_pub = self.node.create_publisher(Float64MultiArray, '/steering_controller/commands', 10)
        self.wheel_pub = self.node.create_publisher(Float64MultiArray, '/wheel_controller/commands', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.got_pose = False
        self.odom_sub = self.node.create_subscription(Odometry, '/ground_truth', self.odom_callback, 10)
        
        self.reset_client = self.node.create_client(Empty, '/reset_world')
        
        # MỤC TIÊU LEVEL 2 LÀ 20 MÉT
        self.target_pos = np.array([20.0, 0.0]) 
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.prev_dist = 20.0
        self.laser_data = np.ones(24) * 10.0 # Cập nhật cho 24 tia
        self.current_action = [0.0, 0.0]
        self.current_step = 0
        self.current_steer_angle = 0.0

        self._ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

        print("[INFO] Đang kết nối với dịch vụ Gazebo...")
        self.reset_client.wait_for_service()
        print("[INFO] Hệ thống sẵn sàng. Chào mừng tới LEVEL 2: Né Vật Cản & Đỗ Xe 20m!")

    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if not getattr(self, 'got_pose', False): self.got_pose = True

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        # Chia 360 độ thành 24 vùng, mỗi vùng 15 độ
        self.laser_data = np.array([np.min(ranges[i:i+15]) for i in range(0, 360, 15)])

    def step(self, action):
        self.current_action = action
        self.current_step += 1
        
        # Chân ga có phanh như cũ
        forward_speed = float(action[0]) * 12.5 + 7.5
        
        target_steer = float(action[1]) * 0.5
        max_steer_speed = 0.15 
        steer_diff = target_steer - self.current_steer_angle
        self.current_steer_angle += np.clip(steer_diff, -max_steer_speed, max_steer_speed)
        
        steer_msg = Float64MultiArray()
        steer_msg.data = [self.current_steer_angle, self.current_steer_angle, -self.current_steer_angle, -self.current_steer_angle]
        self.steer_pub.publish(steer_msg)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [forward_speed] * 4
        self.wheel_pub.publish(wheel_msg)

        time.sleep(0.002)

        obs = self._get_obs()
        current_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        min_lidar_dist = np.min(self.laser_data)

        # --- ĐÃ SỬA: Tính khoảng cách theo Trục X (tiến lên) thay vì đường thẳng ---
        current_dist_x = self.target_pos[0] - self.robot_pos[0]
        
        if self.current_step <= 1:
            self.prev_dist = current_dist_x

        reward = 0.0
        done = False

        # Thưởng khi tiến về phía trước theo trục X
        progress = self.prev_dist - current_dist_x
        reward += progress * 100.0
        self.prev_dist = current_dist_x
        
        # Giảm trừ điểm thời gian để nó không cuống
        reward -= 0.05

        if min_lidar_dist < 0.6:
            reward -= (0.6 - min_lidar_dist) * 2.0

        if self.current_step > 10:
            if min_lidar_dist < 0.26:
                reward -= 100.0 
                print(f"[Va chạm] Đâm tường/Vật cản ở {min_lidar_dist:.2f}m. TRỪ 100đ!")
                done = True

        # ĐỖ XE CHÍNH XÁC (Dưới 0.4m và phanh nhẹ)
        if current_dist < 0.4:
            if abs(forward_speed) < 2.0:
                reward += 300.0 
                print(">>> XUẤT SẮC: ĐÃ VƯỢT VẬT CẢN VÀ ĐỖ XE AN TOÀN TẠI TÂM ĐÍCH 20M! <<<")
                done = True
            else:
                reward += 0.5 

        if self.robot_pos[0] > 20.4:
             reward -= 100.0
             print("[Lỗi] Trượt quá đích, đâm tường sau! TRỪ 100đ!")
             done = True

        # Timeout dài hơn để đi hết 20m
        if self.current_step >= 10000:
            print("[Timeout] Hết thời gian. Reset!")
            done = True

        if done:
            stop_msg = Float64MultiArray()
            stop_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.wheel_pub.publish(stop_msg)
            self.steer_pub.publish(stop_msg)

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0, 0.0, 0.0, 0.0]
        self.steer_pub.publish(stop_msg)
        self.wheel_pub.publish(stop_msg)
        time.sleep(0.2)
        future = self.reset_client.call_async(Empty.Request())
        wait_time = 0.0
        while not future.done() and wait_time < 2.0:
            time.sleep(0.1)
            wait_time += 0.1
        time.sleep(0.5)
        self.current_step = 0
        self.current_action = [0.0, 0.0]
        self.current_steer_angle = 0.0
        self.laser_data = np.ones(24) * 10.0 # Đã sửa thành 24 tia
        self.robot_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([20.0, 0.0])
        self.prev_dist = 20.0 # Thiết lập lại khoảng cách ban đầu
        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.target_pos[0] - self.robot_pos[0]
        dy = self.target_pos[1] - self.robot_pos[1]
        cos_y = np.cos(self.robot_yaw)
        sin_y = np.sin(self.robot_yaw)
        return np.append(self.laser_data, [self.current_action[0], cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy]).astype(np.float32)

    def close(self):
        self.node.destroy_node()