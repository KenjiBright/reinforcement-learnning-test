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

class Gazebo4WSEnv(gym.Env):
    def __init__(self):
        super(Gazebo4WSEnv, self).__init__()
        self.node = rclpy.create_node('rl_env_pure')
        
        self.action_space = spaces.Box(low=np.array([-1.0, -0.5]), high=np.array([1.0, 0.5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0]*10 + [-1.0, -10.0, -10.0]), 
                                            high=np.array([10.0]*10 + [ 1.0,  10.0,  10.0]), dtype=np.float32)
        
        self.steer_pub = self.node.create_publisher(Float64MultiArray, '/steering_controller/commands', 10)
        self.wheel_pub = self.node.create_publisher(Float64MultiArray, '/wheel_controller/commands', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        
        self.target_pos = np.array([5.0, 0.0]) 
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.prev_dist = 5.0
        self.laser_data = np.ones(10) * 10.0
        self.current_action = [0.0, 0.0]
        self.current_step = 0
        
        # Vô lăng thực tế chống giật bánh
        self.current_steer_angle = 0.0

        self._ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        self.laser_data = np.array([np.min(ranges[i:i+36]) for i in range(0, 360, 36)])

    def step(self, action):
        self.current_action = action
        self.current_step += 1
        
        # Điều khiển góc lái mượt mà
        target_steer = float(action[1])
        max_steer_speed = 0.03 
        steer_diff = target_steer - self.current_steer_angle
        self.current_steer_angle += np.clip(steer_diff, -max_steer_speed, max_steer_speed)
        
        steer_msg = Float64MultiArray()
        steer_msg.data = [self.current_steer_angle, self.current_steer_angle, -self.current_steer_angle, -self.current_steer_angle]
        self.steer_pub.publish(steer_msg)

        # Điều khiển tốc độ bánh xe
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(action[0]) * 2.0] * 4
        self.wheel_pub.publish(wheel_msg)

        time.sleep(0.002)

        obs = self._get_obs()
        current_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        min_lidar_dist = np.min(self.laser_data)

        if self.current_step <= 1:
            self.prev_dist = current_dist

        reward = 0.0
        done = False

        # Tính điểm tịnh tiến
        progress = self.prev_dist - current_dist
        reward += progress * 100.0
        self.prev_dist = current_dist

        # 1. Kiểm tra va chạm gạch
        if self.current_step > 50:
            if min_lidar_dist < 0.25:
                print("--- Va chạm vật lý! Reset. ---")
                done = True

        # 2. Kiểm tra chạm vòng tròn vàng
        if current_dist < 0.5:
            reward += 100.0
            print("--- TUYỆT VỜI: XE ĐÃ CHẠM ĐÍCH [5.0, 0.0]! ---")
            done = True

        # 3. ĐÃ NÂNG CẤP: Tăng giới hạn lên 2500 bước (~5 giây cuộc đời)
        if self.current_step >= 2500:
            print("--- Hết giờ (Timeout)! Xe đi quá chậm hoặc kẹt đứng yên. ---")
            done = True

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        future = self.reset_client.call_async(Empty.Request())
        time.sleep(0.5) 
        self.current_step = 0
        self.current_action = [0.0, 0.0]
        self.current_steer_angle = 0.0
        self.laser_data = np.ones(10) * 10.0
        self.robot_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([5.0, 0.0])
        self.prev_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.target_pos[0] - self.robot_pos[0]
        dy = self.target_pos[1] - self.robot_pos[1]
        cos_y = np.cos(self.robot_yaw)
        sin_y = np.sin(self.robot_yaw)
        return np.append(self.laser_data, [self.current_action[0], cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy]).astype(np.float32)

    def close(self):
        self.node.destroy_node()