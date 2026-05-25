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

class Gazebo4WSEnvLShape(gym.Env):
    def __init__(self):
        super(Gazebo4WSEnvLShape, self).__init__()
        self.node = rclpy.create_node('rl_env_l_shape')
        
        # Action Space: [Chân ga, Góc lái] trong đoạn [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space: 24 tia Lidar + 1 Vận tốc hành động + 2 Tọa độ Local tới Đích (Tổng cộng 27)
        self.observation_space = spaces.Box(
            low=np.array([0.0]*24 + [-1.0, -30.0, -30.0], dtype=np.float32), 
            high=np.array([10.0]*24 + [ 1.0,  30.0,  30.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # ROS 2 Publishers & Subscribers
        self.steer_pub = self.node.create_publisher(Float64MultiArray, '/steering_controller/commands', 10)
        self.wheel_pub = self.node.create_publisher(Float64MultiArray, '/wheel_controller/commands', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        self.got_pose = False
        self.odom_sub = self.node.create_subscription(Odometry, '/ground_truth', self.odom_callback, 10)
        self.reset_client = self.node.create_client(Empty, '/reset_world')
        
        # --- CHIẾN THUẬT VỤN BÁNH MÌ (DENSE WAYPOINTS) FULL MAP ---
        self.waypoints = [
            [3.0, 0.0],   # Trạm 1: Đi thẳng đoạn đầu
            [6.0, 0.0],   # Trạm 2: Đi thẳng đoạn giữa
            [9.0, 0.0],   # Trạm 3: Tiếp cận góc cua
            [10.0, 3.0],  # Trạm 4: Đã rẽ xong, hướng lên trục Y
            [10.0, 6.0],  # Trạm 5: Đi thẳng làn mới
            [10.0, 9.0]   # Trạm 6: VẠCH ĐÍCH CUỐI CÙNG (Yêu cầu dừng hẳn)
        ]
        self.current_wp_index = 0
        
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.prev_dist = np.linalg.norm(self.robot_pos - np.array(self.waypoints[0]))
        
        self.laser_data = np.ones(24) * 10.0
        self.current_action = [0.0, 0.0]
        self.current_step = 0
        self.current_steer_angle = 0.0

        # Chạy vòng lặp ROS 2 trong luồng riêng để không nghẽn AI
        self._ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

        print("[INFO] Đang kết nối với dịch vụ Gazebo...")
        self.reset_client.wait_for_service()
        print("[INFO] Khởi tạo môi trường L-Shape hoàn chỉnh thành công!")

    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.got_pose = True

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        # Gom 360 tia thành 24 vùng quét (mỗi vùng 15 độ) để giảm tải mạng nơ-ron
        self.laser_data = np.array([np.min(ranges[i:i+15]) for i in range(0, 360, 15)])

    def step(self, action):
        self.current_action = action
        self.current_step += 1
        
        # Vận tốc: Dải từ -3.0 m/s (số lùi) đến +7.0 m/s (số tiến)
        forward_speed = float(action[0]) * 5.0 + 2.0
        
        # ĐIỀU KHIỂN TUYỆT ĐỐI (Absolute Steering) - Trị dứt điểm bệnh đánh võng say xỉn
        self.current_steer_angle = float(action[1]) * 0.5
        
        # Gửi lệnh điều khiển góc lái xuống bộ Swerve Drive
        steer_msg = Float64MultiArray()
        steer_msg.data = [
            float(self.current_steer_angle), 
            float(self.current_steer_angle), 
            float(-self.current_steer_angle), 
            float(-self.current_steer_angle)
        ]
        self.steer_pub.publish(steer_msg)

        # Gửi lệnh vận tốc động cơ
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(forward_speed)] * 4
        self.wheel_pub.publish(wheel_msg)

        time.sleep(0.002)
        obs = self._get_obs()
        
        min_lidar_dist = np.min(self.laser_data)
        done = False
        reward = 0.0

        # Xác định trạm mục tiêu hiện tại
        active_target = np.array(self.waypoints[self.current_wp_index])
        current_dist = np.linalg.norm(self.robot_pos - active_target)

        if self.current_step <= 1:
            self.prev_dist = current_dist

        # --- BƯỚC 1: TÍNH TOÁN PHẦN THƯỞNG TIẾN ĐỘ & BẢO HIỂM LÙI ---
        progress = self.prev_dist - current_dist
        
        # Nếu xe quá sát tường nguy hiểm mà biết chủ động lùi xe cứu hộ -> Hoàn lại tiền phạt đi lùi xa đích
        if min_lidar_dist < 0.6 and forward_speed < 0.0 and progress < 0:
            reward += abs(progress) * 100.0  
            if self.current_step % 100 == 0:
                print("[Tránh né] Xe đang chủ động lùi cứu hộ thoát tường (Đã miễn phạt progress)!")
        else:
            reward += progress * 100.0
            
        self.prev_dist = current_dist
        
        # --- BƯỚC 2: PHẠT THỜI GIAN & PHẠT LỆCH LÀN ĐƯỜNG ---
        reward -= 0.05  # Phạt thời gian cơ bản khuyến khích chạy nhanh
        
        if self.current_wp_index < 3:
            # Đoạn đường đầu nằm ngang: Phạt lệch trục Y=0
            reward -= abs(self.robot_pos[1]) * 1.0
        else:
            # Đoạn đường sau thẳng đứng: Phạt lệch trục X=10.0
            reward -= abs(self.robot_pos[0] - 10.0) * 1.0

        # --- BƯỚC 3: KIỂM TRA VA CHẠM (CẢ CHÉO GÓC) & RANH GIỚI VĂNG MAP ---
        if self.current_step > 10:
            if min_lidar_dist < 0.35:  # Ngưỡng va chạm bao trọn 4 góc vuông của cản trước
                reward -= 500.0 
                print(f"[Va chạm] Đâm tường ở {min_lidar_dist:.2f}m. TRỪ 500đ và Reset!")
                done = True
                
        # Hàng rào điện tử ngăn xe lọt ranh giới hư vô của Gazebo
        if self.robot_pos[0] < -2.0 or self.robot_pos[1] < -2.0 or self.robot_pos[0] > 14.0 or self.robot_pos[1] > 14.0:
            reward -= 500.0
            print("[Lỗi] Xe văng khỏi phạm vi map! TRỪ 500đ và Reset!")
            done = True

        # --- BƯỚC 4: XỬ LÝ CHECKPOINT VỤN BÁNH MÌ & ĐÍCH CUỐI ---
        if current_dist < 0.8:
            if self.current_wp_index < len(self.waypoints) - 1:
                # Ăn trạm trung gian: Thưởng lớn và chuyển mục tiêu tiếp theo
                reward += 100.0  
                self.current_wp_index += 1
                active_target = np.array(self.waypoints[self.current_wp_index])
                self.prev_dist = np.linalg.norm(self.robot_pos - active_target)
                print(f">>> [CheckPoint] Đã qua trạm {self.current_wp_index}! Nhắm mục tiêu: {active_target} <<<")
            else:
                # TRẠM ĐÍCH CUỐI CÙNG: Yêu cầu rà phanh dừng hẳn tại tâm
                if abs(forward_speed) < 1.2:
                    reward += 1500.0  # Giải độc đắc cực đại phá vỡ mọi điểm âm
                    print(">>> TUYỆT ĐỈNH: ĐÃ TIẾN VÀO TÂM ĐÍCH VÀ PHANH XE AN TOÀN! <<<")
                    done = True
                else:
                    # Chạy quá tốc độ: Không reset ván mà ép xe rà phanh từ từ
                    reward -= 2.0  
                    reward += (0.8 - current_dist) * 20.0  # Càng bò sát rốn đích thưởng càng đậm
                    if self.current_step % 100 == 0:
                        print(f"[Đích cuối] Đang ở vùng đích nhưng chạy quá nhanh ({forward_speed:.1f} m/s). Hãy phanh xe!")

        # --- BƯỚC 5: TIMEOUT (GIỚI HẠN BƯỚC CHẠY TẬP LÁI) ---
        if self.current_step >= 10000:
            print("[Timeout] Hết thời gian ván chơi.")
            done = True

        if done:
            stop_msg = Float64MultiArray()
            stop_msg.data = [0.0] * 4
            self.wheel_pub.publish(stop_msg)
            self.steer_pub.publish(stop_msg)

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 4
        self.steer_pub.publish(stop_msg)
        self.wheel_pub.publish(stop_msg)
        time.sleep(0.2)
        
        self.got_pose = False
        self.reset_client.call_async(Empty.Request())
        
        wait_time = 0.0
        while not self.got_pose and wait_time < 2.0:
            time.sleep(0.1)
            wait_time += 0.1
            
        time.sleep(0.3) 
        
        self.current_step = 0
        self.current_action = [0.0, 0.0]
        self.current_steer_angle = 0.0
        self.laser_data = np.ones(24) * 10.0
        
        # Đưa trạm waypoint quay về điểm xuất phát số 0
        self.current_wp_index = 0
        active_target = np.array(self.waypoints[self.current_wp_index])
        self.prev_dist = np.linalg.norm(self.robot_pos - active_target)
        
        return self._get_obs(), {}

    def _get_obs(self):
        active_target = np.array(self.waypoints[self.current_wp_index])
        dx = active_target[0] - self.robot_pos[0]
        dy = active_target[1] - self.robot_pos[1]
        
        # Ma trận xoay chuyển đổi tọa độ Global sang góc nhìn la bàn Local của xe
        cos_y = np.cos(self.robot_yaw)
        sin_y = np.sin(self.robot_yaw)
        local_dx = cos_y * dx + sin_y * dy
        local_dy = -sin_y * dx + cos_y * dy
        
        return np.append(self.laser_data, [self.current_action[0], local_dx, local_dy]).astype(np.float32)

    def close(self):
        self.node.destroy_node()
