import os
import rclpy
from rl_env_lvl1 import Gazebo4WSEnvLvl1
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
# 1. THÊM MỚI: Thư viện callback để tự động lưu dự phòng
from stable_baselines3.common.callbacks import CheckpointCallback 

def main():
    rclpy.init()
    env = Gazebo4WSEnvLvl1()
    check_env(env, warn=True)
    vec_env = DummyVecEnv([lambda: env])

    model_path = "brain_lvl1.zip"

    if os.path.exists(model_path):
        print(f"\n[INFO] Nạp bộ não '{model_path}' để rèn luyện tiếp Level 1...")
        model = PPO.load(model_path, env=vec_env, tensorboard_log="./lvl1_logs/")
    else:
        print("\n[INFO] Tạo bộ não Level 1 hoàn toàn mới (Timeout: 2500 bước)!")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, tensorboard_log="./lvl1_logs/")

    # 2. ĐÃ SỬA: Đẩy tổng số bước lên 1 triệu
    total_steps = 1000000 
    
    # 3. THÊM MỚI: Tự động khởi tạo thư mục và cấu hình lưu bộ não dự phòng
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,           # Cứ sau 50.000 bước sẽ lưu 1 lần
        save_path='./checkpoints/', # Thư mục lưu trữ
        name_prefix='brain_lvl1_backup'
    )

    print(f"\n--- Bắt đầu huấn luyện LEVEL 1 ({total_steps} bước) ---")
    print("[INFO] Hệ thống sẽ tự động lưu dự phòng vào thư mục './checkpoints/' mỗi 50.000 bước.")
    
    try:
        # 4. ĐÃ SỬA: Nhét callback vào tiến trình học
        model.learn(
            total_timesteps=total_steps, 
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n[CẢNH BÁO] Đã dừng huấn luyện thủ công!")

    model.save("brain_lvl1")
    print("[THÀNH CÔNG] Đã lưu bộ não chính thức cuối cùng vào 'brain_lvl1.zip'")
    
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()