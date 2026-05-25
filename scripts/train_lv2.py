import os
import rclpy
from rl_env_lv2 import Gazebo4WSEnvLvl2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback 

def main():
    rclpy.init()
    env = Gazebo4WSEnvLvl2()
    check_env(env, warn=True)
    vec_env = DummyVecEnv([lambda: env])

    model_path = "brain_lvl2.zip"

    if os.path.exists(model_path):
        print(f"\n[INFO] Nạp bộ não '{model_path}' để rèn luyện tiếp Level 2...")
        model = PPO.load(model_path, env=vec_env, tensorboard_log="./lvl2_logs/")
    else:
        print("\n[INFO] Tạo bộ não Level 2 hoàn toàn mới!")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, tensorboard_log="./lvl2_logs/")

    # Đẩy tổng số bước lên 2.5 triệu
    total_steps = 2500000
    
    # Tự động khởi tạo thư mục và cấu hình lưu bộ não dự phòng
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,           # Cứ sau 100.000 bước sẽ lưu 1 lần
        save_path='./checkpoints/',  # Thư mục lưu trữ
        name_prefix='brain_lvl2_backup' # ĐÃ SỬA: Đổi thành backup của lvl2
    )

    print(f"\n--- Bắt đầu huấn luyện LEVEL 2 ({total_steps} bước) ---")
    print("[INFO] Hệ thống sẽ tự động lưu dự phòng vào thư mục './checkpoints/' mỗi 50.000 bước.")
    
    try:
        # Nhét callback vào tiến trình học
        model.learn(
            total_timesteps=total_steps, 
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n[CẢNH BÁO] Đã dừng huấn luyện thủ công!")

    # ĐÃ SỬA: Lưu thành brain_lvl2
    model.save("brain_lvl2")
    print("[THÀNH CÔNG] Đã lưu bộ não chính thức cuối cùng vào 'brain_lvl2.zip'")
    
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()