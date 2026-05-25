import os
import rclpy
from rl_env_l_shape import Gazebo4WSEnvLShape
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback 
from stable_baselines3.common.monitor import Monitor

def main():
    rclpy.init()
    raw_env = Gazebo4WSEnvLShape()
    env = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: env])    

    model_path = "brain_l_shape.zip"

    # Hỗ trợ tự động chạy nối tiếp Transfer Learning nếu tìm thấy file bộ não cũ
    if os.path.exists(model_path):
        print(f"\n[INFO] Tìm thấy bộ não '{model_path}'. Tiến hành tải dữ liệu rèn luyện tiếp tục...")
        model = PPO.load(model_path, env=vec_env, tensorboard_log="./lvl2_logs/")
    else:
        print("\n[INFO] Khởi tạo bộ não mạng nơ-ron PPO L-Shape hoàn toàn mới!")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, tensorboard_log="./lvl2_logs/")

    total_steps = 2000000
    
    # Tạo thư mục tự động lưu bản sao dự phòng
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,             # Cứ sau mỗi 100K bước lưu 1 bản sao để bảo vệ rác máy
        save_path='./checkpoints/',  
        name_prefix='brain_l_shape_backup' 
    )

    print(f"\n--- KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN FULL MAP CHỮ L ({total_steps} BƯỚC) ---")
    
    try:
        model.learn(
            total_timesteps=total_steps, 
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n[CẢNH BÁO] Tiến trình rèn luyện bị tạm dừng thủ công bằng phím bấm!")

    # Đồng bộ hóa lưu file não chính thức cuối cùng
    model.save("brain_l_shape")
    print("[THÀNH CÔNG] Đã lưu bộ não đắc đạo cuối cùng vào file 'brain_l_shape.zip'!")
    
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()