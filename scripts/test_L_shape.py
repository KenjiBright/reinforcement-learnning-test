import os
import rclpy
import time
from rl_env_l_shape import Gazebo4WSEnvLShape
from stable_baselines3 import PPO

def main():
    model_path = "brain_l_shape.zip"
    print(f"\n[INFO] Đang nạp bộ não thử nghiệm L-Shape từ '{model_path}'...")

    if not os.path.exists(model_path):
        print(f"[THẤT BẠI] Không tìm thấy '{model_path}'.\n")
        return

    rclpy.init()
    env = Gazebo4WSEnvLShape()

    try:
        model = PPO.load(model_path)
        print("[THÀNH CÔNG] Đã nạp xong bộ não! Chuẩn bị lăn bánh...")
    except Exception as e:
        print(f"[THẤT BẠI] Lỗi không thể đọc file: {e}")
        env.close()
        rclpy.shutdown()
        return

    obs, _ = env.reset()
    print("\n>>> XE ĐANG TỰ ĐỘNG LÁI (INFERENCE L-SHAPE) <<<")
    print("Nhấn Ctrl+C để kết thúc.\n")

    try:
        while True:
            # deterministic=True ép AI xài kỹ năng tốt nhất nó học được (không random)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                print("--- Tái tạo vòng chạy mới ---")
                time.sleep(1.0)
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n[INFO] Đã kết thúc phiên chạy.")

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()