import os
import rclpy
import time
from rl_env_lvl1 import Gazebo4WSEnvLvl1
from stable_baselines3 import PPO

def main():
    model_path = "brain_lvl1.zip"
    print(f"\n[INFO] Đang nạp bộ não thử nghiệm LEVEL 1 từ '{model_path}'...")

    if not os.path.exists(model_path):
        print(f"[THẤT BẠI] Không tìm thấy '{model_path}'. Hãy chạy 'python3 train_lvl1.py' trước!\n")
        return

    rclpy.init()
    env = Gazebo4WSEnvLvl1()

    try:
        model = PPO.load(model_path)
        print("[THÀNH CÔNG] Đã nạp xong bộ não! Chuẩn bị lăn bánh...")
    except Exception as e:
        print(f"[THẤT BẠI] Lỗi không thể đọc file: {e}")
        env.close()
        rclpy.shutdown()
        return

    obs, _ = env.reset()
    print("\n>>> XE ĐANG TỰ ĐỘNG LÁI (INFERENCE LEVEL 1) <<<")
    print("Nhấn Ctrl+C để kết thúc.\n")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                # Xe đỗ xong sẽ đứng yên tại tâm vàng 2 giây để bạn ngắm
                time.sleep(2.0)
                print("--- Tái tạo vòng chạy mới ---")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\n[INFO] Đã kết thúc phiên chạy.")

    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()