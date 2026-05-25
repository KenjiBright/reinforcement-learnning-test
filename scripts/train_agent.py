import rclpy
from rl_env import Gazebo4WSEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    rclpy.init()
    env = Gazebo4WSEnv()
    check_env(env, warn=True)
    vec_env = DummyVecEnv([lambda: env])

    print("Khởi tạo mạng AI mới (Blank Slate)...")
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_logs/")

    try:
        model.learn(total_timesteps=100000)
    except KeyboardInterrupt:
        print("Đã dừng!")

    model.save("ppo_car_4ws_pure")
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()