import os
import rclpy
from rl_env_u_shape import Gazebo4WSEnvUShape
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

def main():
    rclpy.init()
    raw_env = Gazebo4WSEnvUShape()
    env = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: env])    

    model_path = "brain_u_shape.zip"

    # LR giảm: 2.5e-4→1e-4 để ổn định value_loss đang dao động mạnh
    fixed_lr = 1e-4

    if os.path.exists(model_path):
        print(f"\n[INFO] Đã tìm thấy '{model_path}'! Tiến hành rèn luyện tiếp Map Chữ U...")
        model = PPO.load(model_path, env=vec_env, tensorboard_log="./lvl3_logs/",
                         custom_objects={
                             "n_steps": 2048,
                             "batch_size": 128,
                             "ent_coef": 0.003,     # 0.0→0.003: tái kích hoạt khám phá — ent_coef=0 gây frozen policy (clip_fraction=0, approx_kl≈0)
                             "learning_rate": fixed_lr,
                             "vf_coef": 0.5,        # 0.75→0.5: giảm biên độ value update để hạn chế spike
                             "n_epochs": 10,     # 15→10: value_loss=16400 spike do quá nhiều gradient steps
                             "clip_range_vf": None, # BUG FIX: clip_range_vf=0.2 absolute đóng băng value net (±0.2 vs returns~14000)
                         })
        # Reset log_std trực tiếp: std đang kẹt ở 5.5, mất 2.4M bước để giảm tự nhiên
        # std=e^0.5 ≈ 1.65: đủ khám phá nhưng cho phép điều khiển mịn ở góc cua
        import torch
        with torch.no_grad():
            model.policy.log_std.data.fill_(0.5)
        print(f"[INFO] Đã reset log_std → std={torch.exp(model.policy.log_std).mean().item():.2f}")
    else:
        print("\n[CẢNH BÁO] Không tìm thấy 'brain_u_shape.zip'! Tạo bộ não mới...")
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            learning_rate=fixed_lr,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.0,
            vf_coef=0.75,
            n_epochs=10,
            tensorboard_log="./lvl3_logs/"
        )

    # 2M bước: std reset về 1.65, cần thêm bước để ổn định sau khi reset log_std
    total_steps = 2000000

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./checkpoints/',
        name_prefix='brain_u_shape_backup'
    )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./checkpoints/best_u_shape/',
        log_path='./lvl3_logs/eval/',
        eval_freq=50000,       # 20K→50K: eval ít hơn 2.5× → FPS ổn định hơn
        n_eval_episodes=2,     # 5→2: mỗi eval nhanh hơn 2.5×
        deterministic=True,
        render=False,
    )
    callbacks = [checkpoint_callback, eval_callback]

    print(f"\n--- BẮT ĐẦU CHINH PHỤC BẢN ĐỒ CHỮ U ({total_steps:,} BƯỚC | ~{total_steps/170/3600:.1f}h ước tính) ---")
    
    try:
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=False,
            callback=callbacks
        )
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng huấn luyện thủ công bằng phím tắt!")

    model.save("brain_u_shape")
    print("[THÀNH CÔNG] Đã lưu bộ não đắc đạo cuối cùng vào 'brain_u_shape.zip'")
    
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()