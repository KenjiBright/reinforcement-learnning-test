"""
train_4wd.py  –  PPO + VecNormalize training for the 4WD robot
================================================================
Usage
    python3 train_4wd.py --level 1          # fresh or auto-resume
    python3 train_4wd.py --level 2 --steps 1500000

Curriculum
    Level 1 (corridor)  →  500 K steps
    Level 2 (L-shape)   →  1 M  steps
    Level 3 (U-shape)   →  2 M  steps
"""

import os
import argparse
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from rl_env_4wd import Gazebo4WDEnv

DEFAULTS = {
    1: {"steps": 500_000,   "lr": 3e-4},
    2: {"steps": 1_000_000, "lr": 3e-4},
    3: {"steps": 2_000_000, "lr": 1e-4},
}


class VecNormSaveCallback(BaseCallback):
    """Saves VecNormalize running stats alongside every checkpoint."""

    def __init__(self, venv: VecNormalize, save_dir: str, save_freq: int):
        super().__init__()
        self._venv = venv
        self._dir  = save_dir
        self._freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._freq == 0:
            self._venv.save(os.path.join(self._dir, f"vecnorm_{self.num_timesteps}.pkl"))
            self._venv.save(os.path.join(self._dir, "vecnorm_latest.pkl"))
        return True


def main():
    parser = argparse.ArgumentParser(description="Train 4WD robot with PPO")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=0,
                        help="Total timesteps (0 = use level default)")
    args = parser.parse_args()

    level       = args.level
    total_steps = args.steps if args.steps > 0 else DEFAULTS[level]["steps"]
    lr          = DEFAULTS[level]["lr"]

    model_path     = f"brain_lv{level}_4wd.zip"
    ckpt_dir       = f"./checkpoints_4wd/lv{level}/"
    log_dir        = f"./logs_4wd/lv{level}/"
    vecnorm_latest = os.path.join(ckpt_dir, "vecnorm_latest.pkl")
    ent_coef       = 0.0 if level == 1 else 0.01
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    rclpy.init()

    raw_env = Gazebo4WDEnv(level=level)
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])

    if os.path.exists(model_path) and os.path.exists(vecnorm_latest):
        print(f"\n[INFO] Resuming Level {level} from {model_path} …")
        vec_env = VecNormalize.load(vecnorm_latest, vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True
        model = PPO.load(
            model_path, env=vec_env,
            tensorboard_log=log_dir,
            custom_objects={
                "learning_rate": lr,
                "n_steps"      : 2048,
                "batch_size"   : 64,
                "n_epochs"     : 10,
                "ent_coef"     : ent_coef,
                "vf_coef"      : 0.5,
                "clip_range"   : 0.2,
                "clip_range_vf": None,
                "max_grad_norm": 0.5,
            },
        )
        print(f"[INFO] Continuing from timestep {model.num_timesteps:,}")
    else:
        print(f"\n[INFO] Fresh Level {level} training — {total_steps:,} steps")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
            gamma=0.99,
        )
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate  = lr,
            n_steps        = 2048,
            batch_size     = 64,
            n_epochs       = 10,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            clip_range_vf  = None,
            ent_coef       = ent_coef,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            verbose        = 1,
            device         = "cpu",
            tensorboard_log= log_dir,
            policy_kwargs  = {"net_arch": [128, 128]},
        )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = ckpt_dir,
        name_prefix = f"brain_lv{level}_4wd",
    )
    vecnorm_cb = VecNormSaveCallback(
        venv      = vec_env,
        save_dir  = ckpt_dir,
        save_freq = 50_000,
    )

    print(f"[INFO] Level {level} | {total_steps:,} steps | lr={lr}")
    print(f"[INFO] Press Ctrl+C at any time — progress will be auto-saved.\n")
    try:
        model.learn(
            total_timesteps     = total_steps,
            reset_num_timesteps = False,
            callback            = [checkpoint_cb, vecnorm_cb],
            progress_bar        = False,
        )
        print(f"\n[INFO] Training complete.")
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted at timestep {model.num_timesteps:,} — saving...")
    finally:
        model.save(model_path)
        vec_env.save(vecnorm_latest)
        print(f"[INFO] Saved: {model_path}  +  {vecnorm_latest}")
        vec_env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
