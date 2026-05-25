"""
train_v2.py  –  Clean PPO + VecNormalize training (v2)
========================================================
Usage
    python3 train_v2.py --level 1          # fresh Level-1 run
    python3 train_v2.py --level 1          # auto-resumes if checkpoint exists
    python3 train_v2.py --level 3 --steps 3000000

Curriculum workflow
    1. Train Level 1 until ~400K steps (robot learns straight driving)
    2. Train Level 2 until ~800K steps (L-shape corner)
    3. Train Level 3 until ~2M  steps  (U-shape both corners)

Key design improvements over v1
    ✅ VecNormalize from day 1 → no more -15 000/ep value_loss spikes
    ✅ Standard PPO defaults (no manual vf_coef/ent_coef fighting)
    ✅ VecNormalize stats saved alongside every model checkpoint
    ✅ Unified for all three levels — one script, one codebase
"""

import os
import argparse
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from rl_env_v2 import Gazebo4WSEnvV2

# ─────────────────────────────────────────────
# Defaults per level
# ─────────────────────────────────────────────
DEFAULTS = {
    1: {"steps": 500_000,   "lr": 3e-4},
    2: {"steps": 1_000_000, "lr": 3e-4},
    3: {"steps": 2_000_000, "lr": 1e-4},
}


# ─────────────────────────────────────────────
# Callback: save VecNormalize stats at each checkpoint
# ─────────────────────────────────────────────
class VecNormSaveCallback(BaseCallback):
    """Saves the VecNormalize running stats alongside every checkpoint."""

    def __init__(self, venv: VecNormalize, save_dir: str, save_freq: int):
        super().__init__()
        self._venv     = venv
        self._dir      = save_dir
        self._freq     = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._freq == 0:
            path = os.path.join(self._dir, f"vecnorm_{self.num_timesteps}.pkl")
            self._venv.save(path)
            # Always keep a "latest" copy for easy resume
            self._venv.save(os.path.join(self._dir, "vecnorm_latest.pkl"))
        return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train 4WS robot — v2 clean rewrite")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3],
                        help="Maze level: 1=corridor, 2=L-shape, 3=U-shape")
    parser.add_argument("--steps", type=int, default=0,
                        help="Total timesteps (0 = use level default)")
    args = parser.parse_args()

    level = args.level
    total_steps = args.steps if args.steps > 0 else DEFAULTS[level]["steps"]
    lr          = DEFAULTS[level]["lr"]

    # Paths
    model_path  = f"brain_lv{level}_v2.zip"
    vecnorm_latest = os.path.join(f"checkpoints_v2/lv{level}", "vecnorm_latest.pkl")
    log_dir     = f"./logs_v2/lv{level}/"
    ckpt_dir    = f"./checkpoints_v2/lv{level}/"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    rclpy.init()

    # ── Build training env ──
    raw_env   = Gazebo4WSEnvV2(level=level)
    mon_env   = Monitor(raw_env)
    vec_env   = DummyVecEnv([lambda: mon_env])

    # ── Load or create ──
    if os.path.exists(model_path) and os.path.exists(vecnorm_latest):
        print(f"\n[INFO] Resuming Level {level} from {model_path} + vecnorm stats...")
        vec_env = VecNormalize.load(vecnorm_latest, vec_env)
        vec_env.training   = True
        vec_env.norm_reward = True

        model = PPO.load(
            model_path, env=vec_env,
            tensorboard_log=log_dir,
            custom_objects={
                "learning_rate": lr,
                "n_steps"      : 2048,
                "batch_size"   : 64,
                "n_epochs"     : 10,
                "ent_coef"     : 0.01,
                "vf_coef"      : 0.5,
                "clip_range"   : 0.2,
                "clip_range_vf": None,   # explicit: no value-function clipping
                "max_grad_norm": 0.5,
            },
        )
        print(f"[INFO] Continuing from timestep {model.num_timesteps:,}")

    else:
        print(f"\n[INFO] Fresh Level {level} training — {total_steps:,} steps")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate  = lr,
            n_steps        = 2048,
            batch_size     = 64,
            n_epochs       = 10,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            clip_range_vf  = None,
            ent_coef       = 0.01,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            verbose        = 1,
            device         = "cpu",
            tensorboard_log= log_dir,
            policy_kwargs  = {"net_arch": [128, 128]},
        )

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = ckpt_dir,
        name_prefix = f"brain_lv{level}_v2",
    )
    vecnorm_cb = VecNormSaveCallback(
        venv      = vec_env,
        save_dir  = ckpt_dir,
        save_freq = 50_000,
    )

    # ── Train ──
    print(f"[INFO] Level {level} | {total_steps:,} total steps | lr={lr}")
    model.learn(
        total_timesteps    = total_steps,
        reset_num_timesteps= False,
        callback           = [checkpoint_cb, vecnorm_cb],
        progress_bar       = False,
    )

    # ── Save final ──
    model.save(model_path)
    vec_env.save(vecnorm_latest)
    print(f"\n[INFO] Saved: {model_path}  +  {vecnorm_latest}")

    vec_env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
