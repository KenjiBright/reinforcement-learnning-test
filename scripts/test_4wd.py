"""
test_4wd.py  –  Inference test for the trained 4WD robot
==========================================================
Usage
    python3 test_4wd.py --level 1              # run Level-1 model forever
    python3 test_4wd.py --level 1 --episodes 5 # run 5 episodes then exit
"""

import os

# Prevent ROS2 DDS multicast "Network is unreachable" after long runs
os.environ.setdefault('ROS_LOCALHOST_ONLY', '1')
os.environ.setdefault('ROS_DOMAIN_ID', '42')

import argparse
import time
import rclpy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_4wd import Gazebo4WDEnv


def main():
    parser = argparse.ArgumentParser(description="Test 4WD robot — inference mode")
    parser.add_argument("--level",    type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run (0 = run until Ctrl+C)")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to model zip (default: brain_lv<level>_4wd.zip)")
    parser.add_argument("--vecnorm",  type=str, default=None,
                        help="Path to VecNormalize pkl (default: checkpoints_4wd/lv<level>/vecnorm_latest.pkl)")
    args = parser.parse_args()

    level = args.level

    model_path     = args.model   if args.model   else f"brain_lv{level}_4wd.zip"
    vecnorm_path   = args.vecnorm if args.vecnorm else os.path.join(f"./checkpoints_4wd/lv{level}", "vecnorm_latest.pkl")

    # ── Validate files ──
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print(f"        Run  python3 train_4wd.py --level {level}  first.")
        return
    if not os.path.exists(vecnorm_path):
        print(f"[ERROR] VecNormalize stats not found: {vecnorm_path}")
        print(f"        Run  python3 train_4wd.py --level {level}  first.")
        return

    rclpy.init()

    # ── Build env (must match training setup) ──
    raw_env = Gazebo4WDEnv(level=level)
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training    = False   # freeze running stats
    vec_env.norm_reward = False   # don't normalise reward during inference

    # ── Load model ──
    model = PPO.load(model_path, env=vec_env, device="cpu")
    print(f"\n[INFO] Loaded  {model_path}  (timestep {model.num_timesteps:,})")
    print(f"[INFO] Level {level} — {'run forever' if args.episodes == 0 else f'{args.episodes} episodes'}")
    print("[INFO] Press Ctrl+C to stop.\n")

    # ── Run loop ──
    obs          = vec_env.reset()
    ep           = 0
    ep_reward    = 0.0
    results      = []   # (episode, reward, end_reason)

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])

            if done[0]:
                reason = info[0].get("end_reason", "?")
                wp     = info[0].get("wp", "?")
                ep    += 1
                results.append((ep, ep_reward, reason, wp))
                print(
                    f"[Ep {ep:4d}]  {reason.upper():<10}  "
                    f"wp={wp}/{raw_env.cfg['goal_idx']}  "
                    f"R={ep_reward:+.0f}",
                    flush=True,
                )
                ep_reward = 0.0

                time.sleep(1.5)    # pause so you can see the final position

                if args.episodes > 0 and ep >= args.episodes:
                    break

                obs = vec_env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    # ── Summary ──
    if results:
        goals   = sum(1 for r in results if r[2] == "goal")
        crashes = sum(1 for r in results if r[2] == "collision")
        oobs    = sum(1 for r in results if r[2] == "oob")
        timeouts= sum(1 for r in results if r[2] == "timeout")
        avg_r   = sum(r[1] for r in results) / len(results)
        print(f"\n{'─'*50}")
        print(f"  Episodes : {len(results)}")
        print(f"  Goals    : {goals}  ({100*goals/len(results):.0f}%)")
        print(f"  Crashes  : {crashes}")
        print(f"  OOB      : {oobs}")
        print(f"  Timeouts : {timeouts}")
        print(f"  Avg R    : {avg_r:+.1f}")
        print(f"{'─'*50}\n")

    vec_env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
