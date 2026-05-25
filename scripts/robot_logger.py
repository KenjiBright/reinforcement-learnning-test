"""
Robot Training Logger — ghi dữ liệu chẩn đoán ra CSV để phân tích sau.

Tạo 3 file trong thư mục logs/<run_id>/:
  episode_log.csv   — 1 dòng/episode: reward, steps, waypoint đạt được, lý do kết thúc
  crash_log.csv     — 1 dòng/crash: vị trí, tốc độ, lidar, hướng khi đâm
  reward_log.csv    — mẫu mỗi 100 bước: breakdown từng thành phần reward

Cách dùng trong rl_env_u_shape.py:
    from robot_logger import RobotLogger
    self.logger = RobotLogger(run_name="u_shape")

    # Trong step():
    self.logger.log_reward_step(episode, step, components_dict)
    self.logger.log_crash(episode, step, x, y, speed, min_lidar, sector, wp_index)

    # Cuối episode (trong reset()):
    self.logger.log_episode(episode, total_reward, steps, max_wp, reason)
"""

import csv
import os
import time
import numpy as np
from datetime import datetime


class RobotLogger:
    def __init__(self, run_name: str = "u_shape", log_dir: str = "./robot_logs"):
        run_id = f"{run_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
        self.out_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.out_dir, exist_ok=True)

        # --- Episode log ---
        self._ep_path = os.path.join(self.out_dir, "episode_log.csv")
        self._ep_file = open(self._ep_path, "w", newline="", buffering=1)
        self._ep_writer = csv.writer(self._ep_file)
        self._ep_writer.writerow([
            "episode", "total_reward", "steps",
            "max_wp_reached", "wp_total",
            "end_reason",          # collision | timeout | oob | goal
            "end_x", "end_y",
            "wall_time_s"
        ])

        # --- Crash log ---
        self._crash_path = os.path.join(self.out_dir, "crash_log.csv")
        self._crash_file = open(self._crash_path, "w", newline="", buffering=1)
        self._crash_writer = csv.writer(self._crash_file)
        self._crash_writer.writerow([
            "episode", "step",
            "x", "y", "yaw_deg",
            "speed_mps", "steer_rad",
            "min_lidar_m",
            "closest_sector_deg",   # 0=front, 90=left, 180=rear, 270=right
            "wp_index",
            "wall_time_s"
        ])

        # --- Reward breakdown log (sampled) ---
        self._rew_path = os.path.join(self.out_dir, "reward_log.csv")
        self._rew_file = open(self._rew_path, "w", newline="", buffering=1)
        self._rew_writer = csv.writer(self._rew_file)
        self._rew_writer.writerow([
            "episode", "step",
            "total_reward",
            "r_progress",
            "r_heading",
            "r_lane",
            "r_wall_soft",
            "r_wall_danger",
            "r_speed_wall",
            "r_time_penalty",
            "r_checkpoint",
            "min_lidar_m",
            "wp_index",
            "x", "y",
            "speed_mps", "steer_rad"
        ])

        self._episode_count = 0
        self._start_time = time.time()
        self._sample_every = 100   # log reward breakdown every N steps

        print(f"[Logger] Đang ghi log vào: {self.out_dir}")

    # ------------------------------------------------------------------
    def log_episode(self, total_reward: float, steps: int,
                    max_wp: int, wp_total: int,
                    end_reason: str, end_x: float, end_y: float):
        """Gọi ở cuối mỗi episode (trong reset())."""
        self._episode_count += 1
        self._ep_writer.writerow([
            self._episode_count,
            round(total_reward, 3),
            steps,
            max_wp,
            wp_total,
            end_reason,
            round(end_x, 3),
            round(end_y, 3),
            round(time.time() - self._start_time, 1)
        ])

    # ------------------------------------------------------------------
    def log_crash(self, step: int,
                  x: float, y: float, yaw: float,
                  speed: float, steer: float,
                  laser_array: np.ndarray,
                  wp_index: int):
        """Gọi ngay khi phát hiện va chạm hoặc OOB."""
        min_idx = int(np.argmin(laser_array))
        min_dist = float(laser_array[min_idx])
        # Sector 0 = phía trước, tăng dần theo chiều kim đồng hồ (mỗi bin = 15°)
        sector_deg = min_idx * 15
        self._crash_writer.writerow([
            self._episode_count,
            step,
            round(x, 3), round(y, 3),
            round(np.degrees(yaw), 1),
            round(speed, 3),
            round(steer, 3),
            round(min_dist, 3),
            sector_deg,
            wp_index,
            round(time.time() - self._start_time, 1)
        ])

    # ------------------------------------------------------------------
    def log_reward_step(self, step: int,
                        total_reward: float,
                        r_progress: float,
                        r_heading: float,
                        r_lane: float,
                        r_wall_soft: float,
                        r_wall_danger: float,
                        r_speed_wall: float,
                        r_time_penalty: float,
                        r_checkpoint: float,
                        min_lidar: float,
                        wp_index: int,
                        x: float, y: float,
                        speed: float, steer: float):
        """Gọi mỗi bước — tự lọc theo sample_every."""
        if step % self._sample_every != 0:
            return
        self._rew_writer.writerow([
            self._episode_count, step,
            round(total_reward, 3),
            round(r_progress, 3),
            round(r_heading, 3),
            round(r_lane, 3),
            round(r_wall_soft, 3),
            round(r_wall_danger, 3),
            round(r_speed_wall, 3),
            round(r_time_penalty, 3),
            round(r_checkpoint, 3),
            round(min_lidar, 3),
            wp_index,
            round(x, 3), round(y, 3),
            round(speed, 3), round(steer, 3)
        ])

    # ------------------------------------------------------------------
    def close(self):
        self._ep_file.close()
        self._crash_file.close()
        self._rew_file.close()
        print(f"[Logger] Đã lưu {self._episode_count} episodes → {self.out_dir}")
