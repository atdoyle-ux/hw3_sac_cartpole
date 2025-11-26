# train_sac_dmc_cartpole.py

import os
import glob
import warnings

import gymnasium as gym
import shimmy  # registers dm_control/* Gymnasium env IDs
from gymnasium.wrappers import FlattenObservation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------
# ENV FACTORY
# ----------------------
def make_dmc_cartpole_env(seed: int, log_dir: str, idx: int = 0,
                          render_mode=None, is_eval: bool = False):
    """
    Returns a callable that creates a dm_control CartPole env, flattens obs,
    and wraps it with a Monitor that logs to CSV.
    """

    def _init():
        env = gym.make("dm_control/cartpole-balance-v0", render_mode=render_mode)
        env = FlattenObservation(env)

        # Monitor filename (per-env file for training, single file for eval)
        if is_eval:
            filename = os.path.join(log_dir, "monitor_eval.csv")
        else:
            filename = os.path.join(log_dir, f"monitor_train_{idx}.csv")

        env = Monitor(env, filename=filename)
        env.reset(seed=seed + idx)
        return env

    return _init


# ----------------------
# TRAINING FOR ONE SEED
# ----------------------
def train_for_seed(
    seed: int,
    n_envs: int,
    total_timesteps: int,
    desired_eval_interval: int,
    root_log_dir: str,
):
    """
    Train SAC for a single seed, logging:
      - training returns via Monitor CSVs in root_log_dir/seed_{seed}
      - evaluation results via EvalCallback (evaluations.npz)
    """
    print(f"\n=== Training seed {seed} ===")
    eval_freq = max(desired_eval_interval // n_envs, 1)
    seed_log_dir = os.path.join(root_log_dir, f"seed_{seed}")
    eval_log_dir = os.path.join(seed_log_dir, "eval")
    tb_log_dir = os.path.join(seed_log_dir, "tb")
    best_model_dir = os.path.join(seed_log_dir, "best_model")

    os.makedirs(seed_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Vectorized training env (SubprocVecEnv for speed)
    env_fns = [
        make_dmc_cartpole_env(seed=seed, log_dir=seed_log_dir, idx=i, render_mode=None, is_eval=False)
        for i in range(n_envs)
    ]
    train_env = SubprocVecEnv(env_fns)

    # Separate eval env (single, non-vectorized)
    eval_env = make_dmc_cartpole_env(
        seed=seed, log_dir=eval_log_dir, idx=0, render_mode=None, is_eval=True
    )()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=eval_freq,      # every N train steps (across vec env)
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=tb_log_dir,
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_cb,
        log_interval=10,
        progress_bar=False,
    )

    model.save(os.path.join(seed_log_dir, "sac_dmc_cartpole_balance"))

    train_env.close()
    eval_env.close()


# ----------------------
# LOADING TRAINING CURVES
# ----------------------
def load_training_curve_for_seed(seed: int, root_log_dir: str):
    """
    Use SB3 utilities to load training episode rewards vs timesteps
    for a given seed.

    Assumes monitor CSVs for that seed live in:
        root_log_dir/seed_{seed}
    and were created using SB3's Monitor wrapper.
    """
    seed_log_dir = os.path.join(root_log_dir, f"seed_{seed}")

    # SB3 helper: loads and merges all monitor*.csv in that folder
    df = load_results(seed_log_dir)
    if df is None or len(df) == 0:
        raise ValueError(f"No monitor data found in {seed_log_dir}")

    # x: timesteps, y: episode returns
    x, y = ts2xy(df, "timesteps")
    return x, y



# ----------------------
# LOADING EVAL CURVES
# ----------------------
def load_eval_curve_for_seed(seed: int, root_log_dir: str):
    """
    Loads EvalCallback results for a given seed (evaluations.npz) and
    returns:
      x_eval (timesteps, 1D), y_eval (mean eval return per evaluation)
    """
    seed_log_dir = os.path.join(root_log_dir, f"seed_{seed}")
    eval_log_dir = os.path.join(seed_log_dir, "eval")
    eval_file = os.path.join(eval_log_dir, "evaluations.npz")

    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"No evaluations.npz found for seed {seed} in {eval_log_dir}")

    data = np.load(eval_file)
    timesteps = data["timesteps"]         # shape (n_eval,)
    results = data["results"]             # shape (n_eval, n_eval_episodes)

    mean_per_eval = results.mean(axis=1)  # mean over episodes
    return timesteps, mean_per_eval


# ----------------------
# PLOT FUSED MEAN ± STD
# ----------------------
import matplotlib.pyplot as plt

def plot_fused_results(
    seeds,
    root_log_dir: str,
    total_timesteps: int,
    train_num_points: int = 200,
    save_path: str = "plots/sac_cartpole_mean_std.png",
):
    """
    For all seeds, load SB3-style training curves and EvalCallback curves,
    then compute mean ± std across seeds and save one plot to disk.

    x-axis: environment timesteps
      - Training: episode returns, resampled on a common grid
      - Eval: mean eval returns at all unique eval timesteps, with ±1 std band
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -------------------------
    # 1) TRAINING (episode rewards)
    # -------------------------
    x_train_grid = np.linspace(0, total_timesteps, train_num_points)
    train_y_resampled_per_seed = []

    for seed in seeds:
        x_seed, y_seed = load_training_curve_for_seed(seed, root_log_dir)
        order = np.argsort(x_seed)
        x_seed = x_seed[order]
        y_seed = y_seed[order]

        # Interpolate onto common grid; outside range we set NaN
        # (hack: interpolate then mask outside)
        y_interp = np.interp(x_train_grid, x_seed, y_seed)
        # mask outside the seed's real range
        mask_outside = (x_train_grid < x_seed[0]) | (x_train_grid > x_seed[-1])
        y_interp[mask_outside] = np.nan

        train_y_resampled_per_seed.append(y_interp)

    train_y_resampled_per_seed = np.stack(train_y_resampled_per_seed, axis=0)
    train_mean = np.nanmean(train_y_resampled_per_seed, axis=0)
    train_std = np.nanstd(train_y_resampled_per_seed, axis=0)

    # -------------------------
    # 2) EVAL (EvalCallback)
    # -------------------------
    eval_data = []
    for seed in seeds:
        x_eval, y_eval = load_eval_curve_for_seed(seed, root_log_dir)
        order = np.argsort(x_eval)
        x_eval = x_eval[order]
        y_eval = y_eval[order]
        eval_data.append((x_eval, y_eval))

    # Union of all eval timesteps across seeds
    all_eval_ts = np.unique(
        np.concatenate([x for (x, _) in eval_data], axis=0)
    )
    all_eval_ts = np.sort(all_eval_ts)

    # For each seed, place its eval returns onto the union grid
    eval_y_on_grid = []
    for (x_eval, y_eval) in eval_data:
        y_grid = np.full_like(all_eval_ts, np.nan, dtype=float)
        # map timestep -> index
        ts_to_idx = {t: i for i, t in enumerate(all_eval_ts)}
        for t, r in zip(x_eval, y_eval):
            idx = ts_to_idx[t]
            y_grid[idx] = r
        eval_y_on_grid.append(y_grid)

    eval_y_on_grid = np.stack(eval_y_on_grid, axis=0)
    eval_mean = np.nanmean(eval_y_on_grid, axis=0)
    eval_std = np.nanstd(eval_y_on_grid, axis=0)

    # -------------------------
    # 3) Plot + save
    # -------------------------
    plt.figure(figsize=(9, 5))

    # Training curve
    plt.plot(
        x_train_grid,
        train_mean,
        label="Train return (mean over seeds)",
        linewidth=2,
    )
    plt.fill_between(
        x_train_grid,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        label="Train ±1 std",
    )

    # Eval curve
    plt.plot(
        all_eval_ts,
        eval_mean,
        label="Eval return (mean over seeds)",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=4,
    )
    plt.fill_between(
        all_eval_ts,
        eval_mean - eval_std,
        eval_mean + eval_std,
        alpha=0.2,
        label="Eval ±1 std",
    )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Episode return")
    plt.title("SAC on dm_control/cartpole-balance-v0 (seeds = {0,1,2})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"[plot_fused_results] Saved plot to {save_path}")



# MAIN

if __name__ == "__main__":
    seeds = [0, 1, 2]
    n_envs = 8
    total_timesteps = 100_000
    eval_freq = 1_000
    root_log_dir = "./logs"

    # train_for_seed(...) loop as you already have it
    for seed in seeds:
        train_for_seed(
            seed=seed,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            desired_eval_interval=eval_freq,
            root_log_dir=root_log_dir,
        )

    # then fuse + save plot
    plot_fused_results(
        seeds=seeds,
        root_log_dir=root_log_dir,
        total_timesteps=total_timesteps,
        save_path="plots/sac_cartpole_mean_std.png",
    )
