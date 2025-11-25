# train_sac_dmc_cartpole.py
import gymnasium as gym
import shimmy  # registers dm_control/* Gymnasium env IDs
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
def make_env(render_mode=None):
    env = gym.make("dm_control/cartpole-balance-v0", render_mode=render_mode)
    env = FlattenObservation(env)  # dm_control obs are dicts -> flat Box
    env = Monitor(env)            # episode stats for SB3
    return env


if __name__ == "__main__":
    seed = 0

    # Vectorized training env (SB3 uses VecEnv internally) :contentReference[oaicite:1]{index=1}
    train_env = make_vec_env(
        make_env,
        n_envs=8,                 # parallel envs for faster data collection
        seed=seed,
        vec_env_cls=None
    )

    # Separate eval env
    eval_env = make_env()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=10_000,         # every N train steps (across vec env)
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
        tensorboard_log="./logs/tb",
        seed=seed,
    )  # SAC example + key args per SB3 docs :contentReference[oaicite:2]{index=2}

    model.learn(
        total_timesteps=200_000,
        callback=eval_cb,
        log_interval=10,
        progress_bar=False,
    )

    model.save("sac_dmc_cartpole_balance")

    # Quick rollout
    test_env = make_env(render_mode="human")
    obs, info = test_env.reset(seed=10)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            obs, info = test_env.reset()
