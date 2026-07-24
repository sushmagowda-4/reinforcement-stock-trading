import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(
    model: PPO, eval_env: VecNormalize, deterministic: bool = True
):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def main():
    file_path = "D:/RL_project/USDCAD/USDCAD_2020-01-01_2023-12-31.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Time splits: 70% Train, 15% Validation, 15% Out-of-Sample Test
    train_split_idx = int(len(df) * 0.70)
    val_split_idx = int(len(df) * 0.85)

    train_df = df.iloc[:train_split_idx].copy()
    val_df = df.iloc[train_split_idx:val_split_idx].copy()
    test_df = df.iloc[val_split_idx:].copy()

    print("Training bars  :", len(train_df))
    print("Validation bars:", len(val_df))
    print("Testing bars   :", len(test_df))

    # ---- Env Settings ----
    SL_OPTS = [5, 15, 30, 60, 90]
    TP_OPTS = [5, 15, 30, 60, 90]
    WIN = 30

    # Wrapped with Monitor to fix evaluation warning and properly log statistics
    def make_train_env():
        env = ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=256,
            episode_max_steps=512,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=1.0,       # Penalize low-conviction entries
            time_penalty_pips=0.05,      # Encourage timely trade exit
            unrealized_delta_weight=0.5, # Step-by-step PnL guidance
        )
        return Monitor(env)

    def make_val_env():
        env = ForexTradingEnv(
            df=val_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )
        return Monitor(env)

    def make_train_eval_env():
        env = ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )
        return Monitor(env)

    def make_test_eval_env():
        env = ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )
        return Monitor(env)

    # 1. Initialize Training Environment with Normalization
    train_vec_env = DummyVecEnv([make_train_env])
    train_vec_env = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # 2. Setup Callbacks
    ckpt_dir = "./checkpoints"
    best_model_dir = "./best_model"

    for path in [ckpt_dir, best_model_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=15000, save_path=ckpt_dir, name_prefix="ppo_usdcad"
    )

    # Validation vector environment with Monitor wrapper
    val_vec_env = DummyVecEnv([make_val_env])
    val_vec_env = VecNormalize(val_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    val_vec_env.training = False

    eval_callback = EvalCallback(
        val_vec_env,
        best_model_save_path=best_model_dir,
        log_path="./logs/",
        eval_freq=15000,
        deterministic=True,
        render=False,
    )

    # 3. Regularized PPO Model Configuration
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=1e-4,      # Lower LR prevents sudden aggressive fitting
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,         # Tighter policy updates
        ent_coef=0.02,           # Higher entropy coefficient encourages exploration
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
    )

    # 4. Train Model
    total_timesteps = 600000
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    # 5. Save Normalization Statistics
    train_vec_env.save("vecnormalize.pkl")

    # 6. Load Evaluation Envs with Normalization
    train_eval_env = DummyVecEnv([make_train_eval_env])
    train_eval_env = VecNormalize.load("vecnormalize.pkl", train_eval_env)
    train_eval_env.training = False
    train_eval_env.norm_reward = False

    test_eval_env = DummyVecEnv([make_test_eval_env])
    test_eval_env = VecNormalize.load("vecnormalize.pkl", test_eval_env)
    test_eval_env.training = False
    test_eval_env.norm_reward = False

    # 7. Select and Load Best Model (Saved via Validation Evaluation)
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Loading best validation model from {best_model_path}")
        best_model = PPO.load(best_model_path, env=train_vec_env)
    else:
        print("Best model checkpoint not found; utilizing final epoch model.")
        best_model = model

    best_model.save("model_usdcad_best")
    print("Saved final top model as model_usdcad_best")

    # 8. Evaluate & Plot Results
    equity_curve_train, final_equity_train = evaluate_model(
        best_model, train_eval_env
    )
    equity_curve_test, final_equity_test = evaluate_model(
        best_model, test_eval_env
    )

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label=f"Train Equity (Final: ${final_equity_train:,.2f})")
    plt.plot(equity_curve_test, label=f"Test Equity (Final: ${final_equity_test:,.2f})")
    plt.title("Equity Curves: Training vs Out-of-Sample Test (USDCAD)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    combined_plot_path = os.path.join(plots_dir, "equity_curves_combined.png")
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Saved plot: {combined_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()