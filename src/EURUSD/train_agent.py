import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
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

    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Calculate Sharpe ratio to select smooth, stable equity curves
    if len(returns) > 0 and np.std(returns) > 1e-8:
        sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))
    else:
        sharpe_ratio = -np.inf

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity, sharpe_ratio


def main():
    file_path = (
        "D:/RL_project/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    )
    df, feature_cols = load_and_preprocess_data(file_path)

    # 1. 3-way time split: 70% Train, 15% Validation, 15% Test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Training bars  : {len(train_df)}")
    print(f"Validation bars: {len(val_df)}")
    print(f"Testing bars   : {len(test_df)}")

    # ---- Env settings ----
    SL_OPTS = [5, 15, 30, 60, 90]
    TP_OPTS = [5, 15, 30, 60, 90]
    WIN = 30

    # Train env: Add slight penalties to prevent over-trading noise
    def make_train_env():
        return ForexTradingEnv(
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
            open_penalty_pips=0.1,    # Penalize excessive order opening
            time_penalty_pips=0.005,  # Slight penalty for holding stagnant trades
            unrealized_delta_weight=0.0,
        )

    # Validation env: deterministic evaluation
    def make_val_eval_env():
        return ForexTradingEnv(
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
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )

    # Test env: deterministic evaluation
    def make_test_eval_env():
        return ForexTradingEnv(
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
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,
            time_penalty_pips=0.00,
            unrealized_delta_weight=0.0,
        )

    # 2. Training environment with observation & reward normalization
    train_vec_env = DummyVecEnv([make_train_env])
    train_vec_env = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # 3. Setup Checkpoint Directory
    ckpt_dir = "./checkpoints"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=ckpt_dir, name_prefix="ppo_eurusd"
    )

    # Tuned hyper-parameters for better generalization
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,      # Increased batch size to smooth policy gradients
        ent_coef=0.025,      # Higher exploration coefficient to prevent early convergence
        gamma=0.995,         # Higher discount factor to value long-term gains
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
    )

    # 4. Model Training
    total_timesteps = 600000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # 5. Save VecNormalize statistics
    train_vec_env.save("vecnormalize.pkl")

    # 6. Load Normalization stats into Validation & Test Envs
    val_eval_env = DummyVecEnv([make_val_eval_env])
    val_eval_env = VecNormalize.load("vecnormalize.pkl", val_eval_env)
    val_eval_env.training = False
    val_eval_env.norm_reward = False

    test_eval_env = DummyVecEnv([make_test_eval_env])
    test_eval_env = VecNormalize.load("vecnormalize.pkl", test_eval_env)
    test_eval_env.training = False
    test_eval_env.norm_reward = False

    # 7. Select Checkpoint using Validation Sharpe Ratio (Risk-Adjusted Performance)
    _, val_last_eq, val_last_sharpe = evaluate_model(model, val_eval_env)
    print(f"[Val Eval] Last model - Equity: {val_last_eq:.2f}, Sharpe: {val_last_sharpe:.2f}")

    best_sharpe = -np.inf
    best_path = None

    ckpts = sorted(
        [
            f
            for f in os.listdir(ckpt_dir)
            if f.endswith(".zip") and f.startswith("ppo_eurusd")
        ],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=val_eval_env)
            _, final_eq, sharpe = evaluate_model(m, val_eval_env)
            print(f"[Val Eval] {ck} -> Equity: {final_eq:.2f} | Sharpe: {sharpe:.2f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    if best_path is None or val_last_sharpe >= best_sharpe:
        print("Using last trained model as best.")
        best_model = model
    else:
        print(f"Using best checkpoint by Validation Sharpe Ratio: {best_path} (Sharpe: {best_sharpe:.2f})")
        best_model = PPO.load(best_path, env=train_vec_env)

    best_model.save("model_eurusd_best")
    print("Best model saved: model_eurusd_best")

    # 8. Evaluate on both Validation and Test sets
    equity_curve_val, final_equity_val, sharpe_val = evaluate_model(
        best_model, val_eval_env
    )
    equity_curve_test, final_equity_test, sharpe_test = evaluate_model(
        best_model, test_eval_env
    )

    print(f"\n--- Final Results ---")
    print(f"[Val Eval]  Final equity: {final_equity_val:.2f} | Sharpe: {sharpe_val:.2f}")
    print(f"[Test Eval] Final equity: {final_equity_test:.2f} | Sharpe: {sharpe_test:.2f}")

    # Plot Equity Curves
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_val, label=f"Validation Equity (Sharpe: {sharpe_val:.2f})")
    plt.plot(equity_curve_test, label=f"Test OOS Equity (Sharpe: {sharpe_test:.2f})")
    plt.title("Equity Curves: Validation vs Out-of-Sample Test (Sharpe Selected)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(plots_dir, "equity_curves_combined.png")
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Saved plot: {combined_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()