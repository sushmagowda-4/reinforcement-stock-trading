import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # Retrieve equity recorded right before step reset
        eq = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # Load dataset for unseen testing
    file_path = "D:/RL_project/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # If evaluating on the OOS test slice (e.g., last 15% or 20%):
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()

    # Must match training parameters
    SL_OPTS = [5, 15, 30, 60, 90]
    TP_OPTS = [5, 15, 30, 60, 90]
    WIN = 30

    def make_test_env():
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
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )

    # 1. Wrap in DummyVecEnv
    vec_test_env = DummyVecEnv([make_test_env])

    # 2. Load normalization stats saved during training
    vec_test_env = VecNormalize.load("vecnormalize.pkl", vec_test_env)
    vec_test_env.training = False       # Disable updating running stats
    vec_test_env.norm_reward = False   # Disable reward normalization during evaluation

    # 3. Load trained model using the normalized environment
    model = PPO.load("model_eurusd_best", env=vec_test_env)

    # 4. Run evaluation episode
    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trade log
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trades recorded.")

    # Save and show equity plot
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity (Unseen Test)-EURUSD", color="tab:blue")
    plt.title("Equity Curve - Evaluation (Unseen Test Data)-EURUSD")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()

    unseen_test_plot_path = os.path.join(plots_dir, "equity_curve_unseen_test.png")
    plt.savefig(unseen_test_plot_path, dpi=300)
    print(f"Saved plot: {unseen_test_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()