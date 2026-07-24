import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# ---- Configuration Constants ----
TICKER = "GOOG"
DATA_PATH = r"D:/RL_project/GOOG/GOOG_60min_2023_2025.csv"

SL_OPTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
TP_OPTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
WIN = 30

ENV_KWARGS = dict(
    window_size=WIN,
    sl_options=SL_OPTS,
    tp_options=TP_OPTS,
    spread_pips=0.01,
    commission_pips=0.001,
    max_slippage_pips=0.01,
    hold_reward_weight=0.001,
    open_penalty_pips=0.05,
    time_penalty_pips=0.0,
    unrealized_delta_weight=0.1,
)


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

        # Safely capture equity state before auto-reset on done
        eq = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        # Record closed trades
        trade_info = info.get("last_trade_info", vec_env.get_attr("last_trade_info")[0])
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # 1. Load Data
    df, feature_cols = load_and_preprocess_data(DATA_PATH)

    # Use entire dataset (or test slice if preferred)
    test_df = df.copy()

    # 2. Build Test Environment
    def make_test_env():
        return ForexTradingEnv(
            df=test_df,
            feature_columns=feature_cols,
            random_start=False,
            episode_max_steps=None,
            **ENV_KWARGS,
        )

    vec_test_env = DummyVecEnv([make_test_env])

    # 3. Load Saved Normalization Statistics
    vecnorm_path = "vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        vec_test_env = VecNormalize.load(vecnorm_path, vec_test_env)
        vec_test_env.training = False
        vec_test_env.norm_reward = False
        print("Loaded VecNormalize statistics successfully.")
    else:
        print("Warning: vecnormalize.pkl not found! Inference might be inaccurate without feature scaling.")

    # 4. Load Trained Model
    model_path = f"model_{TICKER.lower()}_best"
    model = PPO.load(model_path, env=vec_test_env)
    print(f"Loaded trained model: {model_path}")

    # 5. Run Evaluation Episode
    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # 6. Export Trade History CSV
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = f"trade_history_{TICKER.lower()}_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to '{out_csv}'. Total Trades: {len(trades_df)}")
    else:
        print("No closed trades recorded.")

    print(f"Final Equity: ${equity_curve[-1]:.2f}")

    # 7. Plotting
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f"Equity (Test) - {TICKER}", color="tab:orange")
    plt.title(f"Out-of-Sample Equity Curve Evaluation ({TICKER} 2023–2025)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    unseen_test_plot_path = os.path.join(plots_dir, f"equity_curve_unseen_test_{TICKER.lower()}.png")
    plt.savefig(unseen_test_plot_path, dpi=300)
    print(f"Saved plot: {unseen_test_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()