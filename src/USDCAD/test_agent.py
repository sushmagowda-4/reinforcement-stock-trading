import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to ensure reliable file export
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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
        eq = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # Load unseen out-of-sample evaluation dataset
    file_path = "D:/RL_project/USDCAD/USDCAD_2024-01-01_2026-07-22.csv"
    df, feature_cols = load_and_preprocess_data(file_path)

    # Use the entire dataset for full unseen evaluation
    eval_df = df.copy()

    # Environment parameters matching training configuration
    SL_OPTS = [5, 15, 30, 60, 90]
    TP_OPTS = [5, 15, 30, 60, 90]
    WIN = 30

    def make_test_env():
        env = ForexTradingEnv(
            df=eval_df,
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

    # 1. Setup Base Vectorized Environment
    raw_test_env = DummyVecEnv([make_test_env])

    # 2. Apply Saved Observation Normalization
    vecnorm_path = "vecnormalize.pkl"
    if os.path.exists(vecnorm_path):
        vec_test_env = VecNormalize.load(vecnorm_path, raw_test_env)
        vec_test_env.training = False
        vec_test_env.norm_reward = False
        print(f"Successfully loaded normalization stats from {vecnorm_path}")
    else:
        print("Warning: vecnormalize.pkl not found! Evaluating without observation normalization.")
        vec_test_env = raw_test_env

    # 3. Load Trained Model
    model_path = "model_usdcad_best"
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        model = PPO.load(model_path, env=vec_test_env)
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found! Run the training script first.")

    # 4. Run Out-of-Sample Evaluation
    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # 5. Process & Save Closed Trades History
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)

        # Print trade summary statistics
        win_trades = trades_df[trades_df["pnl_usd"] > 0] if "pnl_usd" in trades_df.columns else []
        win_rate = (len(win_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0.0

        print(f"\n--- Evaluation Results ---")
        print(f"Total Trades Recorded : {len(trades_df)}")
        print(f"Win Rate              : {win_rate:.2f}%")
        print(f"Closed Trade History  : Saved to {out_csv}")
    else:
        print("\nNo closed trades recorded during evaluation.")

  # 6. Plot & Export Equity Curve Graph
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    initial_equity = equity_curve[0] if equity_curve else 10000.0
    final_equity = equity_curve[-1] if equity_curve else initial_equity
    total_return = ((final_equity - initial_equity) / initial_equity) * 100

    print(f"Initial Equity        : ${initial_equity:,.2f}")
    print(f"Final Equity          : ${final_equity:,.2f}")
    print(f"Total Return          : {total_return:+.2f}%\n")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        equity_curve,
        label=f"Unseen Test Equity (Final: ${final_equity:,.2f} | Return: {total_return:+.2f}%)",
        color="tab:blue",
    )
    ax.set_title("Out-of-Sample Equity Curve Evaluation - USDCAD")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Equity ($)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Save PNG format only for export
    unseen_test_plot_path = os.path.join(
        plots_dir, "equity_curve_unseen_test.png"
    )
    fig.savefig(unseen_test_plot_path, dpi=300, bbox_inches="tight")

    print(f"Exported plot PNG: {os.path.abspath(unseen_test_plot_path)}")

    plt.close(fig)

if __name__ == "__main__":
    main()