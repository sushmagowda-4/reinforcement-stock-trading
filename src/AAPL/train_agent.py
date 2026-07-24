import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# ---- Global Configuration ----
TICKER = "AAPL"
DATA_PATH = r"D:/RL_project/APPL/AAPL_60min_2017_2022.csv"
TOTAL_TIMESTEPS = 450_000

SL_OPTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
TP_OPTS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
WIN = 30

# Common environment parameters (ensures consistency across train and eval envs)
ENV_KWARGS = dict(
    window_size=WIN,
    sl_options=SL_OPTS,
    tp_options=TP_OPTS,
    spread_pips=0.01,         # Reduced friction for stock trading
    commission_pips=0.001,
    max_slippage_pips=0.01,
    hold_reward_weight=0.001, # Small bonus/penalty for holding positions
    open_penalty_pips=0.05,   # Penalize overtrading
    time_penalty_pips=0.0,
    unrealized_delta_weight=0.1,
)


def evaluate_model(model: PPO, eval_env: VecNormalize, deterministic: bool = True):
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
        # Retrieve equity safely before reset on done
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def make_env_factory(df, feature_cols, random_start=False, min_steps=500, max_steps=None):
    """Creates a zero-argument environment constructor compatible with DummyVecEnv."""
    def _init():
        return ForexTradingEnv(
            df=df,
            feature_columns=feature_cols,
            random_start=random_start,
            min_episode_steps=min_steps,
            episode_max_steps=max_steps,
            **ENV_KWARGS,
        )
    return _init


def main():
    # 1. Load Data
    df, feature_cols = load_and_preprocess_data(DATA_PATH)

    # Time-based split: 80% Train, 20% In-Sample Test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"[{TICKER}] Training bars : {len(train_df)}")
    print(f"[{TICKER}] Testing bars  : {len(test_df)}")

    # 2. Build Training Vectorized Environment
    train_env_fn = make_env_factory(
        train_df, 
        feature_cols, 
        random_start=True, 
        min_steps=500, 
        max_steps=2000
    )
    train_vec_env = DummyVecEnv([train_env_fn])
    train_vec_env = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    # 3. Setup Directories and Callbacks
    ckpt_dir = "./checkpoints"
    plots_dir = "./plots"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    prefix = f"ppo_{TICKER.lower()}"
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=ckpt_dir, name_prefix=prefix
    )

    # 4. Initialize & Train PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/",
    )

    print(f"--- Starting Training ({TOTAL_TIMESTEPS} steps) ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    # 5. Save Normalization Statistics
    vecnorm_file = "vecnormalize.pkl"
    train_vec_env.save(vecnorm_file)
    print(f"Saved observation/reward normalization stats to {vecnorm_file}")

    # 6. Build Evaluation Environments
    train_eval_fn = make_env_factory(train_df, feature_cols, random_start=False)
    test_eval_fn = make_env_factory(test_df, feature_cols, random_start=False)

    train_eval_env = VecNormalize.load(vecnorm_file, DummyVecEnv([train_eval_fn]))
    train_eval_env.training = False
    train_eval_env.norm_reward = False

    test_eval_env = VecNormalize.load(vecnorm_file, DummyVecEnv([test_eval_fn]))
    test_eval_env.training = False
    test_eval_env.norm_reward = False

    # 7. Model Selection based on Out-Of-Sample (OOS) Equity
    _, final_equity_test_last = evaluate_model(model, test_eval_env)
    print(f"[OOS Eval] Final Step Model Equity: {final_equity_test_last:.2f}")

    best_equity = -np.inf
    best_path = None

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".zip") and f.startswith(prefix)],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq = evaluate_model(m, test_eval_env)
            print(f"[OOS Eval] Checkpoint '{ck}' -> Equity: ${final_eq:.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    if best_path is None or final_equity_test_last >= best_equity:
        print("--> Using last trained model as best.")
        best_model = model
    else:
        print(f"--> Using best checkpoint: {best_path} (Equity: ${best_equity:.2f})")
        best_model = PPO.load(best_path, env=train_vec_env)

    saved_model_name = f"model_{TICKER.lower()}_best"
    best_model.save(saved_model_name)
    print(f"Saved best model to '{saved_model_name}'")

    # 8. Evaluate Best Model (In-Sample vs Out-of-Sample)
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"[IS  Eval] Final Equity (Train): ${final_equity_train:.2f}")
    print(f"[OOS Eval] Final Equity (Test) : ${final_equity_test:.2f}")

    # 9. Plot Results
    # Plot 1: Combined In-Sample vs Out-of-Sample Equity Curves
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label=f"Train (In-Sample) - {TICKER}", color="tab:blue")
    plt.plot(equity_curve_test, label=f"Test (Out-of-Sample) - {TICKER}", color="tab:orange")
    plt.title(f"Equity Curves: In-Sample vs Out-of-Sample ({TICKER})")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(plots_dir, "equity_curves_combined.png")
    plt.savefig(combined_plot_path, dpi=300)
    print(f"Saved plot: {combined_plot_path}")
    plt.show()

    # Plot 2: Out-of-Sample Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_test, color="tab:orange", label=f"Test (OOS) Equity - {TICKER}")
    plt.title(f"Out-of-Sample Equity Curve ({TICKER})")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    oos_plot_path = os.path.join(plots_dir, "equity_curve_oos.png")
    plt.savefig(oos_plot_path, dpi=300)
    print(f"Saved plot: {oos_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()