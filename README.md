# Reinforcement Learning for Automated Stock Trading  

This project is part of my MCA dissertation at Amrita University. It demonstrates how **Deep Reinforcement Learning (DRL)** can be applied to financial trading using PPO agents trained on historical datasets. The system is designed for **research and educational purposes only** and does not involve live trading.  

---

## 📌 Project Overview  
Traditional trading strategies rely on fixed rules or predictive models, which often fail to adapt to sudden market changes. This project explores **Reinforcement Learning (RL)** for trading, where an agent learns buy, sell, and hold decisions by interacting with a simulated environment.  

**Workflow:**  
```
Historical Market Data
        ↓
Data Preprocessing + Technical Indicators
        ↓
Custom Trading Environment
        ↓
PPO Reinforcement Learning Agent
        ↓
Trading Decisions
        ↓
Backtesting & Evaluation
        ↓
Equity Curve Analysis
```

---

## 🎯 Objectives  
- Build a reinforcement learning trading system.  
- Create a custom Gymnasium environment for financial data.  
- Generate technical features (RSI, ATR, MA, EMA, MACD, VWAP, slopes, spreads).  
- Train a PPO agent using Stable Baselines3.  
- Include realistic trading frictions (spread, commission, slippage).  
- Use stop‑loss and take‑profit mechanisms.  
- Evaluate models on in‑sample, out‑of‑sample, and unseen data.  
- Visualize portfolio/equity performance.  

---

## 🛠️ Technologies Used  
| Technology        | Purpose                           |
| ----------------- | --------------------------------- |
| Python            | Programming language              |
| Pandas / NumPy    | Data processing & computation     |
| Pandas‑TA         | Technical indicators              |
| Matplotlib        | Visualization                     |
| Gym / Gymnasium   | RL environment interface          |
| Stable‑Baselines3 | PPO algorithm implementation      |
| PyTorch           | Deep learning backend             |
| yfinance          | Market data access                |
| Backtesting       | Strategy evaluation               |

---

## 📂 Project Structure  
```
reinforcement-stock-trading/
│
├── data/                  # Market datasets
│
├── results/               # Trained models & equity curves
│   ├── RL_AAPL/
│   ├── RL_EURUSD/
│   ├── RL_GOOG/
│   └── RL_USDCAD/
│
├── src/                   # Source code for each instrument
│   ├── AAPL/
│   │   ├── indicators.py
│   │   ├── trading_env.py
│   │   ├── train_agent.py
│   │   └── test_agent.py
│   ├── EURUSD/
│   ├── GOOG/
│   └── USDCAD/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Training & Evaluation  
- **Algorithm:** PPO (policy gradient)  
- **Observation Window:** 30 steps  
- **Training Steps:** ~450,000  
- **Evaluation:**  
  - In‑sample (training data)  
  - Out‑of‑sample (test split)  
  - Unseen dataset (later period)  

Equity curves are generated under `results/` for each instrument.  

---

## 🚀 Getting Started  

### Installation  
```bash
git clone https://github.com/sushmagowda-4/reinforcement-stock-trading.git
cd reinforcement-stock-trading
pip install -r requirements.txt
```

### Training  
```bash
cd src/AAPL
python train_agent.py
```

### Testing  
```bash
python test_agent.py
```

Outputs include equity curves and trade history CSV files.  

---

## 📊 Results  
- Agents learned adaptive trading strategies.  
- Equity curves showed portfolio growth with fluctuations.  
- Performance varied across instruments (AAPL, GOOG, EUR/USD, USD/CAD).  

---

## 🔮 Future Work  
- Compare PPO with DQN, A2C, SAC, TD3.  
- Add portfolio‑level trading.  
- Integrate risk metrics (Sharpe ratio, max drawdown).  
- Experiment with recurrent policies (LSTM agents).  
- Add sentiment/fundamental features.  
- Build a dashboard for monitoring trades.  

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

## 👩‍💻 Author  
**Sushma Gowda**  
GitHub: sushmagowda-4 [(github.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fgithub.com%2Fsushmagowda-4")  

---

## ⚠️ Disclaimer  
This project is for **academic and research purposes only**. Backtesting results do not guarantee future profitability. Real‑world trading requires additional validation, risk management, and financial supervision.  
