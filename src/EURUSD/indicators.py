import pandas as pd
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns:
    [Time (EET), Open, High, Low, Close, Volume]
    """

    df = pd.read_csv(csv_path)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Datetime index
    time_col = "Time (EET)" if "Time (EET)" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
    df = df.set_index(time_col)
    df.sort_index(inplace=True)

    # Convert columns to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ============================================================
    # Technical Indicators
    # ============================================================

    # RSI
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    # ATR
    df["atr_14"] = ta.atr(
        df["High"],
        df["Low"],
        df["Close"],
        length=14
    )

    # Moving Averages
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    # 200 EMA
    df["ema_200"] = ta.ema(df["Close"], length=200)

    # ============================================================
    # MACD
    # ============================================================

    macd = ta.macd(df["Close"])

    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # ============================================================
    # VWAP
    # ============================================================

    df["vwap"] = ta.vwap(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    )

    # ============================================================
    # Relative Features
    # ============================================================

    # MA slopes
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # EMA slope
    df["ema200_slope"] = df["ema_200"].diff()

    # Distance from MAs
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # Distance from EMA200
    df["close_ema200_diff"] = df["Close"] - df["ema_200"]

    # Distance from VWAP
    df["close_vwap_diff"] = df["Close"] - df["vwap"]

    # MA spread
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # MACD relative features
    df["macd_signal_diff"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff()

    # ============================================================
    # Clean Data
    # ============================================================

    df.dropna(inplace=True)

    # ============================================================
    # Features for RL Agent
    # ============================================================

    feature_cols = [
        "rsi_14",
        "atr_14",

        "ma_20_slope",
        "ma_50_slope",
        "ema200_slope",

        "close_ma20_diff",
        "close_ma50_diff",
        "close_ema200_diff",
        "close_vwap_diff",

        "ma_spread",
        "ma_spread_slope",

        "macd_signal_diff",
        "macd_hist",
        "macd_hist_slope",
    ]

    return df, feature_cols