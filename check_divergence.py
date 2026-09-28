import os
import json
import smtplib
import time
import requests
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
SYMBOLS = {
    "NQ=F": "Nasdaq",
}

RSI_PERIOD     = 14
PIVOT_LEN      = 5
MIN_RSI_DIFF   = 2.0
MIN_PRICE_DIFF = 0.0

STATE_FILE = "state.json"

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
EMAIL_TO_RAW = os.environ.get("EMAIL_TO", GMAIL_USER or "")
EMAIL_TO_LIST = [a.strip() for a in EMAIL_TO_RAW.replace(";", ",").split(",") if a.strip()]

# ============================================================
# UTIL
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS:
        print("⚠ Email nincs konfigurálva (GMAIL_USER / GMAIL_PASS hiányzik).")
        return
    if not EMAIL_TO_LIST:
        print("⚠ Nincs érvényes címzett az EMAIL_TO secret-ben.")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"]    = GMAIL_USER
        msg["To"]      = ", ".join(EMAIL_TO_LIST)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg, from_addr=GMAIL_USER, to_addrs=EMAIL_TO_LIST)
        print(f"✅ Email elküldve: {subject} → {EMAIL_TO_LIST}")
    except Exception as e:
        print(f"❌ Email küldés hiba: {e}")

# ============================================================
# RSI (Wilder's smoothing)
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================
# ADAT LETÖLTÉS — KÖZVETLEN YAHOO API
# ============================================================
def download_4h(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "interval": "15m",
            "range": "1mo",
            "includePrePost": "true",
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            print(f"⚠ Nincs timestamp a válaszban: {symbol}")
            return None

        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open":   quote["open"],
            "High":   quote["high"],
            "Low":    quote["low"],
            "Close":  quote["close"],
            "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df = df.dropna()
        if df.empty:
            print(f"⚠ Üres adat: {symbol}")
            return None

        # DEBUG
        print(f"🔍 DEBUG | Nyers sorok: {len(df)}")
        print(f"🔍 DEBUG | Utolsó sor:  {df.index[-1]}")
        print(f"🔍 DEBUG | Utolsó close: {df['Close'].iloc[-1]:.4f}")

        # 4h resample
        df4 = df.resample("4h").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

        now_utc = pd.Timestamp.now(tz="UTC")
        df4 = df4[df4.index + pd.Timedelta(hours=4) <= now_utc]

        return df4
    except Exception as e:
        print(f"⚠ Hiba a letöltésnél ({symbol}): {e}")
        return None

# ============================================================
# PIVOT DETEKTÁLÁS (ZigZag-szerű)
# ============================================================
def find_pivots(df, left, right):
    highs = df["High"].values
    lows  = df["Low"].values
    n = len(df)

    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        is_ph = True
        is_pl = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                is_ph = False
            if lows[j] <= lows[i]:
                is_pl = False
            if not is_ph and not is_pl:
                break
        ph[i] = is_ph
        pl[i] = is_pl

    return ph, pl

# ============================================================
# DIVERGENCIA LOGIKA
# ============================================================
def check_symbol(symbol, display_name, state):
    print(f"\n=== {symbol} ({display_name}) ===")

    df = download_4h(symbol)
    if df is None or len(df) < 30:
        print(f"⚠ Kevés adat: {symbol}")
        return

    print(f"Gyertyák száma: {len(df)}  |  utolsó lezárt: {df.index[-1]}")

    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)
    ph, pl = find_pivots(df, PIVOT_LEN, PIVOT_LEN)

    pivot_highs = [(df.index[i], df["High"].iloc[i], df["RSI"].iloc[i])
                   for i in range(len(df)) if ph[i]]
    pivot_lows  = [(df.index[i], df["Low"].iloc[i],  df["RSI"].iloc[i])
                   for i in range(len(df)) if pl[i]]

    print(f"Pivot high-ok: {len(pivot_highs)}  |  pivot low-ok: {len(pivot_lows)}")

    if pivot_lows:
        print(f"  Utolsó pivot low:  {pivot_lows[-1][0]}  ár={pivot_lows[-1][1]:.4f}  RSI={pivot_lows[-1][2]:.2f}")
    if pivot_highs:
        print(f"  Utolsó pivot high: {pivot_highs[-1][0]}  ár={pivot_highs[-1][1]:.4f}  RSI={pivot_highs[-1][2]:.2f}")

    if len(pivot_lows) < 2 and len(pivot_highs) < 2:
        print(f"Nincs elég pivot.")
        return

    key_buy  = f"{symbol}_last_buy_pivot"
    key_sell = f"{symbol}_last_sell_pivot"

    # ---- BULLISH divergencia ----
    if len(pivot_lows) >= 2:
        curr = pivot_lows[-1]
        prev = pivot_lows[-2]

        pivot_id = curr[0].isoformat()
        already_notified = (state.get(key_buy) == pivot_id)

        if not already_notified:
            price_ll = curr[1] < prev[1]
            rsi_hl   = curr[2] > prev[2]
            price_ok = (prev[1] - curr[1]) >= MIN_PRICE_DIFF
            rsi_ok   = (curr[2] - prev[2]) >= MIN_RSI_DIFF

            print(f"BUY check: price_ll={price_ll}  rsi_hl={rsi_hl}  price_diff={prev[1]-curr[1]:.4f}  rsi_diff={curr[2]-prev[2]:.2f}")

            if price_ll and rsi_hl and price_ok and rsi_ok:
                curr_time_str = curr[0].strftime("%Y-%m-%d %H:%M UTC")
                subject = "Nasdaq LQQ , 3QQQ VÉTELI lehetőség keletkezett, H4 RSI Divergencia"
                body = (
                    f"Nasdaq LQQ , 3QQQ VÉTELI lehetőség keletkezett, H4 RSI Divergencia\n\n"
                    f"Symbol: {symbol}\n"
                    f"Pivot idő: {curr_time_str}\n\n"
                    f"Előző pivot low: {prev[1]:.4f}  RSI: {prev[2]:.2f}  ({prev[0]})\n"
                    f"Mostani pivot low: {curr[1]:.4f}  RSI: {curr[2]:.2f}  ({curr[0]})\n"
                    f"Ár különbség: {prev[1] - curr[1]:.4f}\n"
                    f"RSI különbség: {curr[2] - prev[2]:.2f}\n"
                )
                send_email(subject, body)
                state[key_buy] = pivot_id
        else:
            print(f"BUY pivot már jelezve korábban: {curr[0]}")

    # ---- BEARISH divergencia ----
    if len(pivot_highs) >= 2:
        curr = pivot_highs[-1]
        prev = pivot_highs[-2]

        pivot_id = curr[0].isoformat()
        already_notified = (state.get(key_sell) == pivot_id)

        if not already_notified:
            price_hh = curr[1] > prev[1]
            rsi_lh   = curr[2] < prev[2]
            price_ok = (curr[1] - prev[1]) >= MIN_PRICE_DIFF
            rsi_ok   = (prev[2] - curr[2]) >= MIN_RSI_DIFF

            print(f"SELL check: price_hh={price_hh}  rsi_lh={rsi_lh}  price_diff={curr[1]-prev[1]:.4f}  rsi_diff={prev[2]-curr[2]:.2f}")

            if price_hh and rsi_lh and price_ok and rsi_ok:
                curr_time_str = curr[0].strftime("%Y-%m-%d %H:%M UTC")
                subject = "Nasdaq LQQ , 3QQQ ELADÁSI lehetőség keletkezett, H4 RSI Divergencia"
                body = (
                    f"Nasdaq LQQ , 3QQQ ELADÁSI lehetőség keletkezett, H4 RSI Divergencia\n\n"
                    f"Symbol: {symbol}\n"
                    f"Pivot idő: {curr_time_str}\n\n"
                    f"Előző pivot high: {prev[1]:.4f}  RSI: {prev[2]:.2f}  ({prev[0]})\n"
                    f"Mostani pivot high: {curr[1]:.4f}  RSI: {curr[2]:.2f}  ({curr[0]})\n"
                    f"Ár különbség: {curr[1] - prev[1]:.4f}\n"
                    f"RSI különbség: {prev[2] - curr[2]:.2f}\n"
                )
                send_email(subject, body)
                state[key_sell] = pivot_id
        else:
            print(f"SELL pivot már jelezve korábban: {curr[0]}")

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Futás: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Címzettek: {EMAIL_TO_LIST}")
    state = load_state()

    for symbol, name in SYMBOLS.items():
        try:
            check_symbol(symbol, name, state)
        except Exception as e:
            print(f"❌ Hiba {symbol}-nál: {e}")

    save_state(state)
    print("\n✅ Kész.")

if __name__ == "__main__":
    main()
