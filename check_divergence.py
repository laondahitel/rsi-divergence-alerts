# ============================================================
# RSI DIVERGENCIA FIGYELŐ RENDSZER (v13 - Javított CME szesszió)
# ============================================================

import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ============================================================
# AUTOMATA FUTURES VÁLTÓ (ROLLOVER) LOGIKA
# ============================================================
def get_active_contract():
    """Kiszámolja az aktuális negyedéves CME futures szerződés nevét."""
    # A képeid alapján 2026. decemberi kontraktust tesztelsz. 
    # Ha élesben akarod futtatni a jelent, kommenteld ki a return-t és vedd ki a lenti sort.
    return "NQZ26.CME" 
    
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    if month in [1, 2]: code, year_str = "H", year
    elif month in [3, 4, 5]: code, year_str = "M", year
    elif month in [6, 7, 8]: code, year_str = "U", year
    elif month in [9, 10, 11]: code, year_str = "Z", year
    else: code, year_str = "H", year + 1
    return f"NQ{code}{str(year_str)[-2:]}.CME"

# ============================================================
# CONFIG
# ============================================================
ACTIVE_CONTRACT = get_active_contract()
SYMBOLS = {ACTIVE_CONTRACT: "Nasdaq"}

TIMEFRAMES = ["H1", "H4", "D1", "W1"]
RSI_PERIOD = 14
PIVOT_LEN = 5
RSI_REBOUND_MIN = 5.0
CLOSE_BUFFER = pd.Timedelta(minutes=15)

# JAVÍTÁS: Megnöveltük az adatkorlátokat, hogy a Yahoo késése ne dobja el a jelet!
MAX_DATA_AGE = {
    "H1": pd.Timedelta(hours=3),   # 90 percről 3 órára növelve
    "H4": pd.Timedelta(hours=8),   # 5 óráról 8 órára növelve
    "D1": pd.Timedelta(days=3),
    "W1": pd.Timedelta(weeks=2),
}

RESAMPLE_RULE = {"H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W"}
BAR_DURATION = {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(weeks=1)}

STATE_FILE = "state.json"
CURRENT_STATE_VERSION = 13

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
            with open(STATE_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

def migrate_state(state):
    if state.get("_version") != CURRENT_STATE_VERSION:
        state.clear() # Teljes reset a v13 miatt
        state["_version"] = CURRENT_STATE_VERSION

    for symbol in SYMBOLS:
        sym = state.setdefault(symbol, {})
        for tf in TIMEFRAMES:
            tf_state = sym.setdefault(tf, {})
            tf_state.setdefault("buy_state", "IDLE")
            tf_state.setdefault("buy_ref_low_price", 0.0)
            tf_state.setdefault("buy_ref_low_rsi", 0.0)
            tf_state.setdefault("buy_ref_low_time", None)
            tf_state.setdefault("sell_state", "IDLE")
            tf_state.setdefault("sell_ref_high_price", 0.0)
            tf_state.setdefault("sell_ref_high_rsi", 0.0)
            tf_state.setdefault("sell_ref_high_time", None)
            tf_state.setdefault("last_buy_alert_candle", None)
            tf_state.setdefault("last_sell_alert_candle", None)

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS:
        print("⚠ Email nincs konfigurálva.")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"]    = GMAIL_USER
        msg["To"]      = ", ".join(EMAIL_TO_LIST)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg, from_addr=GMAIL_USER, to_addrs=EMAIL_TO_LIST)
        print(f"✅ Email elküldve: {subject}")
    except Exception as e:
        print(f"❌ Email küldés hiba: {e}")

# ============================================================
# RSI & ADAT LETÖLTÉS
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def download_tf(symbol, tf):
    try:
        if tf in ["H1", "H4"]:
            api_interval = "5m"
            api_range = "1mo"  # Több adat, stabilabb RSI
        elif tf == "D1":
            api_interval = "1d"
            api_range = "1y"
        elif tf == "W1":
            api_interval = "1wk"
            api_range = "2y"

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": api_interval, "range": api_range, "includePrePost": "true"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        print(f"   ⏳ Letöltés: {symbol} [{tf}] (API: {api_interval}, Range: {api_range})")
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "chart" not in data or not data["chart"]["result"]:
            print(f"   ❌ NINCS ADAT a Yahoo-n: {symbol}")
            return None

        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps: return None

        quote = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": quote["open"], "High": quote["high"],
            "Low": quote["low"], "Close": quote["close"], "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit="s", utc=True))

        df = df.dropna()
        if df.empty: return None

        # --- JAVÍTÁS: CME 23 órás szesszió kezelése (New York-i időzónához igazítás) ---
        if tf in ["H1", "H4"]:
            df_ny = df.tz_convert('America/New_York')
            df_shifted = df_ny.shift(freq='6h') # Eltolás, hogy a 18:00 ET legyen a 00:00
            df_tf_ny = df_shifted.resample(RESAMPLE_RULE[tf]).agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
            df_tf_ny.index = df_tf_ny.index - pd.Timedelta(hours=6)
            df_tf = df_tf_ny.tz_convert('UTC')
        else:
            df_tf = df.resample(RESAMPLE_RULE[tf]).agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
        # -----------------------------------------------------------------------------

        now_utc = pd.Timestamp.now(tz="UTC")
        df_tf = df_tf[df_tf.index + BAR_DURATION[tf] + CLOSE_BUFFER <= now_utc]
        return df_tf
    except Exception as e:
        print(f"   ⚠ Hiba a letöltésnél ({symbol} [{tf}]): {e}")
        return None

# ============================================================
# EMAIL FORMÁZÁS
# ============================================================
def build_email_body(tf, symbol, display_name, side, time_str, ref, last_price, last_rsi, price_diff, rsi_diff):
    side_hu = "VÉTELI" if side == "BUY" else "ELADÁSI"
    ref_label = "pivot low" if side == "BUY" else "pivot high"
    fresh_label = "Low" if side == "BUY" else "High"
    tf_full = {"H1": "1 órás", "H4": "4 órás", "D1": "Napos", "W1": "Hetes"}.get(tf, tf)
    banner = "═" * 60
    
    body = (
        f"{banner}\n   ▶▶  {tf}  ◀◀    {display_name.upper()} — {side_hu} LEHETŐSÉG\n{banner}\n\n"
        f"IDŐSÍK: {tf}  ({tf_full} gyertya)\nIRÁNY:  {side_hu}\n\n{banner}\n\n"
        f"Symbol:                 {symbol}  ({display_name})\n"
        f"Frissen lezárt gyertya: {time_str}\n\n"
        f"Referencia {ref_label}:  {ref['p']:.4f}   RSI: {ref['r']:.2f}   ({ref['t']})\n"
        f"Friss gyertya {fresh_label}:      {last_price:.4f}   RSI: {last_rsi:.2f}   ({time_str})\n\n"
        f"Ár különbség (referencia - friss): {price_diff:.4f}\n"
        f"RSI különbség (friss - referencia): {rsi_diff:.2f}\n\n{banner}\n"
    )
    return body

# ============================================================
# DIVERGENCIA LOGIKA — Állapotgép (State Machine)
# ============================================================
def check_symbol_tf(symbol, display_name, tf, state):
    df = download_tf(symbol, tf)
    if df is None or len(df) < 15: 
        print(f"   ⚠ [{tf}] {symbol} | Kihagyva: Nincs elég adat (df: {len(df) if df is not None else 'None'})")
        return

    last_time       = df.index[-1]
    last_close_time = last_time + BAR_DURATION[tf]
    now_utc         = pd.Timestamp.now(tz="UTC")
    data_age        = now_utc - last_close_time
    
    if data_age > MAX_DATA_AGE[tf]: 
        print(f"   ⚠ [{tf}] {symbol} | Kihagyva: Az adat túl régi! (Kor: {data_age}, Max: {MAX_DATA_AGE[tf]})")
        return

    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    sym_state = state.setdefault(symbol, {})
    tf_state  = sym_state.setdefault(tf, {})

    last_high  = float(df["High"].iloc[-1])
    last_low   = float(df["Low"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    last_rsi   = float(df["RSI"].iloc[-1])
    last_iso   = last_time.isoformat()
    time_str   = last_time.strftime("%Y-%m-%d %H:%M UTC")

    print(f"   [{tf}] {symbol} | Ár: {last_close:.2f} | RSI: {last_rsi:.2f} | Státusz: {tf_state['buy_state']} / {tf_state['sell_state']} | Adat kora: {data_age}")

    # VÉTELI (BULLISH) LOGIKA
    buy_state = tf_state["buy_state"]
    if buy_state == "IDLE":
        if last_rsi < 40:
            tf_state["buy_state"] = "TRACKING_LOW"
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso
            print(f"   [{tf}] {symbol} | 🟢 BUY: Feltétel 1 teljesült (RSI < 40).")
    elif buy_state == "TRACKING_LOW":
        if last_low < tf_state["buy_ref_low_price"]:
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso
        if last_rsi >= tf_state["buy_ref_low_rsi"] + RSI_REBOUND_MIN:
            tf_state["buy_state"] = "WAIT_DIVERGENCE"
            print(f"   [{tf}] {symbol} | 🟢 BUY: Felpattanás (>={RSI_REBOUND_MIN}). Várjuk az új mélypontot...")
    elif buy_state == "WAIT_DIVERGENCE":
        if last_low <= tf_state["buy_ref_low_price"] and (last_rsi - tf_state["buy_ref_low_rsi"]) >= RSI_REBOUND_MIN:
            if tf_state.get("last_buy_alert_candle") != last_iso:
                subject = f"{tf} | Nasdaq VÉTELI lehetőség - RSI Divergencia"
                body = build_email_body(tf, symbol, display_name, "BUY", time_str, 
                                        {'t': tf_state["buy_ref_low_time"], 'p': tf_state["buy_ref_low_price"], 'r': tf_state["buy_ref_low_rsi"]},
                                        last_low, last_rsi, tf_state["buy_ref_low_price"] - last_low, last_rsi - tf_state["buy_ref_low_rsi"])
                send_email(subject, body)
                tf_state["last_buy_alert_candle"] = last_iso
                tf_state["buy_state"] = "IDLE"
                print(f"   [{tf}] {symbol} | 📧 BUY: Divergencia jelezve! Reset.")
        elif last_low < tf_state["buy_ref_low_price"] and last_rsi < tf_state["buy_ref_low_rsi"]:
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso
            print(f"   [{tf}] {symbol} | 🟢 BUY: Új mélypont, de RSI is alacsonyabb. Referencia frissítve.")

    # ELADÁSI (BEARISH) LOGIKA
    sell_state = tf_state["sell_state"]
    if sell_state == "IDLE":
        if last_rsi > 60:
            tf_state["sell_state"] = "TRACKING_HIGH"
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso
            print(f"   [{tf}] {symbol} | 🔴 SELL: Feltétel 1 teljesült (RSI > 60).")
    elif sell_state == "TRACKING_HIGH":
        if last_high > tf_state["sell_ref_high_price"]:
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso
        if last_rsi <= tf_state["sell_ref_high_rsi"] - RSI_REBOUND_MIN:
            tf_state["sell_state"] = "WAIT_DIVERGENCE_SELL"
            print(f"   [{tf}] {symbol} | 🔴 SELL: Lepattanás. Várjuk az új csúcsot...")
    elif sell_state == "WAIT_DIVERGENCE_SELL":
        if last_high >= tf_state["sell_ref_high_price"] and (tf_state["sell_ref_high_rsi"] - last_rsi) >= RSI_REBOUND_MIN:
            if tf_state.get("last_sell_alert_candle") != last_iso:
                subject = f"{tf} | Nasdaq ELADÁSI lehetőség - RSI Divergencia"
                body = build_email_body(tf, symbol, display_name, "SELL", time_str,
                                        {'t': tf_state["sell_ref_high_time"], 'p': tf_state["sell_ref_high_price"], 'r': tf_state["sell_ref_high_rsi"]},
                                        last_high, last_rsi, last_high - tf_state["sell_ref_high_price"], tf_state["sell_ref_high_rsi"] - last_rsi)
                send_email(subject, body)
                tf_state["last_sell_alert_candle"] = last_iso
                tf_state["sell_state"] = "IDLE"
                print(f"   [{tf}] {symbol} | 📧 SELL: Divergencia jelezve! Reset.")
        elif last_high > tf_state["sell_ref_high_price"] and last_rsi > tf_state["sell_ref_high_rsi"]:
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso
            print(f"   [{tf}] {symbol} | 🔴 SELL: Új csúcs, de RSI is magasabb. Referencia frissítve.")

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Futás: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📊 Aktív szerződés: {ACTIVE_CONTRACT}")
    state = load_state()
    migrate_state(state)

    for symbol, name in SYMBOLS.items():
        for tf in TIMEFRAMES:
            try:
                check_symbol_tf(symbol, name, tf, state)
            except Exception as e:
                print(f"❌ Hiba {symbol} [{tf}]-nál: {e}")

    save_state(state)
    print("✅ Kész.")

if __name__ == "__main__":
    main()
