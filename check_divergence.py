# ============================================================
# RSI DIVERGENCIA FIGYELŐ RENDSZER (v11)
# ============================================================
# A logika a felhasználó által pontosított szigorú szűrőn alapul:
# 
# FELTÉTEL 1: A bot csak akkor kezd el Candle Low/High pontokat gyűjteni,
#             ha az RSI értéke a gyertya zárásakor 40 alá (Buy) vagy 
#             60 fölé (Sell) esik.
# 
# FELTÉTEL 2: A gyűjtés során figyeli a visszapattanást (RSI_REBOUND_MIN),
#             majd az újabb mélypont/csúcs elérésekor vizsgálja a divergenciát.
#             A divergenciának is legalább RSI_REBOUND_MIN értékűnek kell lennie!
# 
# ISMÉTLŐDÉS: Az első email kiküldése után a rendszer teljesen kinullázódik.
#             A második emailhez a piacnak egy teljesen új ciklust kell 
#             felépítenie (újra 40 alá/60 fölé kell mennie, majd visszapattannia).
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
# CONFIG
# ============================================================
SYMBOLS = {
    "NQ=F": "Nasdaq",
}

# Figyelt idősíkok (H1, H4, Napos, Hetes)
TIMEFRAMES = ["H1", "H4", "D1", "W1"]

RSI_PERIOD     = 14
PIVOT_LEN      = 5

# EZ A LÉNYEG: Mind a felpattanás, mind a divergencia mértékének ez a minimuma!
# Ha szigorúbb akarsz lenni, írd át 7.0-ra vagy 10.0-ra.
RSI_REBOUND_MIN = 5.0

# A gyertya hivatalos zárása után ennyi ráhagyás (Yahoo késés miatt)
CLOSE_BUFFER = pd.Timedelta(minutes=15)

# Frissességi limitek idősíkonként
MAX_DATA_AGE = {
    "H1": pd.Timedelta(minutes=90),
    "H4": pd.Timedelta(hours=5),
    "D1": pd.Timedelta(days=2),
    "W1": pd.Timedelta(weeks=1),
}

# pandas resample szabály idősíkonként
RESAMPLE_RULE = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}

# Egy gyertya hossza idősíkonként (a zárás számításához)
BAR_DURATION = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
    "W1": pd.Timedelta(weeks=1),
}

STATE_FILE = "state.json"
CURRENT_STATE_VERSION = 11   # v11: RSI_REBOUND_MIN alkalmazása a divergencia ellenőrzésére is

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


def migrate_state(state):
    for symbol in SYMBOLS:
        for old_key in (f"{symbol}_last_buy_pivot", f"{symbol}_last_sell_pivot"):
            state.pop(old_key, None)

    if state.get("_version") != CURRENT_STATE_VERSION:
        for symbol in SYMBOLS:
            state.pop(symbol, None)
        state["_version"] = CURRENT_STATE_VERSION

    for symbol in SYMBOLS:
        sym = state.setdefault(symbol, {})
        for tf in TIMEFRAMES:
            tf_state = sym.setdefault(tf, {})
            
            # Állapotgép (State Machine) változók
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
# ADAT LETÖLTÉS — YAHOO API
# ============================================================
def download_tf(symbol, tf):
    try:
        api_interval = "15m"
        api_range = "1mo"
        
        if tf in ["H1", "H4"]:
            api_interval = "15m"
            api_range = "1mo"
        elif tf == "D1":
            api_interval = "1d"
            api_range = "1y"
        elif tf == "W1":
            api_interval = "1wk"
            api_range = "2y"

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "interval": api_interval,
            "range": api_range,
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
            print(f"⚠ Nincs timestamp a válaszban: {symbol} [{tf}]")
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
            print(f"⚠ Üres adat: {symbol} [{tf}]")
            return None

        df_tf = df.resample(RESAMPLE_RULE[tf]).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

        now_utc = pd.Timestamp.now(tz="UTC")
        df_tf = df_tf[df_tf.index + BAR_DURATION[tf] + CLOSE_BUFFER <= now_utc]

        return df_tf
    except Exception as e:
        print(f"⚠ Hiba a letöltésnél ({symbol} [{tf}]): {e}")
        return None


# ============================================================
# EMAIL FORMÁZÁS
# ============================================================
def build_email_body(tf, symbol, display_name, side, time_str, ref, last_price, last_rsi,
                     price_diff, rsi_diff):
    side_hu = "VÉTELI" if side == "BUY" else "ELADÁSI"
    ref_label = "pivot low" if side == "BUY" else "pivot high"
    fresh_label = "Low" if side == "BUY" else "High"
    
    if tf == "H1": tf_full = "1 órás"
    elif tf == "H4": tf_full = "4 órás"
    elif tf == "D1": tf_full = "Napos"
    elif tf == "W1": tf_full = "Hetes"
    else: tf_full = tf

    banner = "═" * 60
    body = (
        f"{banner}\n"
        f"   ▶▶  {tf}  ◀◀    {display_name.upper()} — {side_hu} LEHETŐSÉG\n"
        f"{banner}\n"
        f"\n"
        f"IDŐSÍK: {tf}  ({tf_full} gyertya)\n"
        f"IRÁNY:  {side_hu}\n"
        f"\n"
        f"{banner}\n"
        f"\n"
        f"Symbol:                 {symbol}  ({display_name})\n"
        f"Frissen lezárt gyertya: {time_str}\n"
        f"\n"
        f"Referencia {ref_label}:  {ref['p']:.4f}   RSI: {ref['r']:.2f}   ({ref['t']})\n"
        f"Friss gyertya {fresh_label}:      {last_price:.4f}   RSI: {last_rsi:.2f}   ({time_str})\n"
        f"\n"
        f"Ár különbség (referencia - friss): {price_diff:.4f}\n"
        f"RSI különbség (friss - referencia): {rsi_diff:.2f}\n"
        f"\n"
        f"{banner}\n"
    )
    return body


# ============================================================
# DIVERGENCIA LOGIKA — Állapotgép (State Machine)
# ============================================================
def check_symbol_tf(symbol, display_name, tf, state):
    df = download_tf(symbol, tf)
    if df is None or len(df) < 15:
        return

    last_time       = df.index[-1]
    last_close_time = last_time + BAR_DURATION[tf]
    now_utc         = pd.Timestamp.now(tz="UTC")
    data_age        = now_utc - last_close_time
    age_limit       = MAX_DATA_AGE[tf]
    if data_age > age_limit:
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

    print(f"   [{tf}] {symbol} | Ár: {last_close:.2f} | RSI: {last_rsi:.2f} | Állapot: {tf_state['buy_state']} / {tf_state['sell_state']}")

    # =========================================================
    # VÉTELI (BULLISH) LOGIKA - Feltétel 1: Csak 40 alatt gyűjtünk
    # =========================================================
    buy_state = tf_state["buy_state"]
    
    if buy_state == "IDLE":
        # Feltétel 1: Csak akkor kezdjük el a gyűjtést, ha az RSI 40 alá esett
        if last_rsi < 40:
            tf_state["buy_state"] = "TRACKING_LOW"
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso
            print(f"   [{tf}] {symbol} | BUY: Feltétel 1 teljesült (RSI < 40). Figyeljük a Low pontokat...")

    elif buy_state == "TRACKING_LOW":
        # Ha mélyebb Low jön, frissítjük a referencia pontot
        if last_low < tf_state["buy_ref_low_price"]:
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso
        
        # Ha az RSI visszapattan a mélypontról (Feltétel 2 első fele)
        if last_rsi >= tf_state["buy_ref_low_rsi"] + RSI_REBOUND_MIN:
            tf_state["buy_state"] = "WAIT_DIVERGENCE"
            print(f"   [{tf}] {symbol} | BUY: Felpattanás (+{RSI_REBOUND_MIN} RSI). Várjuk az új mélypontot divergenciával...")

    elif buy_state == "WAIT_DIVERGENCE":
        # Feltétel 2 második fele: Újra esik, de RSI legalább RSI_REBOUND_MIN-nel magasabb
        if last_low <= tf_state["buy_ref_low_price"] and (last_rsi - tf_state["buy_ref_low_rsi"]) >= RSI_REBOUND_MIN:
            if tf_state.get("last_buy_alert_candle") != last_iso:
                subject = f"{tf} | Nasdaq VÉTELI lehetőség - RSI Divergencia"
                body = build_email_body(
                    tf=tf, symbol=symbol, display_name=display_name, side="BUY",
                    time_str=time_str, 
                    ref={'t': tf_state["buy_ref_low_time"], 'p': tf_state["buy_ref_low_price"], 'r': tf_state["buy_ref_low_rsi"]},
                    last_price=last_low, last_rsi=last_rsi,
                    price_diff=tf_state["buy_ref_low_price"] - last_low,
                    rsi_diff=last_rsi - tf_state["buy_ref_low_rsi"]
                )
                send_email(subject, body)
                tf_state["last_buy_alert_candle"] = last_iso
                
                # TELJES RESET: A következő emailhez újra 40 alá kell esnie és felpattannia!
                tf_state["buy_state"] = "IDLE"
                print(f"   [{tf}] {symbol} | BUY: Divergencia jelezve! Rendszer alaphelyzetbe állítva.")

        # Ha az ár új mélypontot csinál ÉS az RSI is lejjebb megy, ez nem divergencia, hanem trend folytatás
        elif last_low < tf_state["buy_ref_low_price"] and last_rsi < tf_state["buy_ref_low_rsi"]:
            tf_state["buy_ref_low_price"] = last_low
            tf_state["buy_ref_low_rsi"] = last_rsi
            tf_state["buy_ref_low_time"] = last_iso

    # =========================================================
    # ELADÁSI (BEARISH) LOGIKA - Feltétel 1: Csak 60 felett gyűjtünk
    # =========================================================
    sell_state = tf_state["sell_state"]
    
    if sell_state == "IDLE":
        # Feltétel 1: Csak akkor kezdjük el a gyűjtést, ha az RSI 60 fölé emelkedett
        if last_rsi > 60:
            tf_state["sell_state"] = "TRACKING_HIGH"
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso
            print(f"   [{tf}] {symbol} | SELL: Feltétel 1 teljesült (RSI > 60). Figyeljük a High pontokat...")

    elif sell_state == "TRACKING_HIGH":
        # Ha magasabb High jön, frissítjük a referencia pontot
        if last_high > tf_state["sell_ref_high_price"]:
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso
        
        # Ha az RSI lepattan a csúcsról (Feltétel 2 első fele)
        if last_rsi <= tf_state["sell_ref_high_rsi"] - RSI_REBOUND_MIN:
            tf_state["sell_state"] = "WAIT_DIVERGENCE_SELL"
            print(f"   [{tf}] {symbol} | SELL: Lepattanás (-{RSI_REBOUND_MIN} RSI). Várjuk az új csúcsot divergenciával...")

    elif sell_state == "WAIT_DIVERGENCE_SELL":
        # Feltétel 2 második fele: Újra emelkedik, de RSI legalább RSI_REBOUND_MIN-nel alacsonyabb
        if last_high >= tf_state["sell_ref_high_price"] and (tf_state["sell_ref_high_rsi"] - last_rsi) >= RSI_REBOUND_MIN:
            if tf_state.get("last_sell_alert_candle") != last_iso:
                subject = f"{tf} | Nasdaq ELADÁSI lehetőség - RSI Divergencia"
                body = build_email_body(
                    tf=tf, symbol=symbol, display_name=display_name, side="SELL",
                    time_str=time_str, 
                    ref={'t': tf_state["sell_ref_high_time"], 'p': tf_state["sell_ref_high_price"], 'r': tf_state["sell_ref_high_rsi"]},
                    last_price=last_high, last_rsi=last_rsi,
                    price_diff=last_high - tf_state["sell_ref_high_price"],
                    rsi_diff=tf_state["sell_ref_high_rsi"] - last_rsi
                )
                send_email(subject, body)
                tf_state["last_sell_alert_candle"] = last_iso
                
                # TELJES RESET: A következő emailhez újra 60 fölé kell mennie és lepattannia!
                tf_state["sell_state"] = "IDLE"
                print(f"   [{tf}] {symbol} | SELL: Divergencia jelezve! Rendszer alaphelyzetbe állítva.")

        # Ha az ár új csúcsot csinál ÉS az RSI is feljebb megy, ez nem divergencia
        elif last_high > tf_state["sell_ref_high_price"] and last_rsi > tf_state["sell_ref_high_rsi"]:
            tf_state["sell_ref_high_price"] = last_high
            tf_state["sell_ref_high_rsi"] = last_rsi
            tf_state["sell_ref_high_time"] = last_iso


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Futás: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
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
