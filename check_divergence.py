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

# Figyelt idősíkok
TIMEFRAMES = ["H1", "H4"]

RSI_PERIOD     = 14
PIVOT_LEN      = 5

# Divergencia szűrők — egyelőre 0.0
MIN_RSI_DIFF   = 0.0
MIN_PRICE_DIFF = 0.0

MIN_STORED_PIVOTS = 1

# A gyertya hivatalos zárása után ennyi ráhagyás (Yahoo késés miatt)
CLOSE_BUFFER = pd.Timedelta(minutes=15)

# Frissességi limitek idősíkonként
MAX_DATA_AGE = {
    "H1": pd.Timedelta(minutes=90),
    "H4": pd.Timedelta(hours=5),
}

# pandas resample szabály idősíkonként
RESAMPLE_RULE = {
    "H1": "1h",
    "H4": "4h",
}

# Egy gyertya hossza idősíkonként (a zárás számításához)
BAR_DURATION = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
}

STATE_FILE = "state.json"
CURRENT_STATE_VERSION = 4   # v4: H1 + H4 struktúra

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

    # Verzióváltás: H1+H4 struktúra bevezetése, mindent újraépítünk
    if state.get("_version") != CURRENT_STATE_VERSION:
        for symbol in SYMBOLS:
            state.pop(symbol, None)
        state["_version"] = CURRENT_STATE_VERSION

    for symbol in SYMBOLS:
        sym = state.setdefault(symbol, {})
        for tf in TIMEFRAMES:
            tf_state = sym.setdefault(tf, {})
            tf_state.setdefault("pivot_highs", [])
            tf_state.setdefault("pivot_lows", [])
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
    """
    Egy Yahoo-hívást indít, majd az adatot a kért idősíkra resampleli.
    A visszatérő df csak a biztosan lezárt gyertyákat tartalmazza.
    """
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

        # Resample az idősíkra
        df_tf = df.resample(RESAMPLE_RULE[tf]).agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

        # Csak a biztosan lezárt gyertyák
        now_utc = pd.Timestamp.now(tz="UTC")
        df_tf = df_tf[df_tf.index + BAR_DURATION[tf] + CLOSE_BUFFER <= now_utc]

        return df_tf
    except Exception as e:
        print(f"⚠ Hiba a letöltésnél ({symbol} [{tf}]): {e}")
        return None


# ============================================================
# PIVOT DETEKTÁLÁS
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
# EMAIL FORMÁZÁS
# ============================================================
def build_email_body(tf, symbol, display_name, side, time_str, ref, last_price, last_rsi,
                     price_diff, rsi_diff):
    """
    Az idősíkot nagyban, feltűnően jeleníti meg a levél tetején.
    side: 'BUY' vagy 'SELL'
    ref: { 't': iso_timestamp, 'p': price, 'r': rsi }
    """
    side_hu = "VÉTELI" if side == "BUY" else "ELADÁSI"
    ref_label = "pivot low" if side == "BUY" else "pivot high"
    fresh_label = "Low" if side == "BUY" else "High"
    tf_full = "1 órás" if tf == "H1" else "4 órás"

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
# DIVERGENCIA LOGIKA — egy (symbol, timeframe) párra
# ============================================================
def check_symbol_tf(symbol, display_name, tf, state):
    df = download_tf(symbol, tf)
    if df is None or len(df) < 15:
        print(f"⚠ Kevés adat: {symbol} [{tf}]")
        return

    # ── Frissesség ellenőrzés ──────────────────────────────────
    last_time       = df.index[-1]
    last_close_time = last_time + BAR_DURATION[tf]
    now_utc         = pd.Timestamp.now(tz="UTC")
    data_age        = now_utc - last_close_time
    age_limit       = MAX_DATA_AGE[tf]
    if data_age > age_limit:
        print(
            f"⚠ Elavult Yahoo adat ({symbol} [{tf}]): utolsó lezárt gyertya "
            f"zárása {last_close_time} ({data_age} ezelőtt, limit={age_limit}). Email kihagyva."
        )
        return

    # ── RSI Close ──────────────────────────────────────────────
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    ph, pl = find_pivots(df, PIVOT_LEN, PIVOT_LEN)
    n = len(df)

    confirmed_highs = [
        (df.index[i], float(df["High"].iloc[i]), float(df["RSI"].iloc[i]))
        for i in range(n) if ph[i]
    ]
    confirmed_lows = [
        (df.index[i], float(df["Low"].iloc[i]), float(df["RSI"].iloc[i]))
        for i in range(n) if pl[i]
    ]

    # Idősíkonként külön pivot lista
    sym_state = state.setdefault(symbol, {})
    tf_state  = sym_state.setdefault(tf, {})
    saved_highs = tf_state.setdefault("pivot_highs", [])
    saved_lows  = tf_state.setdefault("pivot_lows", [])

    def merge(saved, new_items):
        seen = {p["t"] for p in saved}
        for t, price, rsi in new_items:
            iso = t.isoformat()
            if iso not in seen:
                saved.append({"t": iso, "p": price, "r": rsi})
                seen.add(iso)
        saved.sort(key=lambda x: x["t"])

    merge(saved_highs, confirmed_highs)
    merge(saved_lows,  confirmed_lows)

    print(
        f"   [{tf}] {symbol} | utolsó lezárt: {last_time} ({data_age} ezelőtt) "
        f"| pivot high: {len(saved_highs)} | pivot low: {len(saved_lows)}"
    )

    # ── Friss gyertya (wick) ──────────────────────────────────
    last_high  = float(df["High"].iloc[-1])
    last_low   = float(df["Low"].iloc[-1])
    last_rsi   = float(df["RSI"].iloc[-1])
    last_iso   = last_time.isoformat()
    time_str   = last_time.strftime("%Y-%m-%d %H:%M UTC")

    # ============ BULLISH (vételi) ============
    if len(saved_lows) >= MIN_STORED_PIVOTS:
        ref = saved_lows[-1]
        if ref["t"] < last_iso and tf_state.get("last_buy_alert_candle") != last_iso:
            low_lower  = last_low < ref["p"]
            rsi_higher = last_rsi > ref["r"]
            price_ok   = (ref["p"] - last_low)   >= MIN_PRICE_DIFF
            rsi_ok     = (last_rsi - ref["r"])   >= MIN_RSI_DIFF

            if low_lower and rsi_higher and price_ok and rsi_ok:
                subject = (
                    f"{tf} | Nasdaq LQQ , 3QQQ VÉTELI lehetőség keletkezett, RSI Divergencia"
                )
                body = build_email_body(
                    tf=tf, symbol=symbol, display_name=display_name, side="BUY",
                    time_str=time_str, ref=ref,
                    last_price=last_low, last_rsi=last_rsi,
                    price_diff=ref["p"] - last_low,
                    rsi_diff=last_rsi - ref["r"]
                )
                send_email(subject, body)
                tf_state["last_buy_alert_candle"] = last_iso

    # ============ BEARISH (eladási) ============
    if len(saved_highs) >= MIN_STORED_PIVOTS:
        ref = saved_highs[-1]
        if ref["t"] < last_iso and tf_state.get("last_sell_alert_candle") != last_iso:
            high_higher = last_high > ref["p"]
            rsi_lower   = last_rsi < ref["r"]
            price_ok    = (last_high - ref["p"])  >= MIN_PRICE_DIFF
            rsi_ok      = (ref["r"] - last_rsi)   >= MIN_RSI_DIFF

            if high_higher and rsi_lower and price_ok and rsi_ok:
                subject = (
                    f"{tf} | Nasdaq LQQ , 3QQQ ELADÁSI lehetőség keletkezett, RSI Divergencia"
                )
                body = build_email_body(
                    tf=tf, symbol=symbol, display_name=display_name, side="SELL",
                    time_str=time_str, ref=ref,
                    last_price=last_high, last_rsi=last_rsi,
                    price_diff=last_high - ref["p"],
                    rsi_diff=ref["r"] - last_rsi
                )
                send_email(subject, body)
                tf_state["last_sell_alert_candle"] = last_iso


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Futás: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Címzettek: {EMAIL_TO_LIST}")
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
