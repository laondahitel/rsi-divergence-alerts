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

RSI_PERIOD     = 14
PIVOT_LEN      = 5

# Divergencia szűrők — ideiglenesen 0.0, első tesztekhez
MIN_RSI_DIFF   = 0.0
MIN_PRICE_DIFF = 0.0

# Legalább ennyi eltárolt pivot kell az email-küldéshez
MIN_STORED_PIVOTS = 1

# A H4 gyertya hivatalos zárása után ennyi ráhagyás,
# mert a Yahoo 15m adat ~15-20 percet késik
CLOSE_BUFFER = pd.Timedelta(minutes=15)

# Ha a legutolsó lezárt H4 gyertya zárása ennél régebbi,
# a Yahoo feed beragadt -> nem küldünk emailt
MAX_DATA_AGE = pd.Timedelta(hours=5)

STATE_FILE = "state.json"
CURRENT_STATE_VERSION = 3   # v3: RSI Close (nem High/Low)

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
            if symbol in state and isinstance(state[symbol], dict):
                state[symbol]["pivot_highs"] = []
                state[symbol]["pivot_lows"]  = []
        state["_version"] = CURRENT_STATE_VERSION

    for symbol in SYMBOLS:
        sym = state.setdefault(symbol, {})
        sym.setdefault("pivot_highs", [])
        sym.setdefault("pivot_lows", [])
        sym.setdefault("last_buy_alert_candle", None)
        sym.setdefault("last_sell_alert_candle", None)


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
# RSI (Wilder's smoothing, alpha = 1/period)
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

        df4 = df.resample("4h").agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna()

        now_utc = pd.Timestamp.now(tz="UTC")
        df4 = df4[df4.index + pd.Timedelta(hours=4) + CLOSE_BUFFER <= now_utc]

        return df4
    except Exception as e:
        print(f"⚠ Hiba a letöltésnél ({symbol}): {e}")
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
# DIVERGENCIA LOGIKA — AZONNALI EMAIL
# ============================================================
def check_symbol(symbol, display_name, state):
    df = download_4h(symbol)
    if df is None or len(df) < 15:
        print(f"⚠ Kevés adat: {symbol}")
        return

    # ── Frissesség ellenőrzés ──────────────────────────────────
    last_time       = df.index[-1]
    last_close_time = last_time + pd.Timedelta(hours=4)
    now_utc         = pd.Timestamp.now(tz="UTC")
    data_age        = now_utc - last_close_time
    if data_age > MAX_DATA_AGE:
        print(
            f"⚠ Elavult Yahoo adat ({symbol}): utolsó lezárt H4 gyertya "
            f"zárása {last_close_time} ({data_age} ezelőtt). Email kihagyva."
        )
        return

    # ── RSI Close (egyetlen sorozat) ───────────────────────────
    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)

    ph, pl = find_pivots(df, PIVOT_LEN, PIVOT_LEN)
    n = len(df)

    # Pivot pozíció: High/Low. RSI érték a pivot gyertya Close-jából.
    confirmed_highs = [
        (df.index[i], float(df["High"].iloc[i]), float(df["RSI"].iloc[i]))
        for i in range(n) if ph[i]
    ]
    confirmed_lows = [
        (df.index[i], float(df["Low"].iloc[i]), float(df["RSI"].iloc[i]))
        for i in range(n) if pl[i]
    ]

    sym_state   = state.setdefault(symbol, {})
    saved_highs = sym_state.setdefault("pivot_highs", [])
    saved_lows  = sym_state.setdefault("pivot_lows", [])

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

    # ── Friss (legutolsó lezárt) H4 gyertya ────────────────────
    # A wick (Low/High) számít a pivot-átlépéshez.
    # A Close-t csak az RSI-hez használjuk.
    # Ha a wick átlépte a pivot szintet, JELEZÜNK – akkor is,
    # ha a Close visszament a pivot fölé/alá.
    last_high  = float(df["High"].iloc[-1])
    last_low   = float(df["Low"].iloc[-1])
    last_rsi   = float(df["RSI"].iloc[-1])
    last_iso   = last_time.isoformat()
    time_str   = last_time.strftime("%Y-%m-%d %H:%M UTC")

    # ============ BULLISH (vételi) ============
    if len(saved_lows) >= MIN_STORED_PIVOTS:
        ref = saved_lows[-1]
        if ref["t"] < last_iso and sym_state.get("last_buy_alert_candle") != last_iso:
            low_lower  = last_low < ref["p"]              # friss wick mélyebben
            rsi_higher = last_rsi > ref["r"]              # RSI Close magasabban
            price_ok   = (ref["p"] - last_low)   >= MIN_PRICE_DIFF
            rsi_ok     = (last_rsi - ref["r"])   >= MIN_RSI_DIFF

            if low_lower and rsi_higher and price_ok and rsi_ok:
                subject = "Nasdaq LQQ , 3QQQ VÉTELI lehetőség keletkezett, H4 RSI Divergencia"
                body = (
                    f"Nasdaq LQQ , 3QQQ VÉTELI lehetőség keletkezett, H4 RSI Divergencia\n\n"
                    f"Symbol: {symbol}\n"
                    f"Frissen lezárt H4 gyertya: {time_str}\n\n"
                    f"Referencia pivot low: {ref['p']:.4f}  RSI: {ref['r']:.2f}  ({ref['t']})\n"
                    f"Friss H4 gyertya Low:  {last_low:.4f}  RSI: {last_rsi:.2f}  ({last_iso})\n"
                    f"Ár különbség (pivot - friss): {ref['p'] - last_low:.4f}\n"
                    f"RSI különbség (friss - pivot): {last_rsi - ref['r']:.2f}\n"
                )
                send_email(subject, body)
                sym_state["last_buy_alert_candle"] = last_iso

    # ============ BEARISH (eladási) ============
    if len(saved_highs) >= MIN_STORED_PIVOTS:
        ref = saved_highs[-1]
        if ref["t"] < last_iso and sym_state.get("last_sell_alert_candle") != last_iso:
            high_higher = last_high > ref["p"]            # friss wick magasabban
            rsi_lower   = last_rsi < ref["r"]             # RSI Close lejjebb
            price_ok    = (last_high - ref["p"])  >= MIN_PRICE_DIFF
            rsi_ok      = (ref["r"] - last_rsi)   >= MIN_RSI_DIFF

            if high_higher and rsi_lower and price_ok and rsi_ok:
                subject = "Nasdaq LQQ , 3QQQ ELADÁSI lehetőség keletkezett, H4 RSI Divergencia"
                body = (
                    f"Nasdaq LQQ , 3QQQ ELADÁSI lehetőség keletkezett, H4 RSI Divergencia\n\n"
                    f"Symbol: {symbol}\n"
                    f"Frissen lezárt H4 gyertya: {time_str}\n\n"
                    f"Referencia pivot high: {ref['p']:.4f}  RSI: {ref['r']:.2f}  ({ref['t']})\n"
                    f"Friss H4 gyertya High:  {last_high:.4f}  RSI: {last_rsi:.2f}  ({last_iso})\n"
                    f"Ár különbség (friss - pivot): {last_high - ref['p']:.4f}\n"
                    f"RSI különbség (pivot - friss): {ref['r'] - last_rsi:.2f}\n"
                )
                send_email(subject, body)
                sym_state["last_sell_alert_candle"] = last_iso


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Futás: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Címzettek: {EMAIL_TO_LIST}")
    state = load_state()
    migrate_state(state)

    for symbol, name in SYMBOLS.items():
        try:
            check_symbol(symbol, name, state)
        except Exception as e:
            print(f"❌ Hiba {symbol}-nál: {e}")

    save_state(state)
    print("✅ Kész.")


if __name__ == "__main__":
    main()
