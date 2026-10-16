# Figyelt idősíkok
TIMEFRAMES = ["H1", "H4", "D1", "W1"]

# ... (RSI_PERIOD, PIVOT_LEN marad) ...

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

# Egy gyertya hossza idősíkonként
BAR_DURATION = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
    "W1": pd.Timedelta(weeks=1),
}

CURRENT_STATE_VERSION = 6   # v6: Új state machine logika + D1/W1 támogatás
