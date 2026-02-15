import os
import json
import time
import math
import random
import logging
from logging.handlers import TimedRotatingFileHandler
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta

import requests
from binance.spot import Spot as Client  # pip install binance-connector

try:
    from binance.error import ClientError, ServerError
except Exception:
    ClientError = ServerError = Exception


# =========================
# AYARLAR
# =========================
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
BASE_ASSET = SYMBOL.replace(QUOTE_ASSET, "")

USE_TESTNET = os.getenv("USE_TESTNET", "1") == "1"
LIVE_TRADING = os.getenv("LIVE_TRADING", "0") == "1"

STATE_FILE = "state.json"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))
OPTIMIZE_EVERY_SEC = int(os.getenv("OPTIMIZE_EVERY_SEC", "3600"))

# Bot bütçesi (sanal)
BOT_START_CAP_USDT = Decimal(os.getenv("BOT_START_CAP_USDT", "100"))
RESET_BUDGET = os.getenv("RESET_BUDGET", "0") == "1"

# Parametre arama (basit optimize)
CANDIDATE_INTERVALS = ["5m", "15m", "1h"]
CANDIDATE_BB_WINDOWS = [20, 30, 50]
CANDIDATE_BB_STD = [1.8, 2.0, 2.2]

# Trade boyutu (bütçe küçükse burada ayarla)
MIN_TRADE_QUOTE = Decimal(os.getenv("MIN_TRADE_QUOTE", "10"))
MAX_TRADE_QUOTE = Decimal(os.getenv("MAX_TRADE_QUOTE", "50"))

# Risk
STOP_LOSS_TOTAL_PCT = Decimal(os.getenv("STOP_LOSS_PCT", "0.03"))       # %3
MAX_DAILY_LOSS_USDT = Decimal(os.getenv("MAX_DAILY_LOSS_USDT", "2.5"))  # 2.5 USDT
MAX_DAILY_LOSS_PCT = Decimal(os.getenv("MAX_DAILY_LOSS_PCT", "0.06"))   # %6
LIQUIDATE_ON_KILL = os.getenv("LIQUIDATE_ON_KILL", "1") == "1"
COOLDOWN_AFTER_SELL_SEC = int(os.getenv("COOLDOWN_AFTER_SELL_SEC", "900"))

# Fee / Slippage
DEFAULT_TAKER_FEE = Decimal(os.getenv("DEFAULT_TAKER_FEE", "0.001"))  # %0.10
DEFAULT_MAKER_FEE = Decimal(os.getenv("DEFAULT_MAKER_FEE", "0.001"))

BACKTEST_SLIP_PER_SIDE = Decimal(os.getenv("BACKTEST_SLIP_PER_SIDE", "0.0002"))
EDGE_SAFETY = Decimal(os.getenv("EDGE_SAFETY", "0.0005"))
NEAR_TRIGGER_PCT = Decimal(os.getenv("NEAR_TRIGGER_PCT", "0.015"))  # %1.5 hedefe yaklaşınca tetik


# =========================
# LOGGING
# =========================
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_BACKUP_DAYS = int(os.getenv("LOG_BACKUP_DAYS", "14"))
LOG_CONSOLE = os.getenv("LOG_CONSOLE", "1") == "1"

# =========================
# FORMAT (TR)
# =========================
def tr_num(x, nd=2) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def now_tr() -> str:
    return datetime.now().strftime("%d.%m.%Y,%H.%M")


# =========================
# LOGGER
# =========================
_LOGGER = None


def setup_logging():
    """Dosyaya ve (opsiyonel) konsola log basar.

    - logs/bot.log içine yazar
    - Her gece (00:00'da) yeni dosya açar (TimedRotatingFileHandler)
    """
    global _LOGGER

    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("bot")

    # aynı process içinde tekrar çağrılırsa handler çoğalmasın
    if logger.handlers:
        _LOGGER = logger
        return logger

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=LOG_BACKUP_DAYS,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    if LOG_CONSOLE:
        console = logging.StreamHandler()
        # Eski print çıktısını bozmayalım
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)

    _LOGGER = logger
    return logger


def log():
    # Lazy init: main() öncesi bir yerde çağrı gelirse de çalışsın
    return _LOGGER or setup_logging()


# =========================
# UTIL
# =========================
def D(x) -> Decimal:
    return Decimal(str(x))

def short_err(e: Exception) -> str:
    msg = str(e).replace("\n", " ")
    return (msg[:180] + "...") if len(msg) > 180 else msg

def retry_call(fn, *args, retries=4, base_sleep=2, **kwargs):
    # Emirlerde kullanılmaz (çift emir riski)
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (ServerError, ClientError) as e:
            wait = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log().warning(f"[API RETRY] {attempt}/{retries} -> {short_err(e)} | sleep {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            wait = base_sleep + random.uniform(0, 0.5)
            log().warning(f"[API RETRY] -> {short_err(e)} | sleep {wait:.1f}s")
            time.sleep(wait)
    return None

def mean(xs):
    return sum(xs) / len(xs)

def stdev(xs):
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return math.sqrt(v)

def bollinger(closes, window, k):
    w = closes[-window:]
    m = mean(w)
    sd = stdev(w)
    upper = m + k * sd
    lower = m - k * sd
    return lower, m, upper

def floor_step(qty: Decimal, step: Decimal) -> Decimal:
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step

def today_str() -> str:
    return datetime.now().date().isoformat()

def next_local_midnight_ts() -> int:
    now = datetime.now()
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(nxt.timestamp())


# =========================
# STATE
# =========================
def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def default_state():
    return {
        "bot_quote": None,
        "bot_base": 0.0,
        "budget_cap": float(BOT_START_CAP_USDT),

        "pos": "NONE",
        "entry_price": None,     # float
        "buy_cost_rate": 0.0,    # float

        "last_opt": 0,
        "best": None,

        "day": today_str(),
        "day_start_equity": None,   # float
        "start_equity": None,       # float (reset budget ile tekrar set)
        "pause_until": 0,
        "cooldown_until": 0,
        "kill_triggered": False,
    }


# =========================
# BINANCE CLIENT
# =========================
def get_client():
    base_url = "https://testnet.binance.vision" if USE_TESTNET else None
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("BINANCE_API_KEY ve BINANCE_API_SECRET gerekli.")
    return Client(api_key=api_key, api_secret=api_secret, base_url=base_url)

def fetch_klines(client, interval, limit=500):
    data = retry_call(client.klines, symbol=SYMBOL, interval=interval, limit=limit)
    if not data:
        return None
    return [float(k[4]) for k in data]  # close

def live_price(client):
    t = retry_call(client.ticker_price, symbol=SYMBOL)
    if not t:
        return None
    return D(t["price"])

def account_balances(client):
    acc = retry_call(client.account)
    if not acc:
        return None, None
    base_free = Decimal("0")
    quote_free = Decimal("0")
    for b in acc["balances"]:
        if b["asset"] == BASE_ASSET:
            base_free = D(b["free"])
        if b["asset"] == QUOTE_ASSET:
            quote_free = D(b["free"])
    return base_free, quote_free

def symbol_filters(client):
    info = retry_call(client.exchange_info, symbol=SYMBOL)
    if not info:
        return None, None
    s = info["symbols"][0]
    step = Decimal("0.00000001")
    min_notional = Decimal("0")
    for f in s["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step = D(f["stepSize"])
        if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
            mn = f.get("minNotional")
            if mn is not None:
                min_notional = D(mn)
    return step, min_notional


# =========================
# FEE (SAPI) - testnette genelde yok -> fallback
# =========================
def signed_get(url: str, api_key: str, api_secret: str, params: dict) -> dict:
    import hmac, hashlib, urllib.parse
    qs = urllib.parse.urlencode(params, doseq=True)
    sig = hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    full = f"{url}?{qs}&signature={sig}"
    r = requests.get(full, headers={"X-MBX-APIKEY": api_key}, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_trade_fees(api_key: str, api_secret: str) -> tuple[Decimal, Decimal]:
    if USE_TESTNET:
        return DEFAULT_MAKER_FEE, DEFAULT_TAKER_FEE
    url = "https://api.binance.com/sapi/v1/asset/tradeFee"
    try:
        data = signed_get(url, api_key, api_secret, {"symbol": SYMBOL, "timestamp": int(time.time() * 1000)})
        row = data[0]
        return D(row["makerCommission"]), D(row["takerCommission"])
    except Exception as e:
        log().warning("[FEE] tradeFee çekilemedi, fallback: " + short_err(e))
        return DEFAULT_MAKER_FEE, DEFAULT_TAKER_FEE


# =========================
# ORDERBOOK SLIPPAGE (opsiyonel basit)
# =========================
def get_orderbook(client, limit=20):
    return retry_call(client.depth, symbol=SYMBOL, limit=limit)

def estimate_buy_slippage_from_quote(client, quote_qty: Decimal, fallback_px: Decimal) -> tuple[Decimal, Decimal]:
    ob = get_orderbook(client, limit=20)
    if not ob:
        return fallback_px, D("0")
    asks = [(D(p), D(q)) for p, q in ob["asks"]]
    bid = D(ob["bids"][0][0])
    ask = D(ob["asks"][0][0])
    mid = (bid + ask) / D("2")

    remaining = quote_qty
    bought = D("0")
    spent = D("0")
    for price, qty in asks:
        max_quote_here = price * qty
        take = remaining if remaining <= max_quote_here else max_quote_here
        bought += take / price
        spent += take
        remaining -= take
        if remaining <= 0:
            break

    if bought <= 0:
        return mid, D("0")
    avg = spent / bought
    slip = (avg / mid) - D("1")
    if slip < 0:
        slip = D("0")
    return avg, slip

def estimate_sell_slippage_from_base(client, base_qty: Decimal, fallback_px: Decimal) -> tuple[Decimal, Decimal]:
    ob = get_orderbook(client, limit=20)
    if not ob:
        return fallback_px, D("0")
    bids = [(D(p), D(q)) for p, q in ob["bids"]]
    bid = D(ob["bids"][0][0])
    ask = D(ob["asks"][0][0])
    mid = (bid + ask) / D("2")

    remaining = base_qty
    sold = D("0")
    proceeds = D("0")
    for price, qty in bids:
        take = remaining if remaining <= qty else qty
        proceeds += take * price
        sold += take
        remaining -= take
        if remaining <= 0:
            break

    if sold <= 0:
        return mid, D("0")
    avg = proceeds / sold
    slip = D("1") - (avg / mid)
    if slip < 0:
        slip = D("0")
    return avg, slip


# =========================
# ORDERS (NO retry!)
# =========================
def place_market_buy(client, quote_qty: Decimal):
    params = dict(symbol=SYMBOL, side="BUY", type="MARKET", quoteOrderQty=str(quote_qty))
    if not LIVE_TRADING:
        return None
    return client.new_order(**params)

def place_market_sell(client, qty: Decimal):
    params = dict(symbol=SYMBOL, side="SELL", type="MARKET", quantity=str(qty))
    if not LIVE_TRADING:
        return None
    return client.new_order(**params)


# =========================
# BACKTEST / OPT
# =========================
def backtest_score(closes, window, k, taker_fee: float, slip_side: float) -> float:
    if len(closes) < window + 5:
        return -1e18
    cash = 1.0
    coin = 0.0
    for i in range(window, len(closes)):
        sub = closes[:i + 1]
        lower, mid, _ = bollinger(sub, window, k)
        price = sub[-1]
        if coin == 0.0 and price < lower:
            effective_cash = cash * (1.0 - taker_fee)
            buy_px = price * (1.0 + slip_side)
            coin = effective_cash / buy_px
            cash = 0.0
        elif coin > 0.0 and price > mid:
            sell_px = price * (1.0 - slip_side)
            cash = (coin * sell_px) * (1.0 - taker_fee)
            coin = 0.0
    last = closes[-1]
    total = cash + coin * last
    return total

def optimize_params(client, taker_fee: Decimal) -> tuple[dict, float]:
    best = None
    best_score = -1e18
    tf = float(taker_fee)
    slip = float(BACKTEST_SLIP_PER_SIDE)

    for interval in CANDIDATE_INTERVALS:
        closes = fetch_klines(client, interval, limit=500)
        if closes is None:
            continue
        for w in CANDIDATE_BB_WINDOWS:
            for kk in CANDIDATE_BB_STD:
                score = backtest_score(closes, w, kk, tf, slip)
                if score > best_score:
                    best_score = score
                    best = {"interval": interval, "window": w, "k": kk}
    return best, best_score


# =========================
# BUDGET / EQUITY
# =========================
def bot_equity(state, px: Decimal) -> Decimal:
    bot_quote = D(state["bot_quote"])
    bot_base = D(state["bot_base"])
    return bot_quote + bot_base * px

def init_budget_if_needed(state, real_quote_free: Decimal):
    if state.get("bot_quote") is not None and not RESET_BUDGET:
        return state

    cap = BOT_START_CAP_USDT
    start_quote = real_quote_free if real_quote_free < cap else cap

    state["budget_cap"] = float(cap)
    state["bot_quote"] = float(start_quote)
    state["bot_base"] = 0.0

    state["pos"] = "NONE"
    state["entry_price"] = None
    state["buy_cost_rate"] = 0.0

    state["day"] = today_str()
    state["day_start_equity"] = None
    state["start_equity"] = float(start_quote)  # toplam PnL referansı
    state["pause_until"] = 0
    state["cooldown_until"] = 0
    state["kill_triggered"] = False

    save_state(state)
    return state

def trigger_kill_switch(state, reason: str):
    if state.get("kill_triggered"):
        return
    state["pause_until"] = next_local_midnight_ts()
    state["kill_triggered"] = True
    save_state(state)


# =========================
# LOG LINE (İSTEDİĞİN FORMAT)
# =========================
def print_status_line(
    action_text: str,
    planned_text: str,
    px_live: Decimal,
    bb_lower: Decimal,
    bb_mid: Decimal,
    dd_usdt: Decimal,
    dd_pct: Decimal,
    pnl_usdt_total: Decimal,
    pnl_pct_total: Decimal,
    state: dict,
):
    bot_base = D(state["bot_base"])
    bot_quote = D(state["bot_quote"])

    line = (
        f"Tarih:{now_tr()} - "
        f"İşlem:{action_text} - "
        f"Hesap: {tr_num(float(bot_base), 8)} {BASE_ASSET}, {tr_num(float(bot_quote), 2)} {QUOTE_ASSET} - "
        f"px: {BASE_ASSET}={tr_num(float(px_live), 2)} {QUOTE_ASSET} - "
        f"BB: lower={tr_num(float(bb_lower), 2)} - mid={tr_num(float(bb_mid), 2)} - "
        f"DD: {tr_num(float(dd_usdt), 2)} ({tr_num(float(dd_pct*100), 2)}%) - "
        f"Plan:{planned_text} - "
        f"PnL: {tr_num(float(pnl_pct_total*100), 2)}% ({tr_num(float(pnl_usdt_total), 2)} {QUOTE_ASSET})"
    )
    log().info(line)


# =========================
# MAIN
# =========================
def main():
    setup_logging()
    log().info("[START] Bot çalıştı")

    client = get_client()

    step, min_notional = symbol_filters(client)
    if step is None:
        log().error("[FATAL] exchange_info alınamadı.")
        return

    state = load_state() or default_state()
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    maker_fee, taker_fee = fetch_trade_fees(api_key, api_secret)

    # İlk optimize için
    if not state.get("best"):
        state["best"] = {"interval": "1h", "window": 30, "k": 1.8}
        save_state(state)

    while True:
        now_ts = int(time.time())
        try:
            # Gün değiştiyse reset
            if state.get("day") != today_str():
                state["day"] = today_str()
                state["day_start_equity"] = None
                state["pause_until"] = 0
                state["cooldown_until"] = 0
                state["kill_triggered"] = False
                save_state(state)

            # Hesap bakiyesi (sadece bütçeyi init için)
            base_free, quote_free = account_balances(client)
            if base_free is None:
                time.sleep(POLL_SECONDS)
                continue

            # Bot bütçesini başlat
            state = init_budget_if_needed(state, quote_free)

            # Optimize (belirli aralıklarla)
            if now_ts - int(state.get("last_opt", 0)) >= OPTIMIZE_EVERY_SEC:
                best, score = optimize_params(client, taker_fee)
                if best:
                    state["best"] = best
                    state["last_opt"] = now_ts
                    save_state(state)
                    log().info(f"[OPT] best={best} score={score:.6f}")

            best = state["best"]

            closes = fetch_klines(client, best["interval"], limit=500)
            if closes is None:
                time.sleep(POLL_SECONDS)
                continue

            bb_lower_f, bb_mid_f, _ = bollinger(closes, best["window"], best["k"])
            bb_lower = D(bb_lower_f)
            bb_mid = D(bb_mid_f)

            # Strateji fiyatı (kline close) + canlı px (log için)
            px_strategy = D(closes[-1])
            px_live = live_price(client) or px_strategy

            # Tetik fiyatları (bu stratejide)
            buy_trigger = bb_lower   # ALIM: px_strategy < bb_lower
            sell_trigger = bb_mid    # SATIM: px_strategy > bb_mid

            # Equity / DD / PnL
            eq = bot_equity(state, px_live)

            if state.get("day_start_equity") is None:
                state["day_start_equity"] = float(eq)
                save_state(state)

            day_start = D(state["day_start_equity"])
            dd_usdt = day_start - eq
            dd_pct = (dd_usdt / day_start) if day_start > 0 else D("0")

            start_eq = D(state.get("start_equity") or float(eq))
            pnl_usdt_total = eq - start_eq
            pnl_pct_total = (pnl_usdt_total / start_eq) if start_eq > 0 else D("0")

            bot_quote = D(state["bot_quote"])
            bot_base = D(state["bot_base"])
            holding = bot_base > D("0.00000001")

            action_text = "BEKLE"
            planned_text = "-"

            # Kill-switch kontrol
            kill_hit = False
            reasons = []
            if MAX_DAILY_LOSS_USDT > 0 and dd_usdt >= MAX_DAILY_LOSS_USDT:
                kill_hit = True
                reasons.append(f"DD_USDT>={MAX_DAILY_LOSS_USDT}")
            if MAX_DAILY_LOSS_PCT > 0 and dd_pct >= MAX_DAILY_LOSS_PCT:
                kill_hit = True
                reasons.append(f"DD_%>={MAX_DAILY_LOSS_PCT}")

            if kill_hit:
                if holding and LIQUIDATE_ON_KILL:
                    sell_qty = floor_step(bot_base, step)
                    if sell_qty * px_live >= min_notional and sell_qty > 0:
                        if LIVE_TRADING:
                            place_market_sell(client, sell_qty)

                        # bot iç cüzdan güncelle (yaklaşık)
                        avg_sell, _ = estimate_sell_slippage_from_base(client, sell_qty, px_live)
                        proceeds = (sell_qty * avg_sell) * (D("1") - taker_fee)

                        state["bot_base"] = float(bot_base - sell_qty)
                        state["bot_quote"] = float(bot_quote + proceeds)
                        state["pos"] = "NONE"
                        state["entry_price"] = None
                        state["buy_cost_rate"] = 0.0
                        state["cooldown_until"] = now_ts + COOLDOWN_AFTER_SELL_SEC
                        save_state(state)

                        action_text = f"{tr_num(float(proceeds),2)} {QUOTE_ASSET} SATIM (KILL)"
                trigger_kill_switch(state, " | ".join(reasons))
                planned_text = "PAUSE (yarına kadar)"

                # log
                eq2 = bot_equity(state, px_live)
                pnl_usdt_total = eq2 - start_eq
                pnl_pct_total = (pnl_usdt_total / start_eq) if start_eq > 0 else D("0")
                print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                time.sleep(POLL_SECONDS)
                continue

            # Pause / cooldown
            if now_ts < int(state.get("pause_until", 0)):
                planned_text = "PAUSE"
                print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                time.sleep(POLL_SECONDS)
                continue

            if now_ts < int(state.get("cooldown_until", 0)):
                planned_text = "COOLDOWN"
                print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                time.sleep(POLL_SECONDS)
                continue

            # STOP-LOSS (pozisyondayken)
            entry_price = state.get("entry_price")
            buy_cost_rate = D(state.get("buy_cost_rate", 0.0))

            if holding and entry_price and STOP_LOSS_TOTAL_PCT > 0:
                sell_qty = floor_step(bot_base, step)
                _, slip_sell = estimate_sell_slippage_from_base(client, sell_qty, px_live)

                expected_total_cost = buy_cost_rate + taker_fee + slip_sell
                effective_drop = STOP_LOSS_TOTAL_PCT - expected_total_cost
                if effective_drop < D("0.001"):
                    effective_drop = D("0.001")

                stop_price = D(entry_price) * (D("1") - effective_drop)

                if px_live <= stop_price:
                    planned_text = f"SATIM (STOP) qty={tr_num(float(sell_qty),8)}"
                    if sell_qty * px_live >= min_notional and sell_qty > 0:
                        if LIVE_TRADING:
                            place_market_sell(client, sell_qty)

                        avg_sell, _ = estimate_sell_slippage_from_base(client, sell_qty, px_live)
                        proceeds = (sell_qty * avg_sell) * (D("1") - taker_fee)

                        state["bot_base"] = float(bot_base - sell_qty)
                        state["bot_quote"] = float(bot_quote + proceeds)
                        state["pos"] = "NONE"
                        state["entry_price"] = None
                        state["buy_cost_rate"] = 0.0
                        state["cooldown_until"] = now_ts + COOLDOWN_AFTER_SELL_SEC
                        save_state(state)

                        action_text = f"{tr_num(float(proceeds),2)} {QUOTE_ASSET} SATIM (STOP)"
                    print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                    time.sleep(POLL_SECONDS)
                    continue

            # =========================
            # STRATEJİ:
            # - AL: px < lower (veya hedefe NEAR_TRIGGER_PCT yaklaşınca)
            # - SAT: (holding) px > mid (veya hedefe NEAR_TRIGGER_PCT yaklaşınca)
            # + maliyet filtresi
            # =========================
            if not holding:
                # Planlanan ALIM (sinyal oluşursa)
                planned_buy = min(MAX_TRADE_QUOTE, bot_quote * D("0.5"))

                gap = px_live - buy_trigger
                gap_pct = (gap / px_live) if px_live > 0 else D("0")

                near_buy = (gap_pct <= NEAR_TRIGGER_PCT)

                if gap > 0:
                    suffix = " (YAKIN)" if near_buy else ""
                    planned_text = (
                        f"ALIM {tr_num(float(planned_buy),2)} {QUOTE_ASSET} "
                        f"@<= {tr_num(float(buy_trigger),2)} "
                        f"(kalan {tr_num(float(gap),2)} / {tr_num(float(gap_pct*100),2)}%)"
                        f"{suffix}"
                    )
                else:
                    planned_text = (
                        f"ALIM {tr_num(float(planned_buy),2)} {QUOTE_ASSET} "
                        f"@<= {tr_num(float(buy_trigger),2)} (TETİK)"
                    )

                if (px_strategy < bb_lower) or near_buy:
                    use = planned_buy
                    if use < MIN_TRADE_QUOTE:
                        planned_text = "ALIM yok (yetersiz bakiye)"
                    else:
                        avg_buy, slip_buy = estimate_buy_slippage_from_quote(client, use, px_live)
                        est_base = use / avg_buy if avg_buy > 0 else D("0")
                        _, slip_sell = estimate_sell_slippage_from_base(client, est_base, px_live)

                        roundtrip_cost = (taker_fee * D("2")) + slip_buy + slip_sell + EDGE_SAFETY
                        expected_edge = (bb_mid - px_strategy) / px_strategy if px_strategy > 0 else D("0")

                        if expected_edge <= roundtrip_cost:
                            planned_text = "ALIM pas (fee+slip yiyor)"
                        else:
                            # BUY
                            if LIVE_TRADING:
                                place_market_buy(client, use)

                            acquired_base = (use * (D("1") - taker_fee)) / avg_buy
                            state["bot_quote"] = float(bot_quote - use)
                            state["bot_base"] = float(bot_base + acquired_base)
                            state["pos"] = "LONG"
                            state["entry_price"] = float(avg_buy)
                            state["buy_cost_rate"] = float(taker_fee + slip_buy)
                            save_state(state)

                            tag = " (YAKIN)" if (near_buy and not (px_strategy < bb_lower)) else ""
                            action_text = f"{tr_num(float(use),2)} {QUOTE_ASSET} ALIM{tag}"
                            planned_text = "-"
                # log
                eq2 = bot_equity(state, px_live)
                pnl_usdt_total = eq2 - start_eq
                pnl_pct_total = (pnl_usdt_total / start_eq) if start_eq > 0 else D("0")
                print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                time.sleep(POLL_SECONDS)
                continue

            else:
                sell_qty = floor_step(bot_base, step)
                est_proceeds = (sell_qty * px_live) * (D("1") - taker_fee)

                gap = sell_trigger - px_live
                gap_pct = (gap / px_live) if px_live > 0 else D("0")

                near_sell = (gap_pct <= NEAR_TRIGGER_PCT)

                if gap > 0:
                    suffix = " (YAKIN)" if near_sell else ""
                    planned_text = (
                        f"SATIM ~{tr_num(float(est_proceeds),2)} {QUOTE_ASSET} "
                        f"@>= {tr_num(float(sell_trigger),2)} "
                        f"(kalan {tr_num(float(gap),2)} / {tr_num(float(gap_pct*100),2)}%)"
                        f"{suffix}"
                    )
                else:
                    planned_text = (
                        f"SATIM ~{tr_num(float(est_proceeds),2)} {QUOTE_ASSET} "
                        f"@>= {tr_num(float(sell_trigger),2)} (TETİK)"
                    )

                if (px_strategy > bb_mid) or near_sell:
                    if sell_qty * px_live >= min_notional and sell_qty > 0:
                        if LIVE_TRADING:
                            place_market_sell(client, sell_qty)

                        avg_sell, _ = estimate_sell_slippage_from_base(client, sell_qty, px_live)
                        proceeds = (sell_qty * avg_sell) * (D("1") - taker_fee)

                        state["bot_base"] = float(bot_base - sell_qty)
                        state["bot_quote"] = float(bot_quote + proceeds)
                        state["pos"] = "NONE"
                        state["entry_price"] = None
                        state["buy_cost_rate"] = 0.0
                        state["cooldown_until"] = now_ts + COOLDOWN_AFTER_SELL_SEC
                        save_state(state)

                        tag = " (YAKIN)" if (near_sell and not (px_strategy > bb_mid)) else ""
                        action_text = f"{tr_num(float(proceeds),2)} {QUOTE_ASSET} SATIM{tag}"
                        planned_text = "-"
                # log
                eq2 = bot_equity(state, px_live)
                pnl_usdt_total = eq2 - start_eq
                pnl_pct_total = (pnl_usdt_total / start_eq) if start_eq > 0 else D("0")
                print_status_line(action_text, planned_text, px_live, bb_lower, bb_mid, dd_usdt, dd_pct, pnl_usdt_total, pnl_pct_total, state)
                time.sleep(POLL_SECONDS)
                continue

        except Exception as e:
            log().exception("[LOOP ERROR] " + short_err(e))
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()