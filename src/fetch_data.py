# src/fetch_data.py
import yfinance as yf, pandas as pd, time, logging
log = logging.getLogger(__name__)
def fetch_one(ticker, retries=3):
    for i in range(retries):
        try:
            df = yf.download(ticker, period='2y', interval='1d', progress=False, auto_adjust=True)
            if df.empty: raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df['Close']
        except Exception as e:
            log.warning(f"{ticker} 第{i+1}次失敗: {e}")
            time.sleep(1)
    return None
def fetch_all(tickers):
    data = {}
    for t in tickers:
        s = fetch_one(t)
        if s is not None and len(s) > 50:
            data[t] = s; log.info(f"✓ {t} {len(s)} 日")
        else:
            log.warning(f"✗ {t} 跳過")
    prices = pd.DataFrame(data).ffill().bfill().dropna(how='all')
    if prices.empty: raise RuntimeError("無數據, 檢查網絡")
    return prices
