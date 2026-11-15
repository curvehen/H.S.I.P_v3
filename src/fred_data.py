# src/fred_data.py - FRED Macro Data
import os, pandas as pd, requests, logging
log = logging.getLogger(__name__)

# FRED API: https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key=XXX&file_type=json
def fetch_fred_series(series_id, api_key, days=730):
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "sort_order": "asc"
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            log.warning(f"FRED {series_id} HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json().get("observations", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value']).set_index('date')['value']
        # Daily series like DGS10 have gaps on weekends, forward fill
        df = df.resample('D').ffill()
        log.info(f"✓ FRED {series_id} {len(df)} rows, latest {df.iloc[-1]:.2f} on {df.index[-1].date()}")
        return df
    except Exception as e:
        log.warning(f"FRED {series_id} failed: {e}")
        return None

def fetch_all_fred(cfg):
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        log.warning("⚠️ 沒有 FRED_API_KEY, 跳過 FRED, 只用 Yahoo 宏觀")
        return pd.DataFrame()
    fred_map = cfg.get('fred_series', {})
    data = {}
    for series_id, name in fred_map.items():
        s = fetch_fred_series(series_id, api_key)
        if s is not None and len(s) > 50:
            data[name] = s
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).ffill().bfill()
    return df

def fred_signals(fred_df):
    """Turn FRED raw values into signals for Factor explanation"""
    if fred_df.empty:
        return {}
    signals = {}
    try:
        # Yield curve: if T10Y2Y < 0, recession warning
        if 'YieldCurve' in fred_df.columns:
            yc = fred_df['YieldCurve'].iloc[-1]
            signals['YieldCurve'] = yc
            signals['YieldCurve_Signal'] = '衰退警告' if yc < 0 else '正常'
        # Fed Funds vs 10Y
        if 'FedFunds' in fred_df.columns and 'US10Y_FRED' in fred_df.columns:
            signals['FedFunds'] = fred_df['FedFunds'].iloc[-1]
            signals['US10Y_FRED'] = fred_df['US10Y_FRED'].iloc[-1]
        # CPI change YoY approx
        if 'US_CPI' in fred_df.columns:
            cpi = fred_df['US_CPI']
            if len(cpi) > 365:
                signals['US_CPI_YoY'] = (cpi.iloc[-1] / cpi.iloc[-365] - 1) * 100
        # Unemployment change
        if 'US_Unemployment' in fred_df.columns:
            un = fred_df['US_Unemployment']
            signals['US_Unemployment'] = un.iloc[-1]
            if len(un) > 90:
                signals['Unemployment_3M_Change'] = un.iloc[-1] - un.iloc[-90]
        # HY Spread - credit stress
        if 'HY_Spread' in fred_df.columns:
            hy = fred_df['HY_Spread'].iloc[-1]
            signals['HY_Spread'] = hy
            signals['Credit_Stress'] = '高壓' if hy > 5.0 else ('正常' if hy < 3.5 else '留意')
    except Exception as e:
        log.warning(f"FRED signals failed: {e}")
    return signals
