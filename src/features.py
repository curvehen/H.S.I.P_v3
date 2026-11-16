# src/features.py - FULL 版 - 每隻獨立特徵 + 恒指獨特性 + Fix 偷睇版
import pandas as pd, numpy as np

def get_ticker_seed(ticker, base_seed=42):
    """
    每隻股票自己嘅 Seed
    0700.HK -> 固定一個數, 每次 Run 都一樣
    1810.HK -> 另一個數, 唔會同 0700 撞
    ^HSI -> 再另一個數, +9999, 保留獨特性
    """
    s = sum(ord(c) for c in str(ticker)) * 100 + base_seed
    if ticker == '^HSI':
        s += 9999
    return int(s % 100000)

def make_features(prices_df, target, cfg, fred_df=None):
    """
    3 層特徵:
    1. 通用技術指標 - 大家都有
    2. 個股獨特性 - 每隻自己計
    3. 恒指獨特性 - 只有恒指先有
    全部 rolling() 都有 shift(1), 唔偷睇今日
    """
    df = pd.DataFrame()
    s = prices_df[target]
    df['close'] = s
    hsi_ticker = cfg.get('hsi', '^HSI')

    # ========== 1. 通用技術指標 ==========
    try:
        df['ret_1'] = s.pct_change(1)
        df['ret_5'] = s.pct_change(5)
        df['ret_20'] = s.pct_change(20)
    except:
        df['ret_1'] = 0; df['ret_5'] = 0; df['ret_20'] = 0

    try:
        d = s.diff()
        g = d.where(d > 0, 0).rolling(14).mean().shift(1)
        l = -d.where(d < 0, 0).rolling(14).mean().shift(1)
        df['rsi'] = 100 - (100 / (1 + g / (l + 1e-9)))
    except:
        df['rsi'] = 50

    try:
        e12 = s.ewm(span=12).mean().shift(1)
        e26 = s.ewm(span=26).mean().shift(1)
        df['macd'] = e12 - e26
        df['macd_signal'] = df['macd'].ewm(span=9).mean().shift(1)
    except:
        df['macd'] = 0; df['macd_signal'] = 0

    try:
        df['ma5'] = s.rolling(5).mean().shift(1)
        df['ma20'] = s.rolling(20).mean().shift(1)
        df['ma60'] = s.rolling(60).mean().shift(1)
        df['ma_ratio'] = df['ma5'] / (df['ma20'] + 1e-9)
    except:
        df['ma5'] = s; df['ma20'] = s; df['ma60'] = s; df['ma_ratio'] = 1

    try:
        df['vol_20'] = df['ret_1'].rolling(20).std().shift(1)
    except:
        df['vol_20'] = 0.01

    # ========== 2. 個股獨特性特徵 ==========
    try:
        df['own_vol_5'] = df['ret_1'].rolling(5).std().shift(1)
        df['own_vol_20'] = df['ret_1'].rolling(20).std().shift(1)
        df['vol_ratio'] = df['own_vol_5'] / (df['own_vol_20'] + 1e-9)
    except:
        df['own_vol_5'] = 0.01; df['own_vol_20'] = 0.01; df['vol_ratio'] = 1.0

    try:
        df['price_vs_ma60'] = s / (s.rolling(60).mean().shift(1) + 1e-9)
        df['price_vs_ma20'] = s / (s.rolling(20).mean().shift(1) + 1e-9)
        df['dist_52w_high'] = s / (s.rolling(252).max().shift(1) + 1e-9)
        df['dist_52w_low'] = s / (s.rolling(252).min().shift(1) + 1e-9)
    except:
        df['price_vs_ma60'] = 1.0; df['price_vs_ma20'] = 1.0
        df['dist_52w_high'] = 1.0; df['dist_52w_low'] = 1.0

    try:
        if target != hsi_ticker and hsi_ticker in prices_df.columns:
            hsi_ret_5 = prices_df[hsi_ticker].pct_change(5)
            df['alpha_vs_hsi_5'] = df['ret_5'] - hsi_ret_5
            hsi_ret_20 = prices_df[hsi_ticker].pct_change(20)
            df['alpha_vs_hsi_20'] = df['ret_20'] - hsi_ret_20
        else:
            df['alpha_vs_hsi_5'] = 0.0
            df['alpha_vs_hsi_20'] = 0.0
    except:
        df['alpha_vs_hsi_5'] = 0.0; df['alpha_vs_hsi_20'] = 0.0

    try:
        top10 = list(cfg.get('top10', {}).keys())
        if target in top10 and len(top10) > 1:
            others = [t for t in top10 if t != target and t in prices_df.columns]
            if others:
                sector_avg_5 = prices_df[others].pct_change(5).mean(axis=1)
                df['sector_alpha_5'] = df['ret_5'] - sector_avg_5
            else:
                df['sector_alpha_5'] = 0.0
        else:
            df['sector_alpha_5'] = 0.0
    except:
        df['sector_alpha_5'] = 0.0

    try:
        df['momentum_persist'] = df['ret_1'].rolling(3).mean().shift(1) / (df['ret_1'].rolling(3).std().shift(1) + 1e-9)
    except:
        df['momentum_persist'] = 0.0

    # ========== 3. 宏觀 + 外圍 ==========
    for t in cfg.get('macro', {}):
        if t in prices_df.columns:
            try:
                df[f'macro_{t}'] = prices_df[t].pct_change(1)
            except:
                pass
    if '^GSPC' in prices_df.columns:
        df['us_sp'] = prices_df['^GSPC'].pct_change(1)
    if '000300.SS' in prices_df.columns:
        df['a_share'] = prices_df['000300.SS'].pct_change(1)
    if '^VIX' in prices_df.columns:
        df['vix'] = prices_df['^VIX']
        df['vix_chg'] = prices_df['^VIX'].pct_change(1)
    if 'DX-Y.NYB' in prices_df.columns:
        df['dxy_chg'] = prices_df['DX-Y.NYB'].pct_change(1)

    # ========== 4. FRED ==========
    if fred_df is not None and not fred_df.empty:
        fred_aligned = fred_df.reindex(df.index, method='ffill')
        if 'US10Y_FRED' in fred_aligned.columns:
            df['fred_us10y'] = fred_aligned['US10Y_FRED']
            df['fred_us10y_chg'] = fred_aligned['US10Y_FRED'].diff()
        if 'YieldCurve' in fred_aligned.columns:
            df['fred_yieldcurve'] = fred_aligned['YieldCurve']
        if 'FedFunds' in fred_aligned.columns:
            df['fred_fedfunds'] = fred_aligned['FedFunds']
        if 'US_CPI' in fred_aligned.columns:
            df['fred_cpi_chg'] = fred_aligned['US_CPI'].pct_change(30)
        if 'US_Unemployment' in fred_aligned.columns:
            df['fred_unrate'] = fred_aligned['US_Unemployment']
            df['fred_unrate_chg'] = fred_aligned['US_Unemployment'].diff(30)
        if 'HY_Spread' in fred_aligned.columns:
            df['fred_hy'] = fred_aligned['HY_Spread']
            df['fred_hy_chg'] = fred_aligned['HY_Spread'].diff(5)

    # ========== 5. 恒指獨特性 - 只有恒指先有 ==========
    if target == hsi_ticker:
        try:
            top10 = list(cfg.get('top10', {}).keys())
            top10_in_data = [t for t in top10 if t in prices_df.columns]
            if top10_in_data:
                daily_ret = prices_df[top10_in_data].pct_change(1)
                df['breadth_up_pct'] = (daily_ret > 0).mean(axis=1)
                df['breadth_avg_ret'] = daily_ret.mean(axis=1)
                df['breadth_max_ret'] = daily_ret.max(axis=1)
                df['breadth_min_ret'] = daily_ret.min(axis=1)
                df['breadth_dispersion'] = df['breadth_max_ret'] - df['breadth_avg_ret']
            else:
                df['breadth_up_pct'] = 0.5; df['breadth_avg_ret'] = 0; df['breadth_dispersion'] = 0
        except:
            df['breadth_up_pct'] = 0.5; df['breadth_avg_ret'] = 0; df['breadth_dispersion'] = 0

        try:
            if '000300.SS' in prices_df.columns:
                df['hsi_a_corr'] = df['ret_1'].rolling(20).corr(prices_df['000300.SS'].pct_change(1)).shift(1)
                df['hsi_a_spread'] = df['ret_5'] - prices_df['000300.SS'].pct_change(5)
            else:
                df['hsi_a_corr'] = 0.5; df['hsi_a_spread'] = 0
        except:
            df['hsi_a_corr'] = 0.5; df['hsi_a_spread'] = 0

        try:
            top10 = list(cfg.get('top10', {}).keys())
            top10_in_data = [t for t in top10 if t in prices_df.columns]
            if top10_in_data:
                stock_vol = prices_df[top10_in_data].pct_change(1).rolling(20).std().shift(1).mean(axis=1)
                df['hsi_vol_vs_stock_vol'] = df['vol_20'] / (stock_vol + 1e-9)
        except:
            df['hsi_vol_vs_stock_vol'] = 0.8
    else:
        df['breadth_up_pct'] = 0.0
        df['breadth_avg_ret'] = 0.0
        df['breadth_dispersion'] = 0.0
        df['hsi_a_corr'] = 0.0
        df['hsi_a_spread'] = 0.0
        df['hsi_vol_vs_stock_vol'] = 0.0

    df['target_next'] = s.pct_change(1).shift(-1)
    return df.dropna()

def factor_explain(last_row, ticker='^HSI'):
    """
    Factor 解釋, 分恒指版同個股版
    一定要傳 ticker, 唔好只傳 last_row
    """
    f = {}
    try: f['美股 S&P'] = float(last_row.get('us_sp', 0) * 0.35)
    except: f['美股 S&P'] = 0
    try: f['A股'] = float(last_row.get('a_share', 0) * 0.25)
    except: f['A股'] = 0
    try: f['VIX'] = float(-last_row.get('vix_chg', 0) * 0.20)
    except: f['VIX'] = 0
    try: f['美元 DXY'] = float(-last_row.get('dxy_chg', 0) * 0.30)
    except: f['美元 DXY'] = 0
    try: f['技術 RSI'] = float((last_row.get('rsi', 50) - 50) / 1000)
    except: f['技術 RSI'] = 0

    try:
        if 'alpha_vs_hsi_5' in last_row:
            f['跑贏恒指 Alpha'] = float(last_row.get('alpha_vs_hsi_5', 0) * 0.4)
        if 'price_vs_ma60' in last_row:
            pv = float(last_row.get('price_vs_ma60', 1.0))
            f['離60日線'] = float((1.0 - pv) * 0.1)
        if 'vol_ratio' in last_row:
            vr = float(last_row.get('vol_ratio', 1.0))
            f['波動放大'] = float(-(vr - 1.0) * 0.02) if vr > 1.5 else 0.0
        if 'sector_alpha_5' in last_row:
            f['板塊龍頭'] = float(last_row.get('sector_alpha_5', 0) * 0.3)
    except:
        pass

    try:
        if 'fred_us10y_chg' in last_row:
            f['美國10Y利率變化'] = float(-last_row.get('fred_us10y_chg', 0) * 0.5)
        if 'fred_yieldcurve' in last_row:
            yc = float(last_row.get('fred_yieldcurve', 0))
            f['收益率曲線'] = -0.02 if yc < 0 else 0.01
        if 'fred_hy_chg' in last_row:
            f['信用利差'] = float(-last_row.get('fred_hy_chg', 0) * 0.2)
    except:
        pass

    if ticker == '^HSI':
        try:
            if 'breadth_up_pct' in last_row:
                b = float(last_row.get('breadth_up_pct', 0.5))
                f['成份股廣度'] = float((b - 0.5) * 0.1)
            if 'breadth_dispersion' in last_row:
                disp = float(last_row.get('breadth_dispersion', 0))
                f['龍頭拉動風險'] = float(-disp * 0.2) if disp > 0.03 else 0.0
            if 'hsi_a_spread' in last_row:
                f['恒指 vs A股'] = float(last_row.get('hsi_a_spread', 0) * 0.2)
        except:
            pass

    return f
