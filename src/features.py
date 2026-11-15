# src/features.py
import pandas as pd, numpy as np
def make_features(prices_df, target, cfg):
    df = pd.DataFrame(); s = prices_df[target]; df['close'] = s
    try: df['ret_1'] = s.pct_change(1); df['ret_5'] = s.pct_change(5); df['ret_20'] = s.pct_change(20)
    except: df['ret_1']=0; df['ret_5']=0; df['ret_20']=0
    try:
        d = s.diff(); g = d.where(d>0,0).rolling(14).mean(); l = -d.where(d<0,0).rolling(14).mean()
        df['rsi'] = 100 - (100/(1+g/(l+1e-9)))
    except: df['rsi']=50
    try:
        e12=s.ewm(span=12).mean(); e26=s.ewm(span=26).mean(); df['macd']=e12-e26; df['macd_signal']=df['macd'].ewm(span=9).mean()
    except: df['macd']=0; df['macd_signal']=0
    try: df['ma5']=s.rolling(5).mean(); df['ma20']=s.rolling(20).mean(); df['ma60']=s.rolling(60).mean(); df['ma_ratio']=df['ma5']/df['ma20']
    except: df['ma5']=s; df['ma20']=s; df['ma60']=s; df['ma_ratio']=1
    try: df['vol_20']=df['ret_1'].rolling(20).std()
    except: df['vol_20']=0.01
    for t in cfg['macro']:
        if t in prices_df.columns:
            try: df[f'macro_{t}']=prices_df[t].pct_change(1)
            except: pass
    if '^GSPC' in prices_df.columns: df['us_sp']=prices_df['^GSPC'].pct_change(1)
    if '000300.SS' in prices_df.columns: df['a_share']=prices_df['000300.SS'].pct_change(1)
    if '^VIX' in prices_df.columns: df['vix']=prices_df['^VIX']; df['vix_chg']=prices_df['^VIX'].pct_change(1)
    if 'DX-Y.NYB' in prices_df.columns: df['dxy_chg']=prices_df['DX-Y.NYB'].pct_change(1)
    df['target_next']=s.pct_change(1).shift(-1)
    return df.dropna()
def factor_explain(last_row):
    f={}
    try: f['美股 S&P']=float(last_row.get('us_sp',0)*0.35)
    except: f['美股 S&P']=0
    try: f['A股']=float(last_row.get('a_share',0)*0.25)
    except: f['A股']=0
    try: f['VIX']=float(-last_row.get('vix_chg',0)*0.20)
    except: f['VIX']=0
    try: f['美元 DXY']=float(-last_row.get('dxy_chg',0)*0.30)
    except: f['美元 DXY']=0
    try: f['技術 RSI']=float((last_row.get('rsi',50)-50)/1000)
    except: f['技術 RSI']=0
    return f
