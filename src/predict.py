# src/predict.py
import sys, os, logging, pandas as pd, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("HSI-V2")
from src.config import load_config, get_all_tickers, get_name_map
from src.fetch_data import fetch_all
from src.features import make_features, factor_explain
from src.models import train_tree_models, train_deep_models
from src.volatility import garch_vol, bayesian_prob, monte_carlo
from src.rl_agent import RLAgent
from src.backtest import walk_forward_backtest
from src.risk import RiskManager
from src.plot import plot_all
from src.fred_data import fetch_all_fred, fred_signals

def train_one(ticker, prices, cfg, name_map, fred_df=None):
    try:
        df = make_features(prices, ticker, cfg, fred_df=fred_df)
        feature_cols=[c for c in df.columns if c not in ['close','target_next']]
        train_df=df.iloc[:-20]; test_df=df.iloc[-20:]
        X_train=train_df[feature_cols].values; y_train=train_df['target_next'].values
        X_test=test_df[feature_cols].values; y_test=test_df['target_next'].values
        preds, maes = train_tree_models(X_train, y_train, X_test, y_test)
        deep_preds, deep_maes = train_deep_models(df[feature_cols].values, df['target_next'].values, cfg['lookback'], cfg['epochs'])
        preds.update(deep_preds); maes.update(deep_maes)
        if not preds: return None
        rl = RLAgent(list(preds.keys())); rl_w = rl.learn(maes)
        weighted = float(sum(preds[k]*rl_w[k] for k in preds))
        vol = garch_vol(df['ret_1']); prob_up = bayesian_prob(weighted, vol)
        mc_prob, mc_p10, mc_p90 = monte_carlo(weighted, vol)
        factor = factor_explain(df.iloc[-1])
        std=float(np.std(list(preds.values()))); conf=float(max(0.35, min(0.88, 0.78-std*10+abs(mc_prob-0.5)*0.3)))
        last_price=float(df['close'].iloc[-1])
        return {'ticker':ticker,'name':name_map.get(ticker,ticker),'last_price':last_price,'pred_1d_pct':weighted*100,'pred_5d_pct':weighted*5*100,'pred_5d_price':last_price*(1+weighted*5),'confidence':conf,'vol':vol*100,'prob_up':prob_up*100,'mc_prob_up':mc_prob*100,'mc_range':(mc_p10*100,mc_p90*100),'models':preds,'rl_weights':rl_w,'factor':factor,'last_date':str(df.index[-1].date()),'maes':maes}
    except Exception as e:
        log.warning(f"{ticker} 失敗: {e}"); return None
def main():
    cfg = load_config(); name_map = get_name_map(cfg); tickers = get_all_tickers(cfg)
    log.info(f"啟動 V2 實戰版 | {len(tickers)} 隻")
    prices = fetch_all(tickers)
    # FRED
    fred_df = fetch_all_fred(cfg)
    fred_sig = fred_signals(fred_df)
    if fred_sig:
        print(f"📊 FRED 宏觀: {fred_sig}")
    else:
        print("FRED 無數據, 只用 Yahoo")
        
    os.makedirs("output", exist_ok=True)
    targets = [cfg['hsi']] + list(cfg['top10'].keys()) + list(cfg.get('my_picks',{}).keys())
    results=[]
    for t in targets:
        log.info(f"訓練 {name_map.get(t,t)} {t}...")
        r=train_one(t, prices, cfg, name_map, fred_df=fred_df)
        if r: results.append(r); log.info(f"✓ {r['name']}: {r['pred_1d_pct']:+.2f}% 信心 {r['confidence']*100:.0f}%")
    if not results: log.error("全部失敗"); return
    rows=[{'股票':f"{r['name']} {r['ticker']}",'現價':round(r['last_price'],2),'預測聽日%':round(r['pred_1d_pct'],2),'預測5日%':round(r['pred_5d_pct'],2),'預測5日價':round(r['pred_5d_price'],2),'信心%':round(r['confidence']*100,1),'波動%':round(r['vol'],2),'Bayesian升%':round(r['prob_up'],1),'MC升%':round(r['mc_prob_up'],1),'RL最信':max(r['rl_weights'], key=r['rl_weights'].get)} for r in results]
    df_res=pd.DataFrame(rows); print(df_res.to_string(index=False)); df_res.to_csv('output/stock_ranking.csv', index=False, encoding='utf-8-sig')
    try: plot_all(prices, results, cfg)
    except Exception as e: log.warning(f"畫圖失敗: {e}")
    try: walk_forward_backtest(prices, cfg['hsi'], initial=cfg.get('initial_capital',100000))
    except Exception as e: log.warning(f"Backtest 失敗: {e}")
    try:
        hsi_res=[r for r in results if r['ticker']==cfg['hsi']][0]
        rm=RiskManager(capital=cfg.get('initial_capital',100000))
        rm.open_trade(price=hsi_res['last_price'], confidence=hsi_res['confidence'], vol_pct=hsi_res['vol']); rm.summary()
    except Exception as e: log.warning(f"Risk 失敗: {e}")
    try:
        from src.email_sender import send_email, calc_levels
        for r in results:
            lv=calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence']); r.update(lv)
            print(f"{r['name']}: 高 {lv['high']} 低 {lv['low']} 入場 {lv['entry']} 離場 {lv['exit']} 止蝕 {lv['stop']} -> {lv['signal']}")
        # 在 main() 入面, 你已經有 fred_sig 同 fred_text 嗰度
        fred_text = ""
        if 'fred_sig' in locals() and fred_sig:
            fred_text = f"FRED: 美國10Y {fred_sig.get('US10Y_FRED','-')}%, Fed {fred_sig.get('FedFunds','-')}%, YieldCurve {fred_sig.get('YieldCurve','-')} ({fred_sig.get('YieldCurve_Signal','')}), CPI YoY {fred_sig.get('US_CPI_YoY',0):.2f}%, 失業率 {fred_sig.get('US_Unemployment','-')}%, HY {fred_sig.get('HY_Spread','-')} ({fred_sig.get('Credit_Stress','')})"
        
        send_email(results, backtest_summary="Backtest 見 output/backtest_result.csv", fred_sig=fred_sig if 'fred_sig' in locals() else None, fred_text=fred_text)
    except Exception as e: print(f"Email 失敗: {e}")
    hsi_res=[r for r in results if r['ticker']==cfg['hsi']][0]
    print(f"恒指 Factor: {hsi_res['factor']}")
    print(f"LLM 反證 Prompt: 你係沽空基金經理, 反駁恒指聽日 {hsi_res['pred_1d_pct']:+.2f}%, 信心 {hsi_res['confidence']*100:.0f}%, Factor {hsi_res['factor']}, 指出3個風險。")
    print("完成! output/ 有結果")
if __name__=="__main__": main()
