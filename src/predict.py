# src/predict.py - 主程式 - 完整版 V2
# 流程: 讀 config -> 拎 Yahoo 數據 -> 拎 FRED 數據 -> Train 6 個模型 -> GARCH + Bayesian + Monte Carlo
#      -> Factor 解釋 -> RL 選權重 -> Backtest -> Risk 風控 -> 畫圖 -> Send Email (多人 + FRED)
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

# FRED - 如果無裝或者無 Key, 會自動 Fallback, 唔會爆
try:
    from src.fred_data import fetch_all_fred, fred_signals
    HAS_FRED = True
except Exception as e:
    HAS_FRED = False
    log.warning(f"無 FRED 模組: {e}, 會跳過 FRED")

def train_one(ticker, prices, cfg, name_map, fred_df=None):
    """
    訓練一隻股票 / 指數
    ticker: 例如 ^HSI, 0700.HK
    prices: DataFrame, Yahoo 拎返嚟嘅所有價
    fred_df: DataFrame, FRED 拎返嚟嘅宏觀, 可以係 None
    """
    try:
        # 1. 特徵 - 已經包埋 FRED 特徵
        df = make_features(prices, ticker, cfg, fred_df=fred_df)
        feature_cols = [c for c in df.columns if c not in ['close', 'target_next']]

        # 2. 切 Train / Test - 最後 20 日做 Test, 唔偷睇未來
        train_df = df.iloc[:-20]
        test_df = df.iloc[-20:]
        X_train = train_df[feature_cols].values
        y_train = train_df['target_next'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target_next'].values

        # 3. Tree 模型: XGBoost + LightGBM
        preds, maes = train_tree_models(X_train, y_train, X_test, y_test)

        # 4. Deep 模型: LSTM + GRU + TCN + Transformer
        deep_preds, deep_maes = train_deep_models(
            df[feature_cols].values,
            df['target_next'].values,
            cfg['lookback'],
            cfg['epochs']
        )
        preds.update(deep_preds)
        maes.update(deep_maes)

        if not preds:
            return None

        # 5. RL - 學邊個模型最準, 自動俾權重
        rl = RLAgent(list(preds.keys()))
        rl_w = rl.learn(maes)

        # 6. 加權預測
        weighted = float(sum(preds[k] * rl_w[k] for k in preds))

        # 7. 波動 + 概率
        vol = garch_vol(df['ret_1'])
        prob_up = bayesian_prob(weighted, vol)
        mc_prob, mc_p10, mc_p90 = monte_carlo(weighted, vol)

        # 8. Factor 解釋
        factor = factor_explain(df.iloc[-1])

        # 9. 信心: 模型越一致, 信心越高
        std = float(np.std(list(preds.values())))
        conf = float(max(0.35, min(0.88, 0.78 - std * 10 + abs(mc_prob - 0.5) * 0.3)))

        last_price = float(df['close'].iloc[-1])

        return {
            'ticker': ticker,
            'name': name_map.get(ticker, ticker),
            'last_price': last_price,
            'pred_1d_pct': weighted * 100,
            'pred_5d_pct': weighted * 5 * 100,
            'pred_5d_price': last_price * (1 + weighted * 5),
            'confidence': conf,
            'vol': vol * 100,
            'prob_up': prob_up * 100,
            'mc_prob_up': mc_prob * 100,
            'mc_range': (mc_p10 * 100, mc_p90 * 100),
            'models': preds,
            'rl_weights': rl_w,
            'factor': factor,
            'last_date': str(df.index[-1].date()),
            'maes': maes,
            'feature_count': len(feature_cols),
        }
    except Exception as e:
        log.warning(f"{ticker} 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # ============ 1. 讀 Config ============
    cfg = load_config()
    name_map = get_name_map(cfg)
    tickers = get_all_tickers(cfg)
    log.info(f"啟動 V2 實戰版 | {len(tickers)} 隻 | FRED={'有' if HAS_FRED else '無'}")

    # ============ 2. 拎 Yahoo 數據 ============
    prices = fetch_all(tickers)
    log.info(f"Yahoo 數據: {prices.shape[0]} 日 x {prices.shape[1]} 隻, 最新 {prices.index[-1].date()}")

    os.makedirs("output", exist_ok=True)

    # ============ 3. 拎 FRED 數據 ============
    fred_df = None
    fred_sig = {}
    fred_text = ""
    if HAS_FRED:
        try:
            fred_df = fetch_all_fred(cfg)
            fred_sig = fred_signals(fred_df) if fred_df is not None and not fred_df.empty else {}
            if fred_sig:
                log.info(f"📊 FRED 宏觀: {fred_sig}")
                # 整一個文字版, 俾 Email 用
                fred_text = (
                    f"FRED: 美國10Y {fred_sig.get('US10Y_FRED','-')}%, "
                    f"Fed Funds {fred_sig.get('FedFunds','-')}%, "
                    f"YieldCurve {fred_sig.get('YieldCurve','-')} ({fred_sig.get('YieldCurve_Signal','')}), "
                    f"CPI YoY {fred_sig.get('US_CPI_YoY',0):.2f}%  " if isinstance(fred_sig.get('US_CPI_YoY',0), (int,float)) else f"CPI YoY -, "
                    f"失業率 {fred_sig.get('US_Unemployment','-')}%, "
                    f"HY Spread {fred_sig.get('HY_Spread','-')} ({fred_sig.get('Credit_Stress','')})"
                )
                print(f"📊 {fred_text}")
            else:
                print("FRED 無數據 (可能無 FRED_API_KEY 或網絡問題), 只用 Yahoo")
                fred_df = None
        except Exception as e:
            log.warning(f"FRED 失敗, 繼續用 Yahoo: {e}")
            fred_df = None
            fred_sig = {}
    else:
        print("無 FRED 模組, 跳過 FRED")

    # ============ 4. 訓練所有目標 ============
    targets = [cfg['hsi']] + list(cfg['top10'].keys()) + list(cfg.get('my_picks', {}).keys())
    log.info(f"訓練目標: {targets}")

    results = []
    for t in targets:
        log.info(f"訓練 {name_map.get(t, t)} {t}...")
        r = train_one(t, prices, cfg, name_map, fred_df=fred_df)
        if r:
            results.append(r)
            rl_best = max(r['rl_weights'], key=r['rl_weights'].get) if r['rl_weights'] else "-"
            log.info(f"✓ {r['name']}: {r['pred_1d_pct']:+.2f}% 信心 {r['confidence']*100:.0f}% 波動 {r['vol']:.2f}% RL最信 {rl_best} 特徵 {r['feature_count']} 個")
        else:
            log.warning(f"✗ {t} 訓練失敗, 跳過")

    if not results:
        log.error("全部失敗, 無結果")
        return

    # ============ 5. 表格 + CSV ============
    rows = []
    for r in results:
        rows.append({
            '股票': f"{r['name']} {r['ticker']}",
            '現價': round(r['last_price'], 2),
            '預測聽日%': round(r['pred_1d_pct'], 2),
            '預測5日%': round(r['pred_5d_pct'], 2),
            '預測5日價': round(r['pred_5d_price'], 2),
            '信心%': round(r['confidence'] * 100, 1),
            '波動%': round(r['vol'], 2),
            'Bayesian升%': round(r['prob_up'], 1),
            'MC升%': round(r['mc_prob_up'], 1),
            'RL最信': max(r['rl_weights'], key=r['rl_weights'].get) if r['rl_weights'] else "-",
        })
    df_res = pd.DataFrame(rows)
    print("\n=== 預測結果 ===")
    print(df_res.to_string(index=False))
    df_res.to_csv('output/stock_ranking.csv', index=False, encoding='utf-8-sig')
    log.info("已儲存 output/stock_ranking.csv")

    # ============ 6. 畫圖 ============
    try:
        plot_all(prices, results, cfg)
        log.info("已畫圖 output/hsi_forecast.png, output/stock_ranking.png")
    except Exception as e:
        log.warning(f"畫圖失敗: {e}")

    # ============ 7. Walk-Forward Backtest ============
    backtest_summary = ""
    try:
        bt_df = walk_forward_backtest(prices, cfg['hsi'], initial=cfg.get('initial_capital', 100000))
        if bt_df is not None and len(bt_df) > 0:
            # 計總回報, 俾 Email 用
            last_cap = bt_df['capital'].iloc[-1] if 'capital' in bt_df.columns else cfg.get('initial_capital', 100000)
            init_cap = cfg.get('initial_capital', 100000)
            total_ret = (last_cap / init_cap - 1) * 100
            backtest_summary = f"Walk-Forward Backtest: 初始 ${init_cap:,.0f} -> 最終 ${last_cap:,.0f}, 總回報 {total_ret:+.2f}%, 交易 {len(bt_df)} 個月, 詳情見 output/backtest_result.csv"
            print(f"📊 {backtest_summary}")
    except Exception as e:
        log.warning(f"Backtest 失敗: {e}")
        backtest_summary = "Backtest 失敗, 見 Log"

    # ============ 8. Risk 風控 ============
    try:
        hsi_res = None
        for r in results:
            if r['ticker'] == cfg['hsi']:
                hsi_res = r
                break
        if hsi_res:
            rm = RiskManager(capital=cfg.get('initial_capital', 100000))
            rm.open_trade(price=hsi_res['last_price'], confidence=hsi_res['confidence'], vol_pct=hsi_res['vol'])
            rm.summary()
    except Exception as e:
        log.warning(f"Risk 失敗: {e}")

    # ============ 9. 計算高低位 + Email ============
    try:
        from src.email_sender import send_email, calc_levels
        for r in results:
            lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
            r.update(lv)
            print(f"{r['name']}: 高 {lv['high']} 低 {lv['low']} 入場 {lv['entry']} 離場 {lv['exit']} 止蝕 {lv['stop']} -> {lv['signal']}")

        # Send Email - 傳埋 FRED + Backtest
        # EMAIL_TO 可以係多人: "a@gmail.com, b@gmail.com"
        send_email(
            results,
            backtest_summary=backtest_summary + (" | " + fred_text if fred_text else ""),
            fred_sig=fred_sig if fred_sig else None,
            fred_text=fred_text
        )
    except Exception as e:
        print(f"Email 失敗: {e}")
        import traceback
        traceback.print_exc()

    # ============ 10. Factor + LLM Prompt Print ============
    try:
        hsi_res = None
        for r in results:
            if r['ticker'] == cfg['hsi']:
                hsi_res = r
                break
        if hsi_res:
            print(f"\n恒指 Factor: {hsi_res['factor']}")
            print(f"恒指 各模型: { {k: round(v*100, 2) for k, v in hsi_res['models'].items()} }")
            print(f"恒指 RL 權重: { {k: round(v*100, 1) for k, v in hsi_res['rl_weights'].items()} }")
            if fred_sig:
                print(f"FRED: {fred_sig}")
            print("\n--- LLM 反證 Prompt (Copy 去 ChatGPT / Meta AI) ---")
            print(f"你係沽空基金經理, 反駁恒指聽日 {hsi_res['pred_1d_pct']:+.2f}%, 信心 {hsi_res['confidence']*100:.0f}%, 波動 {hsi_res['vol']:.2f}%, 現價 {hsi_res['last_price']:.2f}, Factor {hsi_res['factor']}, 各模型 {hsi_res['models']}, FRED {fred_sig}, 指出3個風險, 今日買/唔買/反手沽, 明確決定。")
    except Exception as e:
        log.warning(f"Print Factor 失敗: {e}")

    print("\n✅ 完成! output/ 有 hsi_forecast.png, stock_ranking.png, stock_ranking.csv, backtest_result.csv")
    print("📧 Email 已嘗試發送 (如果設定咗 EMAIL_USER / EMAIL_PASS / EMAIL_TO / FRED_API_KEY)")

if __name__ == "__main__":
    main()
