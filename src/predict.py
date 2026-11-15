# src/predict.py - 主程式 - 完整版 V2
# 流程: 讀 config -> 拎 Yahoo 數據 -> 拎 FRED 數據 -> Train 6 個模型 -> GARCH + Bayesian + Monte Carlo
#      -> Factor 解釋 -> RL 選權重 -> Backtest -> Risk 風控 -> 畫圖 -> Send Email (多人 + FRED)
# src/predict.py - 主程式 - 每隻股票自己 Seed + 個股特性 + 恒指獨特性版
import sys, os, logging, pandas as pd, numpy as np, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("HSI-V2")

from src.config import load_config, get_all_tickers, get_name_map
from src.fetch_data import fetch_all
from src.features import make_features, factor_explain, get_ticker_seed
from src.models import train_tree_models, train_deep_models
from src.volatility import garch_vol, bayesian_prob, monte_carlo
from src.rl_agent import RLAgent
from src.backtest import walk_forward_backtest
from src.risk import RiskManager
from src.plot import plot_all

try:
    from src.fred_data import fetch_all_fred, fred_signals
    HAS_FRED = True
except:
    HAS_FRED = False

def set_seed_for_ticker(ticker):
    """
    每隻股票自己嘅 Seed, 鎖死所有隨機
    0700.HK 每次 Run 都用同一個 Seed, 結果穩定
    1810.HK 用另一個 Seed, 唔會同 0700 撞
    ^HSI 用自己嘅特別 Seed (+9999), 保留恒指獨特性
    """
    seed = get_ticker_seed(ticker, base_seed=42)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass
    return seed

def train_one(ticker, prices, cfg, name_map, fred_df=None, use_deep=True):
    # 1. 每隻股票入 Train 之前, 先鎖自己嘅 Seed
    seed = set_seed_for_ticker(ticker)
    log.info(f"  Seed for {ticker} = {seed} (每隻獨立, 恒指特別 Seed)")

    try:
        df = make_features(prices, ticker, cfg, fred_df=fred_df)
        feature_cols = [c for c in df.columns if c not in ['close', 'target_next']]

        train_df = df.iloc[:-20]
        test_df = df.iloc[-20:]
        X_train = train_df[feature_cols].values
        y_train = train_df['target_next'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target_next'].values

        # Tree - 每隻獨立 Train, 用自己特徵
        preds, maes = train_tree_models(X_train, y_train, X_test, y_test)

        # Deep - 恒指先用全部, 個股用 Tree 就夠 (快 + 唔 Overfit)
        # 如果你想個股都用 Deep, 將下面個 use_deep 喺 main() 改做 True 就得
        if use_deep:
            # Train Deep 之前再鎖一次 Seed, 因為 Tree 已經用咗啲隨機
            set_seed_for_ticker(ticker)
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

        # RL - 每隻自己學自己嘅權重
        # 0700 學出嚟嘅權重同 1810 唔同, 保留獨特性
        # RL 入面都有隨機, 要鎖埋
        set_seed_for_ticker(ticker)
        rl = RLAgent(list(preds.keys()))
        rl_w = rl.learn(maes)

        weighted = float(sum(preds[k] * rl_w[k] for k in preds))

        # GARCH + Bayesian + Monte Carlo - 都用返呢隻嘅 Seed
        set_seed_for_ticker(ticker)
        vol = garch_vol(df['ret_1'])
        prob_up = bayesian_prob(weighted, vol)
        mc_prob, mc_p10, mc_p90 = monte_carlo(weighted, vol)

        # Factor - 傳埋 ticker, 等佢知係恒指定個股, 俾唔同 Factor
        factor = factor_explain(df.iloc[-1], ticker=ticker)

        std = float(np.std(list(preds.values())))
        conf = float(max(0.35, min(0.88, 0.78 - std * 10 + abs(mc_prob - 0.5) * 0.3)))

        last_price = float(df['close'].iloc[-1])

        return {
            'ticker': ticker,
            'name': name_map.get(ticker, ticker),
            'seed': seed,
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
            'feature_names': feature_cols,
        }
    except Exception as e:
        log.warning(f"{ticker} 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    cfg = load_config()
    name_map = get_name_map(cfg)
    tickers = get_all_tickers(cfg)
    log.info(f"啟動 V2 每隻獨立 Seed 版 | {len(tickers)} 隻 | FRED={'有' if HAS_FRED else '無'}")
    log.info(f"Seed 規則: 每隻股票 = sum(ord(Ticker)) + 42, 恒指額外 +9999")

    prices = fetch_all(tickers)
    log.info(f"Yahoo: {prices.shape[0]} 日 x {prices.shape[1]} 隻, 最新 {prices.index[-1].date()}")
    os.makedirs("output", exist_ok=True)

    # FRED
    fred_df = None
    fred_sig = {}
    fred_text = ""
    if HAS_FRED:
        try:
            fred_df = fetch_all_fred(cfg)
            fred_sig = fred_signals(fred_df) if fred_df is not None and not fred_df.empty else {}
            if fred_sig:
                log.info(f"📊 FRED: {fred_sig}")
                fred_text = f"FRED: 美國10Y {fred_sig.get('US10Y_FRED','-')}%, Fed {fred_sig.get('FedFunds','-')}%, YieldCurve {fred_sig.get('YieldCurve','-')} ({fred_sig.get('YieldCurve_Signal','')}), CPI YoY {fred_sig.get('US_CPI_YoY','-')}, 失業率 {fred_sig.get('US_Unemployment','-')}%, HY {fred_sig.get('HY_Spread','-')} ({fred_sig.get('Credit_Stress','')})"
                print(f"📊 {fred_text}")
            else:
                fred_df = None
        except Exception as e:
            log.warning(f"FRED 失敗: {e}")
            fred_df = None; fred_sig = {}

    targets = [cfg['hsi']] + list(cfg['top10'].keys()) + list(cfg.get('my_picks', {}).keys())
    results = []
    for t in targets:
        log.info(f"訓練 {name_map.get(t, t)} {t}...")
        # 恒指用 Deep, 個股用 Tree (快 + 唔 Overfit)
        # 如果你想個股都用 Deep, 改做 use_deep=True 全部
        use_deep = (t == cfg['hsi'])
        r = train_one(t, prices, cfg, name_map, fred_df=fred_df, use_deep=use_deep)
        if r:
            results.append(r)
            rl_best = max(r['rl_weights'], key=r['rl_weights'].get) if r['rl_weights'] else "-"
            # 印埋獨特性特徵俾你睇
            if t == cfg['hsi']:
                print(f"  恒指獨特性: Seed {r['seed']}, 特徵 {r['feature_count']} 個, 包括 breadth_up_pct, hsi_a_corr, hsi_vol_vs_stock_vol")
            else:
                print(f"  個股獨特性: Seed {r['seed']}, 特徵 {r['feature_count']} 個, 包括 alpha_vs_hsi_5, price_vs_ma60, vol_ratio, sector_alpha_5")
            log.info(f"✓ {r['name']}: {r['pred_1d_pct']:+.2f}% 信心 {r['confidence']*100:.0f}% RL最信 {rl_best}")

    if not results:
        log.error("全部失敗")
        return

    rows = []
    for r in results:
        rows.append({
            '股票': f"{r['name']} {r['ticker']}",
            'Seed': r['seed'],
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
    print("\n=== 預測結果 (每隻獨立 Seed + 獨特性特徵) ===")
    print(df_res.to_string(index=False))
    df_res.to_csv('output/stock_ranking.csv', index=False, encoding='utf-8-sig')

    try:
        plot_all(prices, results, cfg)
    except Exception as e:
        log.warning(f"畫圖失敗: {e}")

    backtest_summary = ""
    try:
        bt_df = walk_forward_backtest(prices, cfg['hsi'], initial=cfg.get('initial_capital', 100000), fred_df=fred_df)
        if bt_df is not None and len(bt_df) > 0:
            last_cap = bt_df['capital'].iloc[-1] if 'capital' in bt_df.columns else cfg.get('initial_capital', 100000)
            init_cap = cfg.get('initial_capital', 100000)
            total_ret = (last_cap / init_cap - 1) * 100
            backtest_summary = f"Walk-Forward Backtest: 初始 ${init_cap:,.0f} -> 最終 ${last_cap:,.0f}, 總回報 {total_ret:+.2f}%, 交易 {len(bt_df)} 個月"
            print(f"📊 {backtest_summary}")
    except Exception as e:
        log.warning(f"Backtest 失敗: {e}")
        backtest_summary = "Backtest 失敗"

    try:
        hsi_res = next((r for r in results if r['ticker'] == cfg['hsi']), None)
        if hsi_res:
            rm = RiskManager(capital=cfg.get('initial_capital', 100000))
            rm.open_trade(price=hsi_res['last_price'], confidence=hsi_res['confidence'], vol_pct=hsi_res['vol'])
            rm.summary()
    except Exception as e:
        log.warning(f"Risk 失敗: {e}")

    try:
        from src.email_sender import send_email, calc_levels
        for r in results:
            lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
            r.update(lv)
            print(f"{r['name']} (Seed {r['seed']}): 高 {lv['high']} 低 {lv['low']} 入場 {lv['entry']} 離場 {lv['exit']} 止蝕 {lv['stop']} -> {lv['signal']}")
        send_email(results, backtest_summary=backtest_summary + (" | " + fred_text if fred_text else ""), fred_sig=fred_sig if fred_sig else None, fred_text=fred_text)
    except Exception as e:
        print(f"Email 失敗: {e}")

    try:
        hsi_res = next((r for r in results if r['ticker'] == cfg['hsi']), None)
        if hsi_res:
            print(f"\n恒指 Factor (恒指獨特性): {hsi_res['factor']}")
            print(f"恒指 各模型: { {k: round(v*100,2) for k,v in hsi_res['models'].items()} }")
            print(f"恒指 RL: { {k: round(v*100,1) for k,v in hsi_res['rl_weights'].items()} }")
            print(f"恒指 Seed: {hsi_res['seed']}")
            # 印一個個股例子俾你對比
            stock_ex = next((r for r in results if r['ticker'] != cfg['hsi']), None)
            if stock_ex:
                print(f"\n個股例子 {stock_ex['name']} Factor (個股獨特性): {stock_ex['factor']}")
                print(f"個股 Seed: {stock_ex['seed']} (同恒指唔同)")
            if fred_sig:
                print(f"FRED: {fred_sig}")
            print(f"\nLLM 反證: 你係沽空基金經理, 反駁恒指聽日 {hsi_res['pred_1d_pct']:+.2f}%, 信心 {hsi_res['confidence']*100:.0f}%, Factor {hsi_res['factor']}, FRED {fred_sig}, 指出3個風險")
    except Exception as e:
        log.warning(f"Print 失敗: {e}")

    print("\n✅ 完成! 每隻獨立 Seed + 獨特性特徵, 同一日 Run 兩次結果一樣, 但每隻股票結果唔同")
    print("📧 Email 已嘗試發送")

if __name__ == "__main__":
    main()
