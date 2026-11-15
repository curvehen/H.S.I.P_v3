# src/backtest.py - Walk-Forward Backtest V2 - 改進版
# 改進: 1. FRED 過濾 (高息/高通脹唔入市) 2. 倉位管理 3. 止蝕 2% / 止盈 4% 4. 用 Breadth 過濾
import numpy as np, pandas as pd

# 港股真實成本: 佣金 0.03% + 印花稅 0.1% + 交易費 0.005% + 來回 x2
COST_ONE_WAY = 0.0003 + 0.001 + 0.0005  # 0.18% 單邊
COST_ROUND_TRIP = COST_ONE_WAY * 2  # 0.36% 來回

def walk_forward_backtest(prices, target='^HSI', initial=100000, train_months=12, fred_df=None):
    """
    改進版 Backtest
    prices: Yahoo 價
    fred_df: FRED 數據, 如果有就用嚟過濾, 冇都跑到
    """
    print(f"=== Walk-Forward Backtest V2 {target} | 成本來回 {COST_ROUND_TRIP*100:.2f}% | FRED過濾={'有' if fred_df is not None and not fred_df.empty else '無'} ===")
    s = prices[target].dropna()
    monthly = s.resample('ME').last()
    dates = monthly.index

    capital = initial
    trades = []
    win = loss = 0
    curve = []
    skipped_fred = 0
    skipped_breadth = 0

    for i in range(train_months, len(dates)-1):
        train_end = dates[i]
        test_end = dates[i+1]
        train_data = s[s.index <= train_end]
        test_data = s[(s.index > train_end) & (s.index <= test_end)]
        if len(test_data) < 5 or len(train_data) < 100:
            continue

        # --- 1. 技術指標 ---
        last_20 = train_data.tail(20)
        mom = (last_20.iloc[-1] / last_20.iloc[0] - 1) * 100
        delta = train_data.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        lv = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain / (lv + 1e-9)))

        # --- 2. 成份股廣度 (新) ---
        # Top10 有幾多隻過去 20 日升? 少過 40% = 虛火, 唔買
        breadth_ok = True
        try:
            top10_cols = [c for c in prices.columns if c.endswith('.HK')]
            if top10_cols:
                top10_ret_20 = (prices[top10_cols].loc[:train_end].iloc[-1] / prices[top10_cols].loc[:train_end].iloc[-21] - 1)
                breadth = (top10_ret_20 > 0).mean()
                if breadth < 0.4:
                    breadth_ok = False
            else:
                breadth = 0.5
        except:
            breadth = 0.5
            breadth_ok = True

        # --- 3. FRED 過濾 (新, 最重要) ---
        fred_ok = True
        fred_reason = ""
        try:
            if fred_df is not None and not fred_df.empty:
                # 拎 train_end 嗰日最接近嘅 FRED 值
                fred_at_date = fred_df[fred_df.index <= train_end]
                if len(fred_at_date) > 0:
                    last_fred = fred_at_date.iloc[-1]
                    # 規則 1: 美國10Y > 4.8% = 高息, 唔買
                    if 'US10Y_FRED' in last_fred and pd.notna(last_fred['US10Y_FRED']):
                        if last_fred['US10Y_FRED'] > 4.8:
                            fred_ok = False
                            fred_reason = f"US10Y {last_fred['US10Y_FRED']:.2f}% 太高"
                    # 規則 2: CPI YoY > 3.5% = 通脹高, 唔買
                    if 'US_CPI' in fred_df.columns:
                        cpi_series = fred_df['US_CPI'].dropna()
                        cpi_at = cpi_series[cpi_series.index <= train_end]
                        if len(cpi_at) > 365:
                            cpi_yoy = (cpi_at.iloc[-1] / cpi_at.iloc[-365] - 1) * 100
                            if cpi_yoy > 3.5:
                                fred_ok = False
                                fred_reason += f" CPI {cpi_yoy:.2f}% 太高"
                    # 規則 3: HY Spread > 4.5% = 信貸緊張, 唔買
                    if 'HY_Spread' in last_fred and pd.notna(last_fred['HY_Spread']):
                        if last_fred['HY_Spread'] > 4.5:
                            fred_ok = False
                            fred_reason += f" HY {last_fred['HY_Spread']:.2f}% 緊張"
                    # 規則 4: YieldCurve < -0.2 = 衰退警告, 唔買
                    if 'YieldCurve' in last_fred and pd.notna(last_fred['YieldCurve']):
                        if last_fred['YieldCurve'] < -0.2:
                            fred_ok = False
                            fred_reason += f" YieldCurve {last_fred['YieldCurve']:.2f} 衰退"
        except Exception as e:
            fred_ok = True
            fred_reason = f"FRED 讀取失敗: {e}"

        # --- 4. 綜合信號 (新) ---
        # 舊版: mom>0 and rsi<70 就買
        # 新版: 要同時滿足 4 個條件先買, 唔係寧願唔買
        tech_ok = (mom > 0 and rsi < 70 and rsi > 30)  # RSI 太低都唔買, 可能係跌緊
        signal = 1 if (tech_ok and breadth_ok and fred_ok) else 0

        month_ret = (test_data.iloc[-1] / test_data.iloc[0] - 1)

        # --- 5. 倉位管理 (新) ---
        # 舊版: 全倉
        # 新版: 根據 mom 強弱決定倉位, 最多 30%
        if mom > 3:
            position = 0.30
        elif mom > 1:
            position = 0.20
        elif mom > 0:
            position = 0.10
        else:
            position = 0.0

        # --- 6. 執行 ---
        if signal == 1 and position > 0:
            # 模擬月內止蝕 / 止盈: 如果月內最低跌穿 -2%, 就止蝕走, 唔等到月尾
            intra_low = test_data.min() / test_data.iloc[0] - 1
            intra_high = test_data.max() / test_data.iloc[0] - 1
            if intra_low <= -0.02:
                # 止蝕 -2%
                net = -0.02 * position - COST_ROUND_TRIP * position
                reason = "STOP -2%"
            elif intra_high >= 0.04:
                # 止盈 +4%, 走一半, 唔貪
                net = (0.04 * position * 0.5 + month_ret * position * 0.5) - COST_ROUND_TRIP * position
                reason = "TAKE +4%"
            else:
                net = month_ret * position - COST_ROUND_TRIP * position
                reason = "持有到期"
            capital = capital * (1 + net)
            trades.append({
                'date': str(test_end.date()),
                'signal': 'BUY',
                'position': f"{position*100:.0f}%",
                'mom': round(mom, 2),
                'rsi': round(rsi, 1),
                'breadth': round(breadth, 2) if 'breadth' in locals() else 0.5,
                'month_ret': round(month_ret*100, 2),
                'net_ret': round(net*100, 2),
                'reason': reason,
                'capital': round(capital, 0)
            })
            if net > 0:
                win += 1
            else:
                loss += 1
        else:
            # 唔買嘅原因要寫清楚, 等你知係 FRED 定 Breadth 擋咗
            if not fred_ok:
                skip_reason = f"FRED擋: {fred_reason}"
                skipped_fred += 1
            elif not breadth_ok:
                skip_reason = f"廣度唔夠: {breadth:.0%} 升"
                skipped_breadth += 1
            elif not tech_ok:
                skip_reason = f"技術唔得: mom {mom:.2f}% rsi {rsi:.1f}"
            else:
                skip_reason = "無信號"
            trades.append({
                'date': str(test_end.date()),
                'signal': 'CASH',
                'position': "0%",
                'mom': round(mom, 2),
                'rsi': round(rsi, 1),
                'breadth': round(breadth, 2) if 'breadth' in locals() else 0.5,
                'month_ret': round(month_ret*100, 2),
                'net_ret': 0,
                'reason': skip_reason,
                'capital': round(capital, 0)
            })
        curve.append(capital)

    df = pd.DataFrame(trades)
    total = (capital / initial - 1) * 100
    hsi_bh = (s.iloc[-1] / s.iloc[train_months*20] - 1) * 100 if len(s) > 200 else 0
    curve_s = pd.Series(curve)
    max_dd = ((curve_s - curve_s.cummax()) / curve_s.cummax()).min() * 100 if len(curve_s) > 0 else 0

    print(df.to_string(index=False))
    print(f"\n=== 結果 ===")
    print(f"總回報 {total:+.2f}% | 恒指買入持有 {hsi_bh:+.2f}% | 超額 {total-hsi_bh:+.2f}%")
    print(f"勝率 {win}/{win+loss} = {win/(win+loss)*100:.0f}%  (只計買入嗰啲月)" if (win+loss) > 0 else "勝率: 無買入")
    print(f"最大回撤 {max_dd:.2f}% | 最終 ${capital:,.0f}")
    print(f"FRED 擋咗 {skipped_fred} 次 | 廣度擋咗 {skipped_breadth} 次 | 慳返好多手續費")

    # 評價
    if total > hsi_bh and total > 0:
        print("✅ 跑贏恒指 + 賺錢, 策略有效")
    elif total > hsi_bh:
        print("🟡 跑贏恒指但仲蝕錢, 風控有效但市況太差, 等市況好先入")
    else:
        print("🔴 跑輸恒指, 唔好用呢個策略實戰, 繼續紙上模擬")

    df.to_csv('output/backtest_result.csv', index=False, encoding='utf-8-sig')
    return df
