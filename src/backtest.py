# src/backtest.py
import numpy as np, pandas as pd
COST=0.0003+0.001+0.0005
def walk_forward_backtest(prices, target='^HSI', initial=100000, train_months=12):
    print(f"=== Walk-Forward Backtest {target} | 成本 {COST*100:.2f}% ===")
    s=prices[target].dropna(); monthly=s.resample('ME').last(); dates=monthly.index
    capital=initial; trades=[]; win=loss=0; curve=[]
    for i in range(train_months, len(dates)-1):
        train_end=dates[i]; test_end=dates[i+1]
        train_data=s[s.index<=train_end]; test_data=s[(s.index>train_end)&(s.index<=test_end)]
        if len(test_data)<5 or len(train_data)<100: continue
        last_20=train_data.tail(20); mom=(last_20.iloc[-1]/last_20.iloc[0]-1)*100
        delta=train_data.diff(); gain=delta.where(delta>0,0).rolling(14).mean().iloc[-1]; lv=-delta.where(delta<0,0).rolling(14).mean().iloc[-1]
        rsi=100-(100/(1+gain/(lv+1e-9)))
        signal=1 if (mom>0 and rsi<70) else 0
        month_ret=(test_data.iloc[-1]/test_data.iloc[0]-1)
        if signal==1:
            net=month_ret-COST*2; capital=capital*(1+net)
            trades.append({'date':str(test_end.date()),'signal':'BUY','month_ret':round(month_ret*100,2),'net_ret':round(net*100,2),'capital':round(capital,0)})
            win+=1 if net>0 else 0; loss+=1 if net<=0 else 0
        else:
            trades.append({'date':str(test_end.date()),'signal':'CASH','month_ret':round(month_ret*100,2),'net_ret':0,'capital':round(capital,0)})
        curve.append(capital)
    df=pd.DataFrame(trades); total=(capital/initial-1)*100
    hsi_bh=(s.iloc[-1]/s.iloc[train_months*20]-1)*100 if len(s)>200 else 0
    curve_s=pd.Series(curve); max_dd=((curve_s-curve_s.cummax())/curve_s.cummax()).min()*100 if len(curve_s)>0 else 0
    print(df.to_string(index=False))
    print(f"總回報 {total:+.2f}% | 恒指 {hsi_bh:+.2f}% | 超額 {total-hsi_bh:+.2f}% | 勝率 {win}/{win+loss} | 最大回撤 {max_dd:.2f}% | 最終 ${capital:,.0f}")
    df.to_csv('output/backtest_result.csv', index=False, encoding='utf-8-sig')
    return df
