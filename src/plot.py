# src/plot.py
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
def plot_all(prices, results, cfg):
    hsi_res=[r for r in results if r['ticker']==cfg['hsi']][0]
    hsi_prices=prices[cfg['hsi']].tail(60); last_date=hsi_prices.index[-1]
    future_dates=[last_date+timedelta(days=i+1) for i in range(5)]
    future_prices=[hsi_res['last_price']*(1+(hsi_res['pred_1d_pct']/100)*(i+1)) for i in range(5)]
    vol=hsi_res['vol']/100; upper=[p*(1+vol*np.sqrt(i+1)) for i,p in enumerate(future_prices)]; lower=[p*(1-vol*np.sqrt(i+1)) for i,p in enumerate(future_prices)]
    plt.figure(figsize=(12,6)); plt.plot(hsi_prices.index, hsi_prices.values, color='black', label='HSI past 60d')
    plt.plot(future_dates, future_prices, color='blue', marker='o', linewidth=2, label='Forecast 5d')
    plt.fill_between(future_dates, lower, upper, color='blue', alpha=0.15, label='GARCH interval')
    plt.axvline(last_date, color='red', linestyle='--', alpha=0.6)
    plt.title(f"HSI Forecast {hsi_res['last_date']} | {hsi_res['pred_1d_pct']:+.2f}% Conf {hsi_res['confidence']*100:.0f}% Vol {hsi_res['vol']:.2f}%", fontsize=13)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig('output/hsi_forecast.png', dpi=150); plt.show()
    plt.figure(figsize=(11,6))
    for r in results:
        if r['ticker']==cfg['hsi']: continue
        plt.scatter(r['pred_5d_pct'], r['confidence']*100, s=180, alpha=0.7); plt.text(r['pred_5d_pct']+0.08, r['confidence']*100+0.5, r['name'], fontsize=10)
    plt.axvline(0, color='red', linestyle='--', alpha=0.5); plt.xlabel('Pred 5d %'); plt.ylabel('Confidence %')
    plt.title('Top10 + Picks - Stock Ranking', fontsize=13); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig('output/stock_ranking.png', dpi=150); plt.show()
