# src/risk.py
import pandas as pd
from datetime import datetime
class RiskManager:
    def __init__(self, capital=100000):
        self.initial_capital=capital; self.capital=capital; self.position=0; self.entry_price=0
        self.consecutive_losses=0; self.is_halted=False; self.halt_until=None; self.trade_log=[]
    def check_halt(self):
        if self.consecutive_losses>=3 and not self.is_halted:
            self.is_halted=True; self.halt_until=datetime.now()+pd.Timedelta(days=7)
            print(f"連蝕3次, 停機到 {self.halt_until.date()}"); return True
        if self.is_halted and datetime.now()<self.halt_until: return True
        if self.is_halted and datetime.now()>=self.halt_until:
            self.is_halted=False; self.consecutive_losses=0
        return False
    def position_size(self, confidence, vol_pct):
        if confidence<0.60: return 0.0
        if confidence>=0.75: base=0.30
        elif confidence>=0.68: base=0.20
        elif confidence>=0.60: base=0.10
        else: base=0.0
        if vol_pct>3.0: return 0.0
        if vol_pct>2.0: base*=0.5
        return base
    def check_stop(self, current_price):
        if self.position==0: return None
        pnl_pct=(current_price/self.entry_price-1)*100
        if pnl_pct<=-2.0: self._close(current_price,"STOP_LOSS"); return "STOP_LOSS"
        if pnl_pct>=4.0: self._close(current_price,"TAKE_PROFIT"); return "TAKE_PROFIT"
        return None
    def open_trade(self, price, confidence, vol_pct, ticker="^HSI"):
        if self.check_halt(): return None
        size=self.position_size(confidence, vol_pct)
        if size==0: print(f"唔開倉 {ticker}"); return None
        amount=self.capital*size; self.position=1; self.entry_price=price
        print(f"開倉 {ticker} @ {price:.0f} 倉位 {size*100:.0f}% ${amount:,.0f}")
        self.trade_log.append({'date':str(datetime.now().date()),'action':'BUY','price':price,'size':size,'confidence':confidence})
        return {'price':price,'size':size,'amount':amount}
    def _close(self, price, reason):
        pnl=(price/self.entry_price-1)*self.capital*0.20 - self.capital*0.0018; self.capital+=pnl
        if pnl<0: self.consecutive_losses+=1
        else: self.consecutive_losses=0
        self.position=0; self.entry_price=0; self.trade_log.append({'date':str(datetime.now().date()),'action':reason,'price':price,'pnl':round(pnl,0)})
    def summary(self):
        print(f"初始 ${self.initial_capital:,.0f} | 而家 ${self.capital:,.0f} | 回報 {(self.capital/self.initial_capital-1)*100:+.2f}%")
        return pd.DataFrame(self.trade_log)
