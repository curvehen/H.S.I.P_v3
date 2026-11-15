# src/volatility.py
import numpy as np, logging
log = logging.getLogger(__name__)
try:
    from arch import arch_model; HAS_ARCH=True
except: HAS_ARCH=False
def garch_vol(ret_series):
    try:
        if HAS_ARCH:
            r=ret_series.dropna()*100
            am=arch_model(r, vol='Garch', p=1, q=1, rescale=False)
            res=am.fit(disp='off', show_warning=False)
            f=res.forecast(horizon=1); v=float(np.sqrt(f.variance.values[-1,0])/100)
            if not np.isnan(v) and v>0: return v
    except Exception as e: log.warning(f"GARCH Fallback: {e}")
    return float(ret_series.ewm(span=10).std().iloc[-1])
def bayesian_prob(pred_mean, pred_vol):
    try:
        samples=np.random.normal(loc=pred_mean, scale=max(pred_vol,0.005), size=5000)
        return float((samples>0).mean())
    except: return 0.5
def monte_carlo(pred_mean, vol, n=1000):
    try:
        mc=np.random.normal(loc=pred_mean, scale=vol, size=n)
        return float((mc>0).mean()), float(np.percentile(mc,10)), float(np.percentile(mc,90))
    except: return 0.5, -vol, vol
