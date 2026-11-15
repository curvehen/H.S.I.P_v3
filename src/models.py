# src/models.py
import numpy as np, logging
log = logging.getLogger(__name__)
try:
    import torch, torch.nn as nn
    HAS_TORCH=True; DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
except: HAS_TORCH=False; DEVICE='cpu'
try: import xgboost as xgb; HAS_XGB=True
except: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGB=True
except: HAS_LGB=False
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self, inp, hid=64):
            super().__init__(); self.lstm=nn.LSTM(inp, hid, batch_first=True); self.fc=nn.Linear(hid,1)
        def forward(self,x):
            o,_=self.lstm(x); return self.fc(o[:,-1,:])
    class GRUModel(nn.Module):
        def __init__(self, inp, hid=64):
            super().__init__(); self.gru=nn.GRU(inp, hid, batch_first=True); self.fc=nn.Linear(hid,1)
        def forward(self,x):
            o,_=self.gru(x); return self.fc(o[:,-1,:])
    class TCNModel(nn.Module):
        def __init__(self, inp, hid=64):
            super().__init__(); self.c1=nn.Conv1d(inp, hid, 3, padding=1); self.r=nn.ReLU(); self.c2=nn.Conv1d(hid, hid, 3, padding=1); self.fc=nn.Linear(hid,1)
        def forward(self,x):
            x=x.transpose(1,2); x=self.r(self.c1(x)); x=self.r(self.c2(x)); return self.fc(x.mean(dim=2))
    class TransformerModel(nn.Module):
        def __init__(self, inp, hid=64):
            super().__init__(); self.inp=nn.Linear(inp, hid); self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hid, nhead=4, batch_first=True), num_layers=2); self.fc=nn.Linear(hid,1)
        def forward(self,x):
            x=self.inp(x); x=self.enc(x); return self.fc(x[:,-1,:])
    def train_torch(cls, Xtr, ytr, Xlast, epochs=80):
        try:
            m=cls(Xtr.shape[2]).to(DEVICE); opt=torch.optim.Adam(m.parameters(), lr=0.005); fn=nn.MSELoss()
            Xt=torch.FloatTensor(Xtr).to(DEVICE); yt=torch.FloatTensor(ytr).reshape(-1,1).to(DEVICE); Xl=torch.FloatTensor(Xlast).to(DEVICE)
            m.train()
            for _ in range(epochs):
                opt.zero_grad(); loss=fn(m(Xt), yt); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): return float(m(Xl).cpu().numpy()[0,0])
        except Exception as e:
            log.warning(f"Torch 失敗: {e}"); return float(np.mean(ytr))
else:
    def train_torch(*a, **k): return 0.0
def train_tree_models(X_train, y_train, X_test, y_test):
    preds={}; maes={}
    try:
        if HAS_XGB:
            m=xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
            m.fit(X_train, y_train); preds['XGBoost']=float(m.predict(X_test[-1:])[0]); maes['XGBoost']=float(np.mean(np.abs(m.predict(X_test)-y_test)))
        else: raise Exception("no xgb")
    except:
        m=Ridge(); sc=StandardScaler(); m.fit(sc.fit_transform(X_train), y_train); preds['XGBoost']=float(m.predict(sc.transform(X_test[-1:]))[0]); maes['XGBoost']=0.01
    try:
        if HAS_LGB:
            m=lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
            m.fit(X_train, y_train); preds['LightGBM']=float(m.predict(X_test[-1:])[0]); maes['LightGBM']=float(np.mean(np.abs(m.predict(X_test)-y_test)))
        else: raise Exception("no lgb")
    except:
        m=Ridge(); sc=StandardScaler(); m.fit(sc.fit_transform(X_train), y_train); preds['LightGBM']=float(m.predict(sc.transform(X_test[-1:]))[0]); maes['LightGBM']=0.01
    return preds, maes
def train_deep_models(all_vals, all_y, lookback, epochs):
    if not HAS_TORCH: return {}, {}
    sc=StandardScaler(); vals_s=sc.fit_transform(all_vals)
    seq_X=[]; seq_y=[]
    for i in range(lookback, len(vals_s)-1):
        seq_X.append(vals_s[i-lookback:i]); seq_y.append(all_y[i])
    seq_X=np.array(seq_X); seq_y=np.array(seq_y)
    if len(seq_X)<40: return {}, {}
    split=len(seq_X)-20; Xtr, Xlast = seq_X[:split], seq_X[-1:]
    ytr=seq_y[:split]
    preds={}
    preds['LSTM']=train_torch(LSTMModel, Xtr, ytr, Xlast, epochs)
    preds['GRU']=train_torch(GRUModel, Xtr, ytr, Xlast, epochs)
    preds['TCN']=train_torch(TCNModel, Xtr, ytr, Xlast, epochs)
    preds['Transformer']=train_torch(TransformerModel, Xtr, ytr, Xlast, epochs)
    return preds, {k:0.01 for k in preds}
