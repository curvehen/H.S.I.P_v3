# src/models.py - FULL 版 - Tree 主力 + Deep 開關 + MAE 加權 + Cap 癲數
import numpy as np
import logging
log = logging.getLogger("MODEL")

def _clip_pred(p, limit=0.05):
    """
    單一模型預測 Cap 喺 +-5%
    超過 5% 嘅都係癲數, 例如 Transformer -17.85% -> Cap 做 -5%
    恒指一日跌 5% 已經係好大跌幅
    """
    return float(np.clip(p, -limit, limit))

def train_tree_models(X_train, y_train, X_test, y_test):
    """
    Tree 模型: 數據少都學到, 唔易 Overfit, 係主力
    返嚟: preds = {'XGBoost': 0.008, 'LightGBM': 0.007, 'Ridge': 0.003}
          maes = {'XGBoost': 0.012, 'LightGBM': 0.013, 'Ridge': 0.011}
    """
    preds = {}
    maes = {}

    # --- XGBoost ---
    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(
            n_estimators=100,
            max_depth=3,          # 唔好太深, 3 就夠, 深咗會 Overfit
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_train, y_train)
        p = _clip_pred(m.predict(X_test)[-1])
        preds['XGBoost'] = p
        maes['XGBoost'] = float(np.mean(np.abs(m.predict(X_test) - y_test)))
    except Exception as e:
        log.warning(f"XGB 失敗: {e}")

    # --- LightGBM ---
    try:
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            verbose=-1,
            random_state=42
        )
        m.fit(X_train, y_train)
        p = _clip_pred(m.predict(X_test)[-1])
        preds['LightGBM'] = p
        maes['LightGBM'] = float(np.mean(np.abs(m.predict(X_test) - y_test)))
    except Exception as e:
        log.warning(f"LGB 失敗: {e}")

    # --- Ridge: 最穩, 當 Baseline, 永遠唔會癲 ---
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xt = sc.fit_transform(X_train)
        # alpha 大啲 = 更保守, 唔 Overfit
        m = Ridge(alpha=10.0)
        m.fit(Xt, y_train)
        p = _clip_pred(m.predict(sc.transform(X_test))[-1])
        preds['Ridge'] = p
        maes['Ridge'] = float(np.mean(np.abs(m.predict(sc.transform(X_test)) - y_test)))
    except Exception as e:
        log.warning(f"Ridge 失敗: {e}")

    return preds, maes

def _make_sequences(X, y, lookback):
    """將 2D 數據轉做 3D Sequence, 俾 LSTM/GRU 用"""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_deep_models(X, y, lookback=30, epochs=20):
    """
    Deep 模型: LSTM / GRU / TCN / Transformer
    返嚟: preds, maes, 如果數據唔夠就 Return 空, 等 Tree 話事

    開關喺下面:
    如果你堅持要用 Deep, 就去 Comment 咗 `if len(X) < 1000:` 嗰 3 行
    但 500 個 Sample 用 Deep, 100% Overfit, 會出返 GRU +7% / Transformer -17% 嗰啲癲數
    """
    preds = {}
    maes = {}

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # ========== Deep 開關 - COMMENT WHERE? 就係呢 3 行 ==========
        # 如果你堅持要用, 將下面呢 3 行前面加 # Comment 咗佢
        # 唔 Comment, 數據 < 1000 就自動跳過 Deep
        # Comment 咗, 就算得 500 個 Sample 都強制 Train Deep
        if len(X) < 1000:
            log.info(f"Deep 跳過: 數據得 {len(X)} 個, 太少, 唔 Train Transformer/GRU, 避免 Overfit")
            return preds, maes
        # =========================================================

        # 數據夠多 (1000+), 先至入到嚟呢度
        log.info(f"Deep 開始: 數據 {len(X)} 個, 夠多, 開始 Train LSTM/GRU/TCN/Transformer")

        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        X_seq, y_seq = _make_sequences(Xs, y, lookback)
        if len(X_seq) < 100:
            log.warning(f"Deep Sequence 太少: {len(X_seq)}, 跳過")
            return preds, maes

        # Train / Test 分割: 最尾 20 個做 Test
        X_train = X_seq[:-20]
        y_train = y_seq[:-20]
        X_test = X_seq[-20:]
        y_test = y_seq[-20:]
        n_feat = X_seq.shape[2]

        device = torch.device("cpu")

        # ---------- LSTM ----------
        try:
            class LSTMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(n_feat, 32, batch_first=True)
                    self.fc = nn.Linear(32, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :]).squeeze(1)

            model = LSTMModel().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
            yt = torch.tensor(y_train, dtype=torch.float32).to(device)
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = _clip_pred(model(torch.tensor(X_test, dtype=torch.float32).to(device))[-1].item())
                pred_all = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            preds['LSTM'] = p
            maes['LSTM'] = float(np.mean(np.abs(pred_all - y_test)))
        except Exception as e:
            log.warning(f"LSTM 失敗: {e}")

        # ---------- GRU ----------
        try:
            class GRUModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = nn.GRU(n_feat, 32, batch_first=True)
                    self.fc = nn.Linear(32, 1)
                def forward(self, x):
                    out, _ = self.gru(x)
                    return self.fc(out[:, -1, :]).squeeze(1)

            model = GRUModel().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
            yt = torch.tensor(y_train, dtype=torch.float32).to(device)
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = _clip_pred(model(torch.tensor(X_test, dtype=torch.float32).to(device))[-1].item())
                pred_all = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            preds['GRU'] = p
            maes['GRU'] = float(np.mean(np.abs(pred_all - y_test)))
        except Exception as e:
            log.warning(f"GRU 失敗: {e}")

        # ---------- TCN (1D Conv) ----------
        try:
            class TCNModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv1d(n_feat, 32, kernel_size=3, padding=1)
                    self.relu = nn.ReLU()
                    self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
                    self.fc = nn.Linear(32, 1)
                def forward(self, x):
                    # x: (batch, seq, feat) -> (batch, feat, seq)
                    x = x.permute(0, 2, 1)
                    x = self.relu(self.conv1(x))
                    x = self.relu(self.conv2(x))
                    # 取最後一個時間點
                    x = x[:, :, -1]
                    return self.fc(x).squeeze(1)

            model = TCNModel().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
            yt = torch.tensor(y_train, dtype=torch.float32).to(device)
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = _clip_pred(model(torch.tensor(X_test, dtype=torch.float32).to(device))[-1].item())
                pred_all = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            preds['TCN'] = p
            maes['TCN'] = float(np.mean(np.abs(pred_all - y_test)))
        except Exception as e:
            log.warning(f"TCN 失敗: {e}")

        # ---------- Transformer ----------
        try:
            class TransformerModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_proj = nn.Linear(n_feat, 32)
                    enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
                    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
                    self.fc = nn.Linear(32, 1)
                def forward(self, x):
                    x = self.input_proj(x)
                    x = self.encoder(x)
                    return self.fc(x[:, -1, :]).squeeze(1)

            model = TransformerModel().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
            yt = torch.tensor(y_train, dtype=torch.float32).to(device)
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = _clip_pred(model(torch.tensor(X_test, dtype=torch.float32).to(device))[-1].item())
                pred_all = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
            preds['Transformer'] = p
            maes['Transformer'] = float(np.mean(np.abs(pred_all - y_test)))
        except Exception as e:
            log.warning(f"Transformer 失敗: {e}")

    except Exception as e:
        log.warning(f"Deep 整體失敗: {e}")

    return preds, maes

def ensemble_weighted(preds, maes, max_weight=0.30):
    """
    新加權: 唔好平均, 用 MAE 倒數
    MAE 細 = 準 = 權重大
    每個 Model 最多 30%, 唔俾一個 Model 話事
    最終預測 Cap +-3%, 唔好太癲
    唔用 RL, RL 喺 500 個 Sample 會 Overfit

    返嚟: weighted, weights
    例如: weighted=0.0063, weights={'XGBoost':0.33, 'LightGBM':0.33, 'Ridge':0.33}
    """
    if not preds:
        return 0.0, {}

    # 1. 計倒數權重
    inv = {}
    for k in preds:
        mae = maes.get(k, 0.01)
        inv[k] = 1.0 / (mae + 0.001)
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    # 2. Cap 每個最多 30%, 多出嚟嘅分返俾其他
    for _ in range(3):
        over = {k: v - max_weight for k, v in weights.items() if v > max_weight}
        if not over:
            break
        excess = sum(over.values())
        for k in over:
            weights[k] = max_weight
        under = [k for k, v in weights.items() if v < max_weight]
        if under:
            for k in under:
                weights[k] += excess / len(under)

    # 3. 加權平均, 再 Cap 一次 +-3%
    weighted = sum(preds[k] * weights[k] for k in preds)
    weighted = float(np.clip(weighted, -0.03, 0.03))
    return weighted, weights
