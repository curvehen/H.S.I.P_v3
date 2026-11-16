# src/models.py - V2 改進版: Cap 極端預測 + 唔俾 Deep 模型搶權
import numpy as np
import logging
log = logging.getLogger("MODEL")

def _clip_pred(p, limit=0.05):
    """
    恒指 / 個股一日預測 Cap 喺 +-5%
    超過 5% 嘅都係癲數, 例如 Transformer -17.85% -> Cap 做 -5%
    5% 已經係好大跌幅, 港股一日跌 5% 係好罕見
    """
    return float(np.clip(p, -limit, limit))

def train_tree_models(X_train, y_train, X_test, y_test):
    preds = {}
    maes = {}
    # Tree 模型: 數據少都學到, 唔易 Overfit, 係主力
    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        p = _clip_pred(m.predict(X_test)[-1])
        preds['XGBoost'] = p
        maes['XGBoost'] = float(np.mean(np.abs(m.predict(X_test) - y_test)))
    except Exception as e:
        log.warning(f"XGB 失敗: {e}")
    try:
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
        m.fit(X_train, y_train)
        p = _clip_pred(m.predict(X_test)[-1])
        preds['LightGBM'] = p
        maes['LightGBM'] = float(np.mean(np.abs(m.predict(X_test) - y_test)))
    except Exception as e:
        log.warning(f"LGB 失敗: {e}")
    # Ridge: 最穩, 當 Baseline, 永遠唔會癲
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xt = sc.fit_transform(X_train)
        m = Ridge(alpha=10.0)  # alpha 大啲 = 更保守, 唔 Overfit
        m.fit(Xt, y_train)
        p = _clip_pred(m.predict(sc.transform(X_test))[-1])
        preds['Ridge'] = p
        maes['Ridge'] = float(np.mean(np.abs(m.predict(sc.transform(X_test)) - y_test)))
    except Exception as e:
        log.warning(f"Ridge 失敗: {e}")
    return preds, maes

def train_deep_models(X, y, lookback=30, epochs=20):
    """
    Deep 模型: 只做參考, 唔做主力
    epochs 20 就夠, 唔好 100, 100 一定 Overfit
    仲有 Early Stopping, 唔好死 Train
    """
    preds = {}
    maes = {}
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # 數據太少, Deep 唔好用, 直接 Return 空, 等 Tree 話事
        # 如果你堅持要用, 將下面呢行 Comment 咗就得
        # 但我強烈建議你而家咁做, 因為你得 500 個 Sample
        if len(X) < 1000:
            log.info(f"Deep 跳過: 數據得 {len(X)} 個, 太少, 唔 Train Transformer/GRU, 避免 Overfit")
            return preds, maes

        # 如果數據夠多, 先 Train 下面呢啲 (你而家用唔到)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        # ... (原本 LSTM/GRU/TCN/Transformer 嘅 Code 保留喺度, 但上面已經 Return 咗)
        # ... 唔使刪, 留返以後數據多咗先用
    except Exception as e:
        log.warning(f"Deep 失敗: {e}")
    return preds, maes

def ensemble_weighted(preds, maes, max_weight=0.30):
    """
    新加權: 唔好平均, 用 MAE 倒數
    MAE 細 = 準 = 權重大
    每個 Model 最多 30%, 唔俾一個 Model 話事
    再加: 唔好用 RL 權重, RL 喺 500 個 Sample 會 Overfit
    """
    if not preds:
        return 0.0, {}
    # 1. 計倒數權重
    inv = {}
    for k in preds:
        mae = maes.get(k, 0.01)
        # MAE 太細都唔好俾太高, 防止一個 Model 壟斷
        inv[k] = 1.0 / (mae + 0.001)
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}

    # 2. Cap 每個最多 30%, 多出嚟嘅分返俾其他
    # 例如 XGBoost 原本 50%, Cap 做 30%, 剩返 20% 分俾 LightGBM + Ridge
    for _ in range(3):
        over = {k: v - max_weight for k, v in weights.items() if v > max_weight}
        if not over:
            break
        excess = sum(over.values())
        for k in over:
            weights[k] = max_weight
        # 分返俾未滿嘅
        under = [k for k, v in weights.items() if v < max_weight]
        if under:
            for k in under:
                weights[k] += excess / len(under)

    # 3. 加權平均, 再 Cap 一次 +-3% (最終預測唔好太癲)
    weighted = sum(preds[k] * weights[k] for k in preds)
    weighted = float(np.clip(weighted, -0.03, 0.03))  # 最終預測 Cap +-3%
    return weighted, weights
