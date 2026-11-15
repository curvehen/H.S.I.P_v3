# src/config.py
import yaml, os
def load_config(path="config.yaml"):
    if not os.path.exists(path):
        path = "../config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
def get_all_tickers(cfg):
    return [cfg['hsi']] + list(cfg['top10'].keys()) + list(cfg.get('my_picks',{}).keys()) + list(cfg['macro'].keys())
def get_name_map(cfg):
    m = {cfg['hsi']: '恒指'}
    m.update(cfg['top10']); m.update(cfg.get('my_picks',{})); m.update(cfg['macro'])
    return m
