# H.S.I.P_v3

# HSI Predictor V2 - 實戰版
## 快速開始 (Colab)
!git clone https://github.com/YOUR_NAME/hsi-predictor-v2.git
%cd hsi-predictor-v2
!pip install -r requirements.txt -q
!python src/predict.py
## 每日自動化
GitHub Actions 每日 09:00 HKT 自動跑, 結果在 output/, Email 自動 Send
## 設定 Email
GitHub Secrets: EMAIL_USER, EMAIL_PASS (Gmail App Password), EMAIL_TO
## 加自選股
改 config.yaml -> my_picks, 例如:
my_picks:
  0981.HK: 中芯
  1024.HK: 快手
