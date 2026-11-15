# src/email_sender.py
import smtplib, os, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
def calc_levels(last_price, pred_1d_pct, vol_pct, confidence):
    pred = pred_1d_pct / 100; vol = vol_pct / 100
    high = last_price * (1 + max(pred, 0) + vol)
    low = last_price * (1 - vol + min(pred, 0))
    if pred < 0:
        high = last_price * (1 + vol * 0.5); low = last_price * (1 + pred - vol)
    entry = last_price * (1 - vol * 0.3) if pred > 0 else last_price * (1 + vol * 0.2)
    exit_price = last_price * (1 + pred) if pred > 0 else last_price * (1 + pred * 0.5)
    stop_loss = entry * 0.98
    if confidence < 0.60: stop_loss = entry * 0.99
    signal = '買入' if (pred > 0.003 and confidence >= 0.60) else ('觀望' if confidence < 0.60 else '唔買')
    return {'high': round(high,2),'low': round(low,2),'entry': round(entry,2),'exit': round(exit_price,2),'stop': round(stop_loss,2),'signal': signal}
def build_email_html(results, backtest_summary=""):
    today = datetime.now().strftime("%Y-%m-%d")
    rows=""
    for r in results:
        lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
        color = "#d4edda" if lv['signal']=='買入' else ("#fff3cd" if lv['signal']=='觀望' else "#f8d7da")
        rows+=f'<tr style="background:{color}"><td><b>{r["name"]}</b><br><small>{r["ticker"]}</small></td><td>{r["last_price"]:.2f}</td><td>{lv["high"]:.2f}</td><td>{lv["low"]:.2f}</td><td>{lv["entry"]:.2f}</td><td>{lv["exit"]:.2f}</td><td>{lv["stop"]:.2f}</td><td>{r["pred_1d_pct"]:+.2f}%</td><td>{r["confidence"]*100:.0f}%</td><td>{r["vol"]:.2f}%</td><td><b>{lv["signal"]}</b></td></tr>'
    html=f'<html><body style="font-family:Arial"><h2>HSI V2 - {today}</h2><table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; text-align:center;"><tr style="background:#343a40;color:white;"><th>股票/指數</th><th>現價</th><th>預測高位</th><th>預測低位</th><th>入場位</th><th>離場位</th><th>止蝕位</th><th>預測</th><th>信心</th><th>波動</th><th>操作</th></tr>{rows}</table><p>{backtest_summary}</p><p style="color:#888;font-size:12px;">僅供研究, 非投資建議, 止蝕一定要執行</p></body></html>'
    return html
def send_email(results, backtest_summary=""):
    user=os.getenv("EMAIL_USER"); pwd=os.getenv("EMAIL_PASS"); to=os.getenv("EMAIL_TO", user)
    if not user or not pwd:
        print("未設定 EMAIL_USER/PASS, 跳過 Email"); return False
    today=datetime.now().strftime("%Y-%m-%d")
    hsi=[r for r in results if r['ticker']=='^HSI'][0] if any(r['ticker']=='^HSI' for r in results) else results[0]
    subject=f"HSI預測 {today} 恒指 {hsi['pred_1d_pct']:+.2f}% 信心 {hsi['confidence']*100:.0f}%"
    msg=MIMEMultipart(); msg['From']=user; msg['To']=to; msg['Subject']=subject
    msg.attach(MIMEText(build_email_html(results, backtest_summary), 'html', 'utf-8'))
    for p in ["output/hsi_forecast.png","output/stock_ranking.png"]:
        try:
            with open(p,'rb') as f: img=MIMEImage(f.read()); msg.attach(img)
        except: pass
    try:
        ctx=ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com",465,context=ctx) as s:
            s.login(user,pwd); s.send_message(msg)
        print(f"Email 已 Send 去 {to}"); return True
    except Exception as e:
        print(f"Send Email 失敗: {e}"); return False
