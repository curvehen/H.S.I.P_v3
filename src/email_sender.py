# src/email_sender.py - 每日 Send Email - 完整版
# 功能: 多人收件 + 高低位/入場/離場/止蝕 + FRED + Factor + 模型明細
import smtplib, os, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

def calc_levels(last_price, pred_1d_pct, vol_pct, confidence):
    """
    計算高位 / 低位 / 入場 / 離場 / 止蝕
    高位 = 現價 * (1 + 波動 + 預測升幅)
    低位 = 現價 * (1 - 波動 + 預測跌幅)
    入場 = 預測升 -> 等回調 0.3*波動先入, 唔追高
           預測跌 -> 等反彈, 入場位略高於現價
    離場 = 現價 * (1 + 預測升幅)
    止蝕 = 入場價 * 0.98, 信心低就收窄到 0.99
    """
    pred = pred_1d_pct / 100
    vol = vol_pct / 100
    high = last_price * (1 + max(pred, 0) + vol)
    low = last_price * (1 - vol + min(pred, 0))
    if pred < 0:
        high = last_price * (1 + vol * 0.5)
        low = last_price * (1 + pred - vol)
    entry = last_price * (1 - vol * 0.3) if pred > 0 else last_price * (1 + vol * 0.2)
    exit_price = last_price * (1 + pred) if pred > 0 else last_price * (1 + pred * 0.5)
    stop_loss = entry * 0.98
    if confidence < 0.60:
        stop_loss = entry * 0.99
    signal = '買入' if (pred > 0.003 and confidence >= 0.60) else ('觀望' if confidence < 0.60 else '唔買')
    return {
        'high': round(high, 2),
        'low': round(low, 2),
        'entry': round(entry, 2),
        'exit': round(exit_price, 2),
        'stop': round(stop_loss, 2),
        'signal': signal
    }

def build_email_html(results, backtest_summary="", fred_sig=None, fred_text=""):
    """
    完整 Email HTML - 呢個係你問嗰個 function 嘅完整版
    results: list of dict from predict.py, 每個要有 ticker, name, last_price, pred_1d_pct, vol, confidence, factor, models, rl_weights
    backtest_summary: str, 例如 "Backtest 總回報 +12% ..."
    fred_sig: dict from fred_data.fred_signals(), 例如 {'YieldCurve': 0.15, 'US10Y_FRED': 4.32, ...}
    fred_text: str, 備用, 如果你冇傳 fred_sig, 直接傳文字都得
    """
    today = datetime.now().strftime("%Y-%m-%d")
    weekday = datetime.now().strftime("%A")

    # --- 1. 搵恒指做 headline ---
    hsi = None
    for r in results:
        if r.get('ticker') == '^HSI':
            hsi = r
            break
    if hsi is None and len(results) > 0:
        hsi = results[0]

    hsi_lv = calc_levels(hsi['last_price'], hsi['pred_1d_pct'], hsi['vol'], hsi['confidence']) if hsi else {}

    # --- 2. 買入清單 ---
    buy_list = [r for r in results if calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])['signal'] == '買入']
    buy_names = ", ".join([f"{r['name']} ({r['pred_1d_pct']:+.2f}%)" for r in buy_list]) if buy_list else "今日無買入信號, 全部觀望"

    # --- 3. 主表格 Rows ---
    rows = ""
    for r in results:
        lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
        # 顏色: 買入綠, 觀望黃, 唔買紅
        if lv['signal'] == '買入':
            color = "#d4edda"
            signal_html = f"<b style='color:#155724'>🟢 買入</b>"
        elif lv['signal'] == '觀望':
            color = "#fff3cd"
            signal_html = f"<b style='color:#856404'>🟡 觀望</b>"
        else:
            color = "#f8d7da"
            signal_html = f"<b style='color:#721c24'>🔴 唔買</b>"

        # 升跌顏色
        pred_color = "#0a7a42" if r['pred_1d_pct'] >= 0 else "#c0392b"
        pred_str = f"<b style='color:{pred_color}'>{r['pred_1d_pct']:+.2f}%</b>"

        rows += f"""
        <tr style="background:{color}">
            <td style="text-align:left"><b>{r['name']}</b><br><small style="color:#666">{r['ticker']}</small></td>
            <td>{r['last_price']:.2f}</td>
            <td style="color:#1a5fb4"><b>{lv['high']:.2f}</b></td>
            <td style="color:#e67e22"><b>{lv['low']:.2f}</b></td>
            <td style="color:#0a7a42"><b>{lv['entry']:.2f}</b></td>
            <td><b>{lv['exit']:.2f}</b></td>
            <td style="color:#c0392b"><b>{lv['stop']:.2f}</b></td>
            <td>{pred_str}</td>
            <td>{r['confidence']*100:.0f}%</td>
            <td>{r['vol']:.2f}%</td>
            <td>{signal_html}</td>
        </tr>
        """

    # --- 4. FRED Block ---
    fred_html = ""
    if fred_sig and isinstance(fred_sig, dict) and len(fred_sig) > 0:
        # 安全取值, 唔好因為缺一個 key 而成個 Email 爆
        us10y = fred_sig.get('US10Y_FRED', '-')
        fed = fred_sig.get('FedFunds', '-')
        yc = fred_sig.get('YieldCurve', '-')
        yc_signal = fred_sig.get('YieldCurve_Signal', '')
        cpi_yoy = fred_sig.get('US_CPI_YoY', None)
        unrate = fred_sig.get('US_Unemployment', '-')
        un_chg = fred_sig.get('Unemployment_3M_Change', None)
        hy = fred_sig.get('HY_Spread', '-')
        credit = fred_sig.get('Credit_Stress', '')

        # 格式化
        us10y_str = f"{us10y:.2f}%" if isinstance(us10y, (int, float)) else str(us10y)
        fed_str = f"{fed:.2f}%" if isinstance(fed, (int, float)) else str(fed)
        yc_str = f"{yc:.2f}%" if isinstance(yc, (int, float)) else str(yc)
        cpi_str = f"{cpi_yoy:.2f}%" if isinstance(cpi_yoy, (int, float)) else "-"
        un_str = f"{unrate:.2f}%" if isinstance(unrate, (int, float)) else str(unrate)
        un_chg_str = f"({un_chg:+.2f}% 3個月)" if isinstance(un_chg, (int, float)) else ""
        hy_str = f"{hy:.2f}%" if isinstance(hy, (int, float)) else str(hy)

        # Yield Curve 顏色
        yc_color = "#c0392b" if isinstance(yc, (int, float)) and yc < 0 else "#0a7a42"
        hy_color = "#c0392b" if credit == '高壓' else ("#e67e22" if credit == '留意' else "#0a7a42")

        fred_html = f"""
        <h3>🇺🇸 FRED 美國宏觀 (FRED API)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; text-align:center; font-size:13px;">
        <tr style="background:#1f3a5f; color:white;">
            <th>美國10Y</th><th>Fed Funds</th><th>收益率曲線 10Y-2Y</th><th>CPI YoY</th><th>失業率</th><th>高收益利差</th>
        </tr>
        <tr>
            <td><b>{us10y_str}</b></td>
            <td><b>{fed_str}</b></td>
            <td><b style="color:{yc_color}">{yc_str}<br><small>{yc_signal}</small></b></td>
            <td><b>{cpi_str}</b></td>
            <td><b>{un_str}</b><br><small>{un_chg_str}</small></td>
            <td><b style="color:{hy_color}">{hy_str}<br><small>{credit}</small></b></td>
        </tr>
        </table>
        <p style="font-size:12px; color:#555;">
        解讀: 10Y 升 = 資金流出港股, 唔利好 | 收益率曲線 &lt; 0 = 衰退警告, 要減倉 |
        HY 利差 &gt; 5% = 信貸緊張, 要小心 | 失業率 3個月急升 = 經濟轉差
        </p>
        """
    elif fred_text:
        fred_html = f"""
        <h3>🇺🇸 FRED 美國宏觀</h3>
        <p>{fred_text}</p>
        """
    else:
        fred_html = """
        <h3>🇺🇸 FRED 美國宏觀</h3>
        <p style="color:#888;">未設定 FRED_API_KEY, 未有 FRED 數據。去 GitHub Secrets 加 FRED_API_KEY 即可啟用。</p>
        """

    # --- 5. Factor + 模型明細 (只 show 恒指, 唔好成張 Email 太長) ---
    factor_html = ""
    if hsi and hsi.get('factor'):
        factor_items = ""
        for k, v in hsi['factor'].items():
            # v 係貢獻, 轉做 %
            try:
                v_pct = float(v) * 100
                c = "#0a7a42" if v_pct >= 0 else "#c0392b"
                factor_items += f"<li>{k}: <b style='color:{c}'>{v_pct:+.2f}%</b></li>"
            except:
                factor_items += f"<li>{k}: {v}</li>"
        factor_html += f"<h3>🔍 恒指 Factor 解釋 (點解咁預測)</h3><ul>{factor_items}</ul>"

    model_html = ""
    if hsi and hsi.get('models'):
        model_items = ""
        for k, v in hsi['models'].items():
            try:
                model_items += f"<li>{k}: {float(v)*100:+.2f}%</li>"
            except:
                pass
        rl_best = max(hsi['rl_weights'], key=hsi['rl_weights'].get) if hsi.get('rl_weights') else "-"
        rl_weights_str = ", ".join([f"{k} {v*100:.0f}%" for k, v in hsi.get('rl_weights', {}).items()])
        model_html = f"""
        <h3>🤖 6 個模型預測 (恒指)</h3>
        <ul>{model_items}</ul>
        <p><b>RL 最信:</b> {rl_best} | <b>RL 權重:</b> {rl_weights_str}</p>
        <p><b>Bayesian 升概率:</b> {hsi.get('prob_up', 0):.1f}% | <b>Monte Carlo 升概率:</b> {hsi.get('mc_prob_up', 0):.1f}% | <b>MC 區間:</b> {hsi.get('mc_range', (0,0))[0]:.2f}% ~ {hsi.get('mc_range', (0,0))[1]:.2f}%</p>
        """

    # --- 6. Top10 詳細 (每個股票嘅 Factor, 唔係淨恒指) ---
    detail_rows = ""
    for r in results:
        lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
        detail_rows += f"<li><b>{r['name']}</b> {r['ticker']}: 現價 {r['last_price']:.2f} -> 入場 {lv['entry']:.2f} / 止蝕 {lv['stop']:.2f} / 離場 {lv['exit']:.2f} / 高 {lv['high']:.2f} / 低 {lv['low']:.2f} | 預測 {r['pred_1d_pct']:+.2f}% 信心 {r['confidence']*100:.0f}% -> {lv['signal']}</li>"

    # --- 7. 組合 HTML ---
    html = f"""
    <html><body style="font-family:Arial, sans-serif; font-size:14px; color:#222; line-height:1.5; max-width:900px; margin:auto;">
    <h2 style="margin-bottom:4px;">📈 HSI 終極版 V2 - 每日預測 {today} ({weekday})</h2>
    <p style="color:#555; margin-top:0;">恒指 + Top10 + 自選 | 6 模型 + GARCH + Bayesian + Monte Carlo + Factor + RL + 風控 + FRED</p>

    <div style="background:#eef4ff; border:1px solid #b6c9ff; padding:12px; border-radius:8px;">
        <b>恒指今日:</b> 現價 {hsi['last_price']:.2f} | 預測聽日 <b>{hsi['pred_1d_pct']:+.2f}%</b> | 信心 {hsi['confidence']*100:.0f}% | 波動 {hsi['vol']:.2f}%<br>
        <b>入場:</b> {hsi_lv.get('entry','-')} | <b>止蝕:</b> {hsi_lv.get('stop','-')} | <b>離場:</b> {hsi_lv.get('exit','-')} | <b>高位:</b> {hsi_lv.get('high','-')} | <b>低位:</b> {hsi_lv.get('low','-')} | <b>操作:</b> {hsi_lv.get('signal','-')}<br>
        <b>今日買入清單:</b> {buy_names}
    </div>

    <p><b>點睇:</b> 入場位 = 等回調先買, 唔好追高 | 止蝕位 = 跌穿即斬, 唔好死攬 | 離場位 = 到價走一半, 唔好貪</p>

    <h3>📋 股票 / 指數 - 高位 / 低位 / 入場 / 離場 / 止蝕</h3>
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; text-align:center; font-size:13px;">
    <tr style="background:#343a40; color:white;">
        <th style="text-align:left;">股票 / 指數</th><th>現價</th><th>預測高位</th><th>預測低位</th><th>入場位</th><th>離場位</th><th>止蝕位</th><th>預測升跌</th><th>信心</th><th>波動</th><th>操作</th>
    </tr>
    {rows}
    </table>

    {fred_html}

    {factor_html}

    {model_html}

    <h3>📝 詳細操作清單 (方便複製去 Notes)</h3>
    <ul>{detail_rows}</ul>

    <h3>📊 Backtest</h3>
    <p>{backtest_summary if backtest_summary else "Backtest 會喺 predict.py 每次跑完自動更新, 見 output/backtest_result.csv"}</p>

    <h3>🧠 LLM 反證 Prompt (Copy 去 ChatGPT / Meta AI)</h3>
    <div style="background:#f8f9fa; border:1px solid #ddd; padding:10px; border-radius:6px; font-size:13px;">
    你係一個沽空基金經理, 專門搵預測漏洞, 唔好讚, 要挑骨頭。<br>
    今日預測: 恒指 {hsi['pred_1d_pct']:+.2f}%, 信心 {hsi['confidence']*100:.0f}%, 波動 {hsi['vol']:.2f}%, 現價 {hsi['last_price']:.2f}, 高位 {hsi_lv.get('high','-')}, 低位 {hsi_lv.get('low','-')}, 入場 {hsi_lv.get('entry','-')}, 止蝕 {hsi_lv.get('stop','-')}<br>
    Factor: {hsi.get('factor','-')}<br>
    各模型: {hsi.get('models','-')}<br>
    請輸出: 1. 3個最大風險 2. 如果沽空, 沽邊隻/邊個位/止蝕 3. 有冇偷睇未來/過擬合 4. 今日買/唔買/反手沽, 明確決定。
    </div>

    <p style="color:#888; font-size:12px; margin-top:20px;">⚠️ 僅供研究參考, 非投資建議。先紙上模擬 1 個月, Backtest 跑贏恒指先考慮小注實戰。止蝕一定要執行。</p>
    </body></html>
    """
    return html

def send_email(results, backtest_summary="", fred_sig=None, fred_text=""):
    """
    Send to multiple recipients
    EMAIL_TO 格式: "a@gmail.com, b@gmail.com" 或 "a@gmail.com; b@gmail.com"
    """
    user = os.getenv("EMAIL_USER")
    pwd = os.getenv("EMAIL_PASS")
    to_raw = os.getenv("EMAIL_TO", user or "")
    # 多人: 用 , 或 ; 分開
    to_list = [e.strip() for e in to_raw.replace(";", ",").split(",") if e.strip()]
    if not user or not pwd or not to_list:
        print("⚠️ 未設定 EMAIL_USER / EMAIL_PASS / EMAIL_TO, 跳過 Send Email")
        print(f"  EMAIL_USER={'有' if user else '無'}, EMAIL_PASS={'有' if pwd else '無'}, EMAIL_TO={to_raw}")
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    hsi = None
    for r in results:
        if r.get('ticker') == '^HSI':
            hsi = r
            break
    if hsi is None and results:
        hsi = results[0]

    # 主旨: 恒指預測 + Top 買入
    buy_names = []
    for r in results:
        lv = calc_levels(r['last_price'], r['pred_1d_pct'], r['vol'], r['confidence'])
        if lv['signal'] == '買入':
            buy_names.append(r['name'])
    buy_str = ", ".join(buy_names[:3]) if buy_names else "無買入, 觀望"
    subject = f"📈 HSI預測 {today} 恒指 {hsi['pred_1d_pct']:+.2f}% 信心 {hsi['confidence']*100:.0f}% | 買入: {buy_str}"

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = ", ".join(to_list)
    msg['Subject'] = subject

    html = build_email_html(results, backtest_summary=backtest_summary, fred_sig=fred_sig, fred_text=fred_text)
    msg.attach(MIMEText(html, 'html', 'utf-8'))

    # 附加圖片
    for img_path in ["output/hsi_forecast.png", "output/stock_ranking.png"]:
        try:
            with open(img_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=img_path.split('/')[-1])
                msg.attach(img)
        except Exception:
            pass

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(user, pwd)
            server.send_message(msg, to_addrs=to_list)
        print(f"✅ Email 已 Send 去 {len(to_list)} 個人: {to_list} | 主題: {subject}")
        return True
    except Exception as e:
        print(f"❌ Send Email 失敗: {e}")
        return False
