"""SwingEdge Pro v5.0 — Full Flask Backend"""
from flask import Flask, render_template, jsonify, request
import yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timedelta
import pytz, json, os, warnings, traceback

import db as DB
import ml as ML_MOD
import social as SOCIAL

warnings.filterwarnings('ignore')
import os as _os
app = Flask(__name__, template_folder=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "templates"))
app.secret_key = 'swingEdgePro2025v5'
DB.init_db()
ML = ML_MOD.ML

IST = pytz.timezone('Asia/Kolkata')

NSE_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BAJFINANCE","KOTAKBANK","ASIANPAINT","WIPRO","AXISBANK","MARUTI","SUNPHARMA",
    "TITAN","BAJAJFINSV","NESTLEIND","ULTRACEMCO","TECHM","POWERGRID","NTPC",
    "HCLTECH","ADANIENT","ADANIPORTS","COALINDIA","JSWSTEEL","TATASTEEL",
    "HINDALCO","ONGC","CIPLA","DRREDDY","DIVISLAB","APOLLOHOSP","BPCL","GRASIM",
    "SHREECEM","BRITANNIA","HEROMOTOCO","EICHERMOT","INDUSINDBK","M&M","LT",
    "BHARTIARTL","HDFCLIFE","SBILIFE","UPL","DABUR","PIDILITIND","BAJAJ-AUTO",
    "TATACONSUM","NYKAA","DMART","IRCTC","PAYTM","PERSISTENT","MPHASIS"
]

def market_status():
    now_ist = datetime.now(IST)
    wd = now_ist.weekday()  # 0=Mon, 6=Sun
    h, m = now_ist.hour, now_ist.minute
    mins = h * 60 + m
    if wd >= 5:
        return 'closed', 'Weekend — market is closed', now_ist
    if 555 <= mins <= 930:   # 9:15 AM to 3:30 PM
        return 'open', 'Market is OPEN', now_ist
    elif mins < 555:
        opens_in = 555 - mins
        return 'pre', f'Market opens in {opens_in//60}h {opens_in%60}m', now_ist
    else:
        return 'closed', 'Market closed for today', now_ist

def fetch(sym, period='90d', interval='1d'):
    try:
        t = yf.Ticker(f'{sym}.NS')
        df = t.history(period=period, interval=interval)
        if df.empty or len(df) < 5: return None
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        return df
    except: return None

def indicators(df):
    c=df['Close'];h=df['High'];l=df['Low'];v=df['Volume']
    df['EMA9']=c.ewm(span=9).mean();df['EMA20']=c.ewm(span=20).mean();df['EMA50']=c.ewm(span=50).mean()
    d=c.diff();g=d.clip(lower=0).rolling(14).mean();ls=(-d.clip(upper=0)).rolling(14).mean()
    df['RSI']=100-(100/(1+g/(ls+1e-9)))
    e12=c.ewm(span=12).mean();e26=c.ewm(span=26).mean()
    df['MACD']=e12-e26;df['MACD_Sig']=df['MACD'].ewm(span=9).mean();df['MACD_H']=df['MACD']-df['MACD_Sig']
    bm=c.rolling(20).mean();bs=c.rolling(20).std()
    df['BB_U']=bm+2*bs;df['BB_L']=bm-2*bs;df['BB_B']=(c-df['BB_L'])/(df['BB_U']-df['BB_L']+1e-9)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    df['ATR']=tr.rolling(14).mean()
    pd_=h.diff().clip(lower=0);md=(-l.diff()).clip(lower=0);tr14=tr.rolling(14).mean()
    df['ADX']=(abs(pd_.rolling(14).mean()-md.rolling(14).mean())/(tr14+1e-9)*100).rolling(14).mean()
    df['VolR']=v/v.rolling(20).mean()
    df['VWAP']=(c*v).rolling(20).sum()/v.rolling(20).sum()
    df['Ret20']=c.pct_change(20)*100;df['Ret1']=c.pct_change(1)*100
    return df

def score_stock(df):
    if df is None or len(df)<30: return 0,{}
    l=df.iloc[-1];p=df.iloc[-2];s=0;sig={}
    r=float(l['RSI'])
    if 50<r<70:   s+=15;sig['Momentum meter']=f"Healthy buying pressure ({r:.0f}/100)"
    elif 45<=r<=50: s+=8;sig['Momentum meter']=f"Neutral — neither buyers nor sellers dominating ({r:.0f})"
    elif r>=70:   s+=5;sig['Momentum meter']=f"⚠️ Getting overheated — too many buyers ({r:.0f})"
    else:         sig['Momentum meter']=f"Weak — more sellers than buyers ({r:.0f})"
    if l['MACD']>l['MACD_Sig'] and l['MACD_H']>0:
        s+=15;sig['Trend direction']="✅ Speed picking up — more buyers coming in"
    elif l['MACD']>l['MACD_Sig']:
        s+=8;sig['Trend direction']="Rising trend"
    else: sig['Trend direction']="⬇️ Slowing down"
    pr=float(l['Close'])
    if pr>l['EMA9']>l['EMA20']>l['EMA50']:
        s+=15;sig['Price vs averages']="✅ Above all recent averages — strong uptrend"
    elif pr>l['EMA20']>l['EMA50']:
        s+=10;sig['Price vs averages']="Above key averages — healthy"
    elif pr>l['EMA50']:
        s+=5;sig['Price vs averages']="Above 50-day average"
    else: sig['Price vs averages']="⬇️ Below its own averages — weak"
    vr=float(l['VolR']);up=float(l['Close'])>float(p['Close'])
    if vr>1.5 and up:   s+=15;sig['Trading activity']=f"✅ {vr:.1f}× more trading than usual — breakout signal!"
    elif vr>1.2 and up: s+=10;sig['Trading activity']=f"Active — {vr:.1f}× above normal"
    elif vr>0.8:        s+=5; sig['Trading activity']=f"Normal activity ({vr:.1f}×)"
    else:               sig['Trading activity']=f"⬇️ Quiet day ({vr:.1f}× normal)"
    bb=float(l['BB_B'])
    if 0.5<bb<0.8:   s+=10;sig['Price position']="In the upper half — positive momentum"
    elif 0.3<=bb<=0.5: s+=6;sig['Price position']="In the middle — neutral"
    elif bb>0.8:     s+=3;sig['Price position']="Near the top of its range"
    else:            sig['Price position']="Near the bottom of its range"
    adx=float(l['ADX'])
    if adx>25:   s+=10;sig['Trend strength']=f"✅ Strong clear trend ({adx:.0f}/100)"
    elif adx>20: s+=6; sig['Trend strength']=f"Building trend ({adx:.0f}/100)"
    else:        sig['Trend strength']=f"Drifting — no clear direction ({adx:.0f}/100)"
    ret=float(l['Ret20'])
    if ret>8:   s+=10;sig['20-day growth']=f"✅ Up {ret:.1f}% in last 20 days — strong momentum"
    elif ret>3: s+=7; sig['20-day growth']=f"Up {ret:.1f}% in 20 days"
    elif ret>0: s+=4; sig['20-day growth']=f"Up {ret:.1f}% in 20 days — mild"
    else:       sig['20-day growth']=f"⬇️ Down {ret:.1f}% in 20 days"
    if float(l['Close'])>float(l['VWAP']):
        s+=10;sig['vs Today\'s avg price']=f"✅ Trading above today's average (₹{l['VWAP']:.0f}) — buyers in control"
    else:
        sig['vs Today\'s avg price']=f"⬇️ Below today's average — sellers in control"
    return s,sig

def classify(r):
    sw=0;intra=0
    if r['ADX']>25: sw+=2
    if r['Return_20d']>5: sw+=2
    if r['RSI']<65: sw+=1
    if r['Vol_Ratio']>1.5: intra+=3
    if 60<r['RSI']<75: intra+=1
    if r['MACD_Hist']>0: intra+=1;sw+=1
    return "SWING" if sw>=intra else "INTRADAY"

def plain_english_levels(r, mode):
    p=r['Price'];atr=r['ATR']
    inv_base=100000
    if mode=="SWING":
        entry=round(max(r['EMA9'],r['EMA20'])*1.002,2)
        t1=round(p+2.0*atr,2);t2=round(p+3.5*atr,2);t3=round(p+5.0*atr,2)
        sl=round(p-1.5*atr,2);sl_pct=round((p-sl)/p*100,2)
        t1_pct=round((t1-p)/p*100,2);t2_pct=round((t2-p)/p*100,2)
        t1_inv=round(inv_base*(1+t1_pct/100));t2_inv=round(inv_base*(1+t2_pct/100))
        sl_inv=round(inv_base*(1-sl_pct/100))
        hold="3 to 10 days" if r['ADX']>25 else "5 to 15 days"
        ent=[
            f"Wait for price to reach ₹{entry} — that's when the setup is confirmed. Right now it's ₹{p}.",
            f"The buying pressure score should be between 50-65. Currently it's {r['RSI']:.0f} — {'good ✅' if 50<r['RSI']<65 else 'be careful ⚠️'}.",
            f"The trend speed should be increasing (currently {r['MACD_Hist']:+.2f} — {'positive ✅' if r['MACD_Hist']>0 else 'negative, wait ⚠️'}).",
            f"Trading volume should be at least 20% above normal. Today it's {r['Vol_Ratio']:.1f}× — {'strong ✅' if r['Vol_Ratio']>1.2 else 'low, be cautious ⚠️'}.",
            f"Best entry time: when today's candle closes above ₹{entry}, or on a 15-minute chart breakout.",
            f"Only enter when the overall market (Nifty 50 index) is also going up that day."
        ]
        ext=[
            f"🎯 First target — ₹{t1} (+{t1_pct}% profit): When price reaches ₹{t1}, sell 40% of your shares. On ₹1 lakh → your money becomes ₹{t1_inv:,} (profit of ₹{t1_inv-inv_base:,}).",
            f"🎯 Second target — ₹{t2} (+{t2_pct}% profit): Sell another 40% here. On ₹1 lakh → money becomes ₹{t2_inv:,}.",
            f"🎯 Final 20%: Let the last bit ride with a moving safety net. Exit if price drops back below ₹{t3} — that's your trailing stop.",
            f"🛑 Safety net (Stop Loss) — ₹{sl}: If price falls to ₹{sl}, exit immediately. This limits your loss to {sl_pct}%. On ₹1 lakh → worst case you'd have ₹{sl_inv:,} left (loss of ₹{inv_base-sl_inv:,}).",
            f"🛑 Time limit: If nothing happens in {hold}, exit and look for better opportunities. The setup has failed.",
            f"🛑 Exit signal: If buying pressure drops below 45 AND price falls below its 20-day average → sell everything.",
            f"🛑 Trend reversal: If the trend direction flips negative → sell half your position immediately."
        ]
        return ent,ext,entry,t1,t2,t3,sl
    else:
        entry=round(r['VWAP']*1.001,2)
        t1=round(p+0.5*atr,2);t2=round(p+1.0*atr,2)
        sl=round(p-0.5*atr,2);sl_pct=round((p-sl)/p*100,2)
        t1_pct=round((t1-p)/p*100,2);t2_pct=round((t2-p)/p*100,2)
        t1_inv=round(inv_base*(1+t1_pct/100));t2_inv=round(inv_base*(1+t2_pct/100))
        sl_inv=round(inv_base*(1-sl_pct/100))
        ent=[
            f"After market opens (9:30 AM): Watch the first 15 minutes and don't rush in.",
            f"Enter only if price crosses above ₹{entry} (today's average trading price) with a spike in activity.",
            f"On a 5-minute chart: the 9-day line must be above the 20-day line, and momentum score must be above 55.",
            f"Activity must be 50% above normal. Current level: {r['Vol_Ratio']:.1f}× — {'strong ✅' if r['Vol_Ratio']>1.5 else 'not there yet ⚠️'}.",
            f"Only trade if the broader market (Nifty/Bank Nifty) is green and rising.",
            f"After 1:30 PM, do NOT take any new positions. Max 3 tries per stock per day."
        ]
        ext=[
            f"🎯 First target — ₹{t1} (+{t1_pct}%): Sell half. On ₹1 lakh → ₹{t1_inv:,} (profit ₹{t1_inv-inv_base:,}).",
            f"🎯 Second target — ₹{t2} (+{t2_pct}%): Sell remaining half. On ₹1 lakh → ₹{t2_inv:,}.",
            f"🛑 Safety net — ₹{sl}: If price falls to ₹{sl}, EXIT IMMEDIATELY. Loss limited to {sl_pct}%. On ₹1 lakh → ₹{sl_inv:,} left.",
            f"🛑 MUST CLOSE by 3:15 PM — no matter what. Do not hold overnight.",
            f"🛑 If price drops below today's average on ANY 5-minute candle — exit.",
            f"🛑 First target not reached by 2:00 PM? Close the trade at whatever price you can get.",
            f"🛑 After first target hit → move your safety net up to your entry price so you can't lose money."
        ]
        return ent,ext,entry,t1,t2,None,sl

def run_screener():
    results=[]
    for sym in NSE_STOCKS:
        df=fetch(sym)
        if df is not None and len(df)>=30:
            df=indicators(df)
            sc,sig=score_stock(df)
            l=df.iloc[-1]
            price=round(float(l['Close']),2)
            bb_u=round(float(l['BB_U']),2);bb_l=round(float(l['BB_L']),2)
            bb_b=(price-bb_l)/(bb_u-bb_l+1e-9) if bb_u>bb_l else 0.5
            ema_align=sum([price>float(l['EMA9']),price>float(l['EMA20']),price>float(l['EMA50'])])
            r={
                'Symbol':sym,'Price':price,'Score':round(sc,1),
                'RSI':round(float(l['RSI']),1),'MACD_Hist':round(float(l['MACD_H']),3),
                'Vol_Ratio':round(float(l['VolR']),2),'ATR':round(float(l['ATR']),2),
                'Return_20d':round(float(l['Ret20']),2),'Change_1d':round(float(l['Ret1']),2),
                'EMA9':round(float(l['EMA9']),2),'EMA20':round(float(l['EMA20']),2),
                'EMA50':round(float(l['EMA50']),2),'VWAP':round(float(l['VWAP']),2),
                'ADX':round(float(l['ADX']),1),'BB_Upper':bb_u,'BB_Lower':bb_l,
                'BB_B':round(bb_b,3),'EMA_Align':ema_align,'Signals':sig
            }
            # ML enhancements
            win_prob = ML.predict_win_prob(r)
            enhanced = ML.enhanced_score(sc, r)
            r['WinProb']=win_prob; r['EnhancedScore']=enhanced
            tt=classify(r)
            ent,ext,entry,t1,t2,t3,sl=plain_english_levels(r,tt)
            r.update({'TradeType':tt,'EntryRules':ent,'ExitRules':ext,
                      'EntryTrigger':entry,'Target1':t1,'Target2':t2,'Target3':t3,'StopLoss':sl})
            results.append(r)
    results.sort(key=lambda x:x['EnhancedScore'],reverse=True)
    return results

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/screener')
def api_screener():
    try:
        all_r=run_screener()
        top20=all_r[:20]
        expected=[r for r in all_r if 40<=r['Score']<=72 and r['Return_20d']>0 and r['Vol_Ratio']>0.7]
        expected=sorted(expected,key=lambda x:x['Return_20d']+x['Vol_Ratio']*5,reverse=True)[:20]
        swing=[r for r in top20 if r['TradeType']=='SWING']
        intraday=[r for r in top20 if r['TradeType']=='INTRADAY']
        # ML: cluster stocks
        ML.cluster_stocks(all_r[:30])
        # Log picks
        picks=[]
        for r in top20+expected:
            bb_b=(r['Price']-r['BB_Lower'])/(r['BB_Upper']-r['BB_Lower']+1e-9)
            picks.append({
                'symbol':r['Symbol'],'trade_type':r['TradeType'],
                'entry_price':r['Price'],'entry_trigger':r['EntryTrigger'],
                'target1':r['Target1'],'target2':r['Target2'],
                'target3':r.get('Target3') or r['Target2']*1.05,
                'stop_loss':r['StopLoss'],'score':r['Score'],'atr':r['ATR'],
                'rsi':r['RSI'],'macd_hist':r['MACD_Hist'],'vol_ratio':r['Vol_Ratio'],
                'adx':r['ADX'],'ema_align':r['EMA_Align'],'vwap_above':1 if r['Price']>r['VWAP'] else 0,
                'bb_b':r['BB_B'],'ret20':r['Return_20d'],'win_prob':r['WinProb']
            })
        logged=DB.log_predictions(picks)
        # Update + RL reinforce
        prices={r['Symbol']:r['Price'] for r in all_r}
        updated=DB.update_open_predictions(prices)
        preds=DB.get_all_predictions()
        closed=[p for p in preds if p['status'] in ('WIN_T1','WIN_T2','LOSS')]
        for pred in closed[-5:]:
            won=pred['status'].startswith('WIN')
            ML.reinforce(pred,won)
        # Supervised train
        ML.train_supervised(preds)
        # Generate self notes
        notes=ML.generate_notes(preds)
        DB.save_self_notes(notes)
        return jsonify({
            'status':'ok',
            'generated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S IST'),
            'top20_performing':top20,'top20_expected':expected,
            'swing_trades':swing,'intraday_trades':intraday,
            'new_predictions_logged':logged,'predictions_updated':updated
        })
    except Exception as e:
        return jsonify({'status':'error','message':str(e),'trace':traceback.format_exc()}),500

@app.route('/api/buy_now')
def api_buy_now():
    try:
        status,msg,now_ist=market_status()
        all_r=run_screener()
        swing_top=sorted([r for r in all_r if r['TradeType']=='SWING'],key=lambda x:x['EnhancedScore'],reverse=True)[:10]
        intra_top=sorted([r for r in all_r if r['TradeType']=='INTRADAY'],key=lambda x:x['EnhancedScore'],reverse=True)[:10]
        # For each stock, add plain-English reasoning
        def add_reasoning(r):
            sc=r['EnhancedScore'];wp=r['WinProb']
            if sc>=80: confidence="Very High"
            elif sc>=65: confidence="High"
            elif sc>=50: confidence="Moderate"
            else: confidence="Low"
            reasons=[]
            if r['RSI']>50 and r['RSI']<70: reasons.append(f"Buying pressure is healthy at {r['RSI']:.0f}")
            if r['MACD_Hist']>0: reasons.append("Trend is pointing up")
            if r['Vol_Ratio']>1.3: reasons.append(f"Activity is {r['Vol_Ratio']:.1f}× above normal")
            if r['Return_20d']>5: reasons.append(f"Already up {r['Return_20d']:.1f}% this month")
            if r['ADX']>25: reasons.append("Clear strong trend in place")
            r['confidence']=confidence
            r['why_buy']=reasons
            r['risk_plain']=f"If wrong, stop loss at ₹{r['StopLoss']} limits loss to {round((r['Price']-r['StopLoss'])/r['Price']*100,1)}%"
            r['reward_plain']=f"Target ₹{r['Target1']} = +{round((r['Target1']-r['Price'])/r['Price']*100,1)}% gain"
            return r
        return jsonify({
            'status':'ok','market_status':status,'market_message':msg,
            'time_ist':now_ist.strftime('%H:%M IST, %a %b %d'),
            'section_label': '📈 Best Stocks to Buy RIGHT NOW' if status=='open' else '📅 Best Stocks for NEXT SESSION',
            'swing_picks':[add_reasoning(r) for r in swing_top],
            'intraday_picks':[add_reasoning(r) for r in intra_top],
            'generated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
        })
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}),500

@app.route('/api/chart/<symbol>/<timeframe>')
def api_chart(symbol,timeframe):
    TF={'1m':('1d','1m'),'5m':('5d','5m'),'15m':('5d','15m'),
        '1h':('60d','1h'),'1d':('1y','1d'),'1wk':('5y','1wk'),'1mo':('10y','1mo')}
    try:
        period,interval=TF.get(timeframe,('1y','1d'))
        t=yf.Ticker(f'{symbol.upper()}.NS')
        df=t.history(period=period,interval=interval)
        if df.empty: return jsonify({'error':'No data'}),404
        df.index=df.index.tz_localize(None) if df.index.tz else df.index
        df['EMA9']=df['Close'].ewm(span=9).mean()
        df['EMA20']=df['Close'].ewm(span=20).mean()
        df['EMA50']=df['Close'].ewm(span=50).mean()
        n=min(20,len(df))
        df['VWAP']=(df['Close']*df['Volume']).rolling(n).sum()/df['Volume'].rolling(n).sum()
        return jsonify({
            'symbol':symbol.upper(),'timeframe':timeframe,
            'dates':[str(i) for i in df.index],
            'open':[round(float(x),2) for x in df['Open']],
            'high':[round(float(x),2) for x in df['High']],
            'low':[round(float(x),2) for x in df['Low']],
            'close':[round(float(x),2) for x in df['Close']],
            'volume':[int(x) for x in df['Volume']],
            'ema9':[round(float(x),2) if not pd.isna(x) else None for x in df['EMA9']],
            'ema20':[round(float(x),2) if not pd.isna(x) else None for x in df['EMA20']],
            'ema50':[round(float(x),2) if not pd.isna(x) else None for x in df['EMA50']],
            'vwap':[round(float(x),2) if not pd.isna(x) else None for x in df['VWAP']],
            'note':'Data via Yahoo Finance (15-min delay for NSE)'
        })
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/audit')
def api_audit():
    try:
        data=DB.get_audit_summary()
        notes=ML.generate_notes(data['predictions'])
        DB.save_self_notes(notes)
        data['self_notes']=notes
        return jsonify({'status':'ok',**data})
    except Exception as e: return jsonify({'status':'error','message':str(e)}),500

@app.route('/api/alerts')
def api_alerts():
    try:
        all_r=run_screener();alerts=[]
        for r in all_r[:20]:
            p=r['Price'];entry=r['EntryTrigger'];sl=r['StopLoss'];t1=r['Target1']
            if abs(p-entry)/max(entry,1)*100<=0.8:
                alerts.append({'symbol':r['Symbol'],'price':p,'type':'BUY_ALERT',
                    'trade_type':r['TradeType'],'score':r['EnhancedScore'],'win_prob':r['WinProb'],
                    'msg':f"🟢 {r['Symbol']} is almost at your entry price! Current: ₹{p}, Entry point: ₹{entry}. Win chance: {r['WinProb']}%. Consider entering now."})
            elif abs(p-sl)/max(p,1)*100<=0.8:
                alerts.append({'symbol':r['Symbol'],'price':p,'type':'STOP_ALERT',
                    'trade_type':r['TradeType'],'score':r['EnhancedScore'],'win_prob':r['WinProb'],
                    'msg':f"🔴 {r['Symbol']} is close to your safety net! Current: ₹{p}, Safety net: ₹{sl}. If you're holding this, consider exiting to protect your money."})
            elif abs(p-t1)/max(p,1)*100<=0.5:
                alerts.append({'symbol':r['Symbol'],'price':p,'type':'TARGET_ALERT',
                    'trade_type':r['TradeType'],'score':r['EnhancedScore'],'win_prob':r['WinProb'],
                    'msg':f"🎯 {r['Symbol']} is approaching your first target! Current: ₹{p}, Target: ₹{t1}. Consider booking some profit."})
        return jsonify({'status':'ok','alerts':alerts,'checked_at':datetime.now().strftime('%H:%M:%S')})
    except Exception as e: return jsonify({'status':'error','message':str(e),'alerts':[]}),500

@app.route('/api/social')
def api_social():
    try:
        data=SOCIAL.build_leaderboard()
        return jsonify({'status':'ok',**data})
    except Exception as e: return jsonify({'status':'error','message':str(e),'buy':[],'sell':[]}),500

if __name__=='__main__':
    print('\n🚀 SwingEdge Pro v5.0 on http://localhost:5050\n')
    app.run(debug=False,port=5050,host='0.0.0.0')
