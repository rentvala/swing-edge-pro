"""SwingEdge Pro — Database Manager v4 (with ML state tables)"""
import sqlite3, os, json
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'predictions.db')

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn(); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_date      TEXT NOT NULL, symbol TEXT NOT NULL,
        trade_type       TEXT NOT NULL, entry_price REAL NOT NULL,
        entry_trigger    REAL NOT NULL, target1 REAL NOT NULL,
        target2          REAL NOT NULL, target3 REAL, stop_loss REAL NOT NULL,
        score            REAL NOT NULL, atr REAL NOT NULL,
        rsi_at_entry     REAL, macd_at_entry REAL, vol_at_entry REAL,
        adx_at_entry     REAL, ema_align_at_entry REAL,
        vwap_above_at_entry INTEGER, bb_b_at_entry REAL, ret20_at_entry REAL,
        status           TEXT DEFAULT 'OPEN', outcome_price REAL,
        outcome_date     TEXT, profit_loss_pct REAL, investment_100k REAL,
        win_prob_at_entry REAL,
        created_at       TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS self_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        icon TEXT, type TEXT, note TEXT, created_at TEXT
    )''')
    conn.commit(); conn.close()

def log_predictions(picks):
    conn = get_conn(); c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d'); logged = 0
    for p in picks:
        exists = c.execute(
            'SELECT id FROM predictions WHERE logged_date=? AND symbol=? AND trade_type=?',
            (today, p['symbol'], p['trade_type'])).fetchone()
        if not exists:
            c.execute('''INSERT INTO predictions
                (logged_date,symbol,trade_type,entry_price,entry_trigger,target1,target2,target3,
                 stop_loss,score,atr,rsi_at_entry,macd_at_entry,vol_at_entry,adx_at_entry,
                 ema_align_at_entry,vwap_above_at_entry,bb_b_at_entry,ret20_at_entry,win_prob_at_entry)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (today,p['symbol'],p['trade_type'],p['entry_price'],p['entry_trigger'],
                 p['target1'],p['target2'],p.get('target3'),p['stop_loss'],p['score'],p['atr'],
                 p.get('rsi'),p.get('macd_hist'),p.get('vol_ratio'),p.get('adx'),
                 p.get('ema_align'),p.get('vwap_above'),p.get('bb_b'),p.get('ret20'),
                 p.get('win_prob')))
            logged += 1
    conn.commit(); conn.close(); return logged

def update_open_predictions(current_prices):
    conn = get_conn(); c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    rows = c.execute("SELECT * FROM predictions WHERE status='OPEN'").fetchall()
    updated = 0
    for row in rows:
        sym = row['symbol']
        if sym not in current_prices: continue
        price = current_prices[sym]
        entry = row['entry_price']; t1 = row['target1']; t2 = row['target2']; sl = row['stop_loss']
        status = None
        if price >= t2:   status = 'WIN_T2'
        elif price >= t1: status = 'WIN_T1'
        elif price <= sl: status = 'LOSS'
        if status:
            pnl = round((price - entry) / entry * 100, 2)
            inv = round(100000 + 100000 * pnl / 100, 2)
            c.execute('''UPDATE predictions SET status=?,outcome_price=?,outcome_date=?,
                         profit_loss_pct=?,investment_100k=? WHERE id=?''',
                      (status, price, today, pnl, inv, row['id']))
            updated += 1
    cutoff_intra = today
    cutoff_swing = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    c.execute("UPDATE predictions SET status='EXPIRED' WHERE status='OPEN' AND trade_type='INTRADAY' AND logged_date < ?", (cutoff_intra,))
    c.execute("UPDATE predictions SET status='EXPIRED' WHERE status='OPEN' AND trade_type='SWING' AND logged_date < ?", (cutoff_swing,))
    conn.commit(); conn.close(); return updated

def get_all_predictions():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_self_notes(notes):
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM self_notes")
    for n in notes:
        c.execute("INSERT INTO self_notes(icon,type,note,created_at) VALUES(?,?,?,?)",
                  (n.get('icon','📌'), n.get('type','info'), n.get('note',''), n.get('updated','')))
    conn.commit(); conn.close()

def get_audit_summary():
    preds = get_all_predictions()
    conn = get_conn()
    notes = [dict(r) for r in conn.execute("SELECT * FROM self_notes ORDER BY id DESC").fetchall()]
    conn.close()
    stats = {}
    for tt in ['SWING','INTRADAY']:
        sub    = [p for p in preds if p['trade_type']==tt]
        closed = [p for p in sub if p['status'] in ('WIN_T1','WIN_T2','LOSS')]
        wins   = [p for p in closed if p['status'].startswith('WIN')]
        losses = [p for p in closed if p['status']=='LOSS']
        accuracy = round(len(wins)/len(closed)*100,1) if closed else 0
        avg_win  = round(sum(p.get('profit_loss_pct',0) or 0 for p in wins)/len(wins),2) if wins else 0
        avg_loss = round(sum(p.get('profit_loss_pct',0) or 0 for p in losses)/len(losses),2) if losses else 0
        inv_total = sum(p.get('investment_100k',100000) or 100000 for p in closed)
        net_roi = round((inv_total/(len(closed)*100000)-1)*100,2) if closed else 0
        stats[tt] = {
            'total':len(sub),'open':len([p for p in sub if p['status']=='OPEN']),
            'closed':len(closed),'wins':len(wins),'losses':len(losses),
            'expired':len([p for p in sub if p['status']=='EXPIRED']),
            'accuracy_pct':accuracy,'avg_win_pct':avg_win,'avg_loss_pct':avg_loss,
            'net_roi_pct':net_roi
        }
    return {'predictions':preds,'stats':stats,'self_notes':notes}
