"""SwingEdge Pro — Firestore Database Manager (replaces SQLite)"""
import os, json
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, firestore as fs


# ── Init ───────────────────────────────────────────────────────────────────────

def _get_db():
    """Return Firestore client, initialising the Firebase app if needed."""
    if not firebase_admin._apps:
        cred_json = os.environ.get('FIREBASE_CREDENTIALS', '')
        if not cred_json:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS env var is not set. "
                "Paste your Firebase service-account JSON as a single-line string."
            )
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    return fs.client()


def init_db():
    """No-op — Firestore creates collections on first write."""
    pass


# ── Predictions ────────────────────────────────────────────────────────────────

def log_predictions(picks):
    db = _get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    logged = 0
    col = db.collection('predictions')

    for p in picks:
        # Deduplicate: one prediction per symbol+type per day
        existing = (
            col.where('logged_date', '==', today)
               .where('symbol',      '==', p['symbol'])
               .where('trade_type',  '==', p['trade_type'])
               .limit(1).get()
        )
        if list(existing):
            continue

        col.add({
            'logged_date':          today,
            'symbol':               p['symbol'],
            'trade_type':           p['trade_type'],
            'entry_price':          p['entry_price'],
            'entry_trigger':        p['entry_trigger'],
            'target1':              p['target1'],
            'target2':              p['target2'],
            'target3':              p.get('target3'),
            'stop_loss':            p['stop_loss'],
            'score':                p['score'],
            'atr':                  p['atr'],
            'rsi_at_entry':         p.get('rsi'),
            'macd_at_entry':        p.get('macd_hist'),
            'vol_at_entry':         p.get('vol_ratio'),
            'adx_at_entry':         p.get('adx'),
            'ema_align_at_entry':   p.get('ema_align'),
            'vwap_above_at_entry':  p.get('vwap_above'),
            'bb_b_at_entry':        p.get('bb_b'),
            'ret20_at_entry':       p.get('ret20'),
            'win_prob_at_entry':    p.get('win_prob'),
            'status':               'OPEN',
            'outcome_price':        None,
            'outcome_date':         None,
            'profit_loss_pct':      None,
            'investment_100k':      None,
            'created_at':           datetime.now().isoformat(),
        })
        logged += 1

    return logged


def update_open_predictions(current_prices):
    db = _get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    swing_cutoff  = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    open_docs = db.collection('predictions').where('status', '==', 'OPEN').get()
    updated = 0

    for doc in open_docs:
        row   = doc.to_dict()
        sym   = row['symbol']
        price = current_prices.get(sym)

        if price is None:
            continue

        entry = row['entry_price']
        t1    = row['target1']
        t2    = row['target2']
        sl    = row['stop_loss']
        tt    = row.get('trade_type', 'SWING')
        ld    = row.get('logged_date', today)

        # Check if expired
        if tt == 'INTRADAY' and ld < today:
            doc.reference.update({'status': 'EXPIRED'})
            continue
        if tt == 'SWING' and ld < swing_cutoff:
            doc.reference.update({'status': 'EXPIRED'})
            continue

        # Check outcome
        status = None
        if   price >= t2: status = 'WIN_T2'
        elif price >= t1: status = 'WIN_T1'
        elif price <= sl: status = 'LOSS'

        if status:
            pnl = round((price - entry) / entry * 100, 2)
            inv = round(100000 + 100000 * pnl / 100, 2)
            doc.reference.update({
                'status':          status,
                'outcome_price':   price,
                'outcome_date':    today,
                'profit_loss_pct': pnl,
                'investment_100k': inv,
            })
            updated += 1

    return updated


def get_all_predictions():
    db = _get_db()
    docs = (
        db.collection('predictions')
          .order_by('created_at', direction=fs.Query.DESCENDING)
          .get()
    )
    return [{'id': doc.id, **doc.to_dict()} for doc in docs]


# ── Self Notes ─────────────────────────────────────────────────────────────────

def save_self_notes(notes):
    db = _get_db()
    col = db.collection('self_notes')
    # Wipe previous notes
    for doc in col.get():
        doc.reference.delete()
    for n in notes:
        col.add({
            'icon':       n.get('icon', '📌'),
            'type':       n.get('type', 'info'),
            'note':       n.get('note', ''),
            'created_at': n.get('updated', datetime.now().isoformat()),
        })


# ── Audit ──────────────────────────────────────────────────────────────────────

def get_audit_summary():
    db     = _get_db()
    preds  = get_all_predictions()
    notes  = [
        {'id': d.id, **d.to_dict()}
        for d in db.collection('self_notes')
                   .order_by('created_at', direction=fs.Query.DESCENDING)
                   .get()
    ]
    stats = {}
    for tt in ['SWING', 'INTRADAY']:
        sub    = [p for p in preds if p.get('trade_type') == tt]
        closed = [p for p in sub   if p.get('status') in ('WIN_T1', 'WIN_T2', 'LOSS')]
        wins   = [p for p in closed if p.get('status', '').startswith('WIN')]
        losses = [p for p in closed if p.get('status') == 'LOSS']
        accuracy  = round(len(wins)   / len(closed) * 100, 1) if closed else 0
        avg_win   = round(sum(p.get('profit_loss_pct', 0) or 0 for p in wins)   / len(wins),   2) if wins   else 0
        avg_loss  = round(sum(p.get('profit_loss_pct', 0) or 0 for p in losses) / len(losses), 2) if losses else 0
        inv_total = sum(p.get('investment_100k', 100000) or 100000 for p in closed)
        net_roi   = round((inv_total / (len(closed) * 100000) - 1) * 100, 2) if closed else 0
        stats[tt] = {
            'total':        len(sub),
            'open':         len([p for p in sub if p.get('status') == 'OPEN']),
            'closed':       len(closed),
            'wins':         len(wins),
            'losses':       len(losses),
            'expired':      len([p for p in sub if p.get('status') == 'EXPIRED']),
            'accuracy_pct': accuracy,
            'avg_win_pct':  avg_win,
            'avg_loss_pct': avg_loss,
            'net_roi_pct':  net_roi,
        }
    return {'predictions': preds, 'stats': stats, 'self_notes': notes}
