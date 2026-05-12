"""
SwingEdge Pro — ML Engine
Supervised + Unsupervised + Reinforcement Learning
State persists to Firestore. Model pkl lives in /tmp for the session duration.
"""
import os, json, pickle
import numpy as np
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

MODEL_PATH = '/tmp/ml_model.pkl'   # ephemeral per invocation — rebuilt from Firestore data

FEATURE_NAMES = ['rsi', 'macd_hist', 'vol_ratio', 'adx', 'bb_b',
                 'ret20', 'ema_align', 'vwap_above', 'score']

DEFAULT_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.0, 'EMA': 1.0,
    'Volume': 1.0, 'BB': 1.0, 'ADX': 1.0,
    'Momentum': 1.0, 'VWAP': 1.0,
}


# ── Firestore helpers (imported lazily to avoid circular import) ───────────────

def _firestore_load():
    """Return ml/state dict from Firestore, or {} on any error."""
    try:
        import firebase_admin
        from firebase_admin import firestore as _fs
        if not firebase_admin._apps:
            return {}
        doc = _fs.client().collection('ml').document('state').get()
        return doc.to_dict() if doc.exists else {}
    except Exception:
        return {}


def _firestore_save(data: dict):
    """Write data to ml/state in Firestore. Silent on failure."""
    try:
        import firebase_admin
        from firebase_admin import firestore as _fs
        if not firebase_admin._apps:
            return
        _fs.client().collection('ml').document('state').set(data)
    except Exception:
        pass


# ── ML Class ──────────────────────────────────────────────────────────────────

class SwingEdgeML:
    def __init__(self):
        self.weights      = DEFAULT_WEIGHTS.copy()
        self.model        = None
        self.scaler       = None
        self.notes        = []
        self.accuracy     = {}
        self.cluster_info = {}
        self._load_state()

    # ── STATE PERSISTENCE ─────────────────────────────────────────────────────

    def _load_state(self):
        # 1. Load lightweight state (weights, notes, accuracy) from Firestore
        s = _firestore_load()
        if s:
            self.weights      = s.get('weights',      DEFAULT_WEIGHTS.copy())
            self.notes        = s.get('notes',        [])
            self.accuracy     = s.get('accuracy',     {})
            self.cluster_info = s.get('cluster_info', {})

        # 2. Load sklearn model from /tmp if it was saved earlier this invocation
        if SKLEARN_OK and os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    saved = pickle.load(f)
                self.model  = saved.get('model')
                self.scaler = saved.get('scaler')
            except Exception:
                pass

    def _save_state(self):
        # 1. Persist state to Firestore
        _firestore_save({
            'weights':      self.weights,
            'notes':        self.notes,
            'accuracy':     self.accuracy,
            'cluster_info': self.cluster_info,
            'saved_at':     datetime.now().isoformat(),
        })

        # 2. Cache model in /tmp for the rest of this invocation
        if SKLEARN_OK and self.model:
            try:
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            except Exception:
                pass

    # ── FEATURE EXTRACTION ────────────────────────────────────────────────────

    def extract_features(self, r):
        price     = r.get('Price', 1)
        bb_u      = r.get('BB_Upper', price * 1.02)
        bb_l      = r.get('BB_Lower', price * 0.98)
        bb_b      = (price - bb_l) / (bb_u - bb_l + 1e-9)
        ema_align = sum([
            1 if price > r.get('EMA9',  price) else 0,
            1 if price > r.get('EMA20', price) else 0,
            1 if price > r.get('EMA50', price) else 0,
        ])
        return [
            float(r.get('RSI',       50)),
            float(r.get('MACD_Hist',  0)),
            float(r.get('Vol_Ratio',  1)),
            float(r.get('ADX',       20)),
            float(bb_b),
            float(r.get('Return_20d', 0)),
            float(ema_align),
            1.0 if price > r.get('VWAP', price) else 0.0,
            float(r.get('Score',     50)) / 100.0,
        ]

    # ── SUPERVISED LEARNING ───────────────────────────────────────────────────

    def train_supervised(self, predictions):
        if not SKLEARN_OK:
            return 'scikit-learn not installed'
        closed = [p for p in predictions if p.get('status') in ('WIN_T1', 'WIN_T2', 'LOSS')]
        if len(closed) < 8:
            return f'Need 8+ closed predictions to train (have {len(closed)})'

        X, y = [], []
        for p in closed:
            X.append([
                p.get('rsi_at_entry',       p.get('score', 50) * 0.6),
                p.get('macd_at_entry',       0.5),
                p.get('vol_at_entry',        1.0),
                p.get('adx_at_entry',       22.0),
                p.get('bb_b_at_entry',       0.5),
                p.get('ret20_at_entry',      3.0),
                p.get('ema_align_at_entry',  2.0),
                p.get('vwap_above_at_entry', 1.0),
                float(p.get('score', 60)) / 100.0,
            ])
            y.append(1 if p['status'].startswith('WIN') else 0)

        X, y = np.array(X), np.array(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        if len(closed) >= 20:
            self.model = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        self.model.fit(Xs, y)

        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            weight_keys = ['RSI', 'MACD', 'Volume', 'ADX', 'BB', 'Momentum', 'EMA', 'VWAP', 'RSI']
            for i, key in enumerate(FEATURE_NAMES):
                if i < len(weight_keys):
                    wk = weight_keys[i]
                    if wk in self.weights:
                        self.weights[wk] = round(0.5 + fi[i] * 5.0, 3)

        accuracy = float(np.mean(np.array(y) == self.model.predict(Xs)))
        self.accuracy['supervised'] = round(accuracy * 100, 1)
        self._save_state()
        return f'Trained on {len(closed)} predictions — training accuracy {accuracy*100:.0f}%'

    # ── UNSUPERVISED LEARNING ─────────────────────────────────────────────────

    def cluster_stocks(self, stocks, n_clusters=3):
        if not SKLEARN_OK or len(stocks) < n_clusters * 2:
            return {}
        X      = np.array([self.extract_features(s) for s in stocks])
        km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        clusters = {}
        for s, lbl in zip(stocks, labels):
            clusters.setdefault(int(lbl), []).append(s['Symbol'])
        descriptions = {}
        centers = km.cluster_centers_
        for g, syms in clusters.items():
            c = centers[g]
            avg_rsi, avg_vol, avg_ema = c[0], c[2], c[6]
            if avg_ema >= 2.5 and avg_vol > 1.3:
                desc = '🚀 Momentum Leaders — strong uptrend, high activity'
            elif avg_rsi < 55 and avg_vol < 1.0:
                desc = '😴 Consolidating — taking a breather, watch for breakout'
            else:
                desc = '📈 Steady Climbers — moderate trend, building momentum'
            descriptions[g] = {'name': desc, 'stocks': syms}
        self.cluster_info = {str(k): v for k, v in descriptions.items()}
        self._save_state()
        return self.cluster_info

    # ── REINFORCEMENT LEARNING ────────────────────────────────────────────────

    def reinforce(self, prediction, won):
        lr   = 0.08
        rsi  = prediction.get('score', 60) * 0.6
        vol  = prediction.get('vol_at_entry',  1.0)
        macd = prediction.get('macd_at_entry', 0)
        adx  = prediction.get('adx_at_entry',  20)
        sign = 1.0 if won else -1.0

        def _clamp(v): return round(max(0.3, min(2.0, v)), 3)

        if 50 < rsi < 65:
            self.weights['RSI']    = _clamp(self.weights['RSI']    + sign * lr)
        elif rsi > 70:
            self.weights['RSI']    = _clamp(self.weights['RSI']    - lr * 0.5)
        if vol > 1.5:
            self.weights['Volume'] = _clamp(self.weights['Volume'] + sign * lr * 0.8)
        if macd > 0:
            self.weights['MACD']   = _clamp(self.weights['MACD']   + sign * lr * 0.7)
        if adx > 25:
            self.weights['ADX']    = _clamp(self.weights['ADX']    + sign * lr * 0.6)
        self._save_state()

    # ── WIN PROBABILITY ───────────────────────────────────────────────────────

    def predict_win_prob(self, stock_data):
        feats = self.extract_features(stock_data)
        if self.model and self.scaler and SKLEARN_OK:
            try:
                Xs   = self.scaler.transform([feats])
                prob = float(self.model.predict_proba(Xs)[0][1])
                return round(prob * 100, 1)
            except Exception:
                pass
        # Rule-based fallback
        rsi, macd, vol, adx, bb_b, ret20, ema, vwap, sc = feats
        p = 50.0
        if 50 < rsi < 65:  p += 10
        elif rsi >= 70:    p -= 8
        elif rsi < 40:     p -= 12
        if macd > 0:       p += 8
        if vol > 1.5:      p += 8
        elif vol < 0.7:    p -= 6
        if adx > 25:       p += 7
        if ema == 3:       p += 10
        elif ema == 2:     p += 5
        if vwap:           p += 5
        if ret20 > 8:      p += 6
        p += sc * 20
        return round(min(95, max(5, p)), 1)

    # ── ENHANCED SCORE ────────────────────────────────────────────────────────

    def enhanced_score(self, raw_score, stock_data):
        feats = self.extract_features(stock_data)
        rsi, macd, vol, adx, bb_b, ret20, ema, vwap, sc = feats
        w   = self.weights
        adj = 0
        if 50 < rsi < 65:  adj += 2 * w['RSI']
        elif rsi >= 70:    adj -= 3 * w['RSI']
        if macd > 0:       adj += 2 * w['MACD']
        if vol > 1.5:      adj += 2 * w['Volume']
        if adx > 25:       adj += 1.5 * w['ADX']
        if ema == 3:       adj += 2 * w['EMA']
        if vwap:           adj += 1 * w['VWAP']
        return round(min(100, max(0, raw_score + adj)), 1)

    # ── SELF NOTES ────────────────────────────────────────────────────────────

    def generate_notes(self, predictions):
        notes  = []
        ts     = datetime.now().strftime('%b %d, %Y %H:%M')
        closed = [p for p in predictions if p.get('status') in ('WIN_T1', 'WIN_T2', 'LOSS')]
        wins   = [p for p in closed if p.get('status', '').startswith('WIN')]
        losses = [p for p in closed if p.get('status') == 'LOSS']
        total  = len(closed)

        if total == 0:
            notes.append({
                'icon': '🌱', 'type': 'info', 'updated': ts,
                'note': "I'm brand new! Run the screener a few times over several days and I'll start learning from results.",
            })
            return notes

        acc   = len(wins) / total * 100
        emoji = '🏆' if acc >= 65 else '📊' if acc >= 50 else '⚠️'
        notes.append({
            'icon': emoji, 'type': 'accuracy', 'updated': ts,
            'note': (f"Out of my last {total} predictions, {len(wins)} correct and {len(losses)} wrong. "
                     f"That's a {acc:.0f}% success rate. "
                     + ("Doing well! " if acc >= 65 else "Room to improve. " if acc >= 50 else "Struggling — market may be unusual. ")),
        })

        swing_wins = [p for p in wins   if p.get('trade_type') == 'SWING']
        intra_wins = [p for p in wins   if p.get('trade_type') == 'INTRADAY']
        swing_all  = [p for p in closed if p.get('trade_type') == 'SWING']
        intra_all  = [p for p in closed if p.get('trade_type') == 'INTRADAY']
        if swing_all or intra_all:
            notes.append({
                'icon': '📈', 'type': 'pattern', 'updated': ts,
                'note': (f"Swing wins: {len(swing_wins)}/{len(swing_all)}. "
                         f"Intraday wins: {len(intra_wins)}/{len(intra_all)}. "
                         + ("Better at swing! " if len(swing_wins) > len(intra_wins) else
                            "Better at intraday! " if len(intra_wins) > len(swing_wins) else
                            "Equally good at both.")),
            })

        changed = {k: v for k, v in self.weights.items() if abs(v - 1.0) > 0.05}
        if changed:
            name_map = {
                'RSI': 'Momentum meter', 'MACD': 'Trend direction',
                'Volume': 'Trading activity', 'ADX': 'Trend strength',
                'EMA': 'Moving averages', 'BB': 'Price position',
                'Momentum': 'Price growth', 'VWAP': "Today's avg price",
            }
            up   = [name_map.get(k, k) for k, v in changed.items() if v > 1.0]
            down = [name_map.get(k, k) for k, v in changed.items() if v < 1.0]
            parts = []
            if up:   parts.append('More weight on: ' + ', '.join(up))
            if down: parts.append('Less weight on: ' + ', '.join(down))
            notes.append({
                'icon': '⚙️', 'type': 'weights', 'updated': ts,
                'note': 'Updated internal scoring. ' + ' | '.join(parts) + '.',
            })

        if wins:
            avg_win = np.mean([p.get('profit_loss_pct', 0) for p in wins])
            notes.append({
                'icon': '💰', 'type': 'money', 'updated': ts,
                'note': (f"Avg profit on correct calls: +{avg_win:.1f}%. "
                         f"On ₹1L that's ₹{avg_win * 1000:.0f} per winning trade."),
            })
        if losses:
            avg_loss = abs(np.mean([p.get('profit_loss_pct', 0) for p in losses]))
            notes.append({
                'icon': '🛡️', 'type': 'risk', 'updated': ts,
                'note': (f"When wrong, stop loss limits avg damage to -{avg_loss:.1f}%. "
                         f"On ₹1L that's ₹{avg_loss * 1000:.0f} average loss."),
            })

        for g, info in self.cluster_info.items():
            syms = info.get('stocks', [])[:5]
            if syms:
                notes.append({
                    'icon': '🔬', 'type': 'cluster', 'updated': ts,
                    'note': (f"Pattern Group — {info.get('name', 'Group')}: "
                             f"{', '.join(syms)} show similar price behaviour."),
                })

        if total >= 20:
            notes.append({'icon': '🤖', 'type': 'model', 'updated': ts,
                          'note': f'Gradient Boosting trained on {total} predictions.'})
        elif total >= 8:
            notes.append({'icon': '🤖', 'type': 'model', 'updated': ts,
                          'note': f'Random Forest trained on {total}/20 predictions. Upgrading soon.'})

        self.notes = notes
        self._save_state()
        return notes


ML = SwingEdgeML()
