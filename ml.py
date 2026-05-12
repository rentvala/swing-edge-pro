"""
SwingEdge Pro — ML Engine
Supervised + Unsupervised + Reinforcement Learning
Learns from past prediction outcomes, generates plain-English self notes.
"""
import os, json, pickle, numpy as np
from datetime import datetime, timedelta

# Try sklearn imports gracefully
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.model_selection import cross_val_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'ml_model.pkl')
STATE_PATH = os.path.join(DATA_DIR, 'ml_state.json')

FEATURE_NAMES = ['rsi','macd_hist','vol_ratio','adx','bb_b','ret20','ema_align','vwap_above','score']

# ── Default indicator weights (adjusted by RL) ─────────────────────────────────
DEFAULT_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.0, 'EMA': 1.0,
    'Volume': 1.0, 'BB': 1.0, 'ADX': 1.0,
    'Momentum': 1.0, 'VWAP': 1.0
}

class SwingEdgeML:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.weights     = DEFAULT_WEIGHTS.copy()
        self.model       = None
        self.scaler      = None
        self.notes       = []
        self.accuracy    = {}
        self.cluster_info = {}
        self._load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # STATE PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    def _load_state(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH) as f:
                    s = json.load(f)
                self.weights     = s.get('weights', DEFAULT_WEIGHTS.copy())
                self.notes       = s.get('notes', [])
                self.accuracy    = s.get('accuracy', {})
                self.cluster_info= s.get('cluster_info', {})
            except: pass
        if SKLEARN_OK and os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    saved = pickle.load(f)
                self.model  = saved.get('model')
                self.scaler = saved.get('scaler')
            except: pass

    def _save_state(self):
        with open(STATE_PATH, 'w') as f:
            json.dump({'weights': self.weights, 'notes': self.notes,
                       'accuracy': self.accuracy, 'cluster_info': self.cluster_info,
                       'saved_at': datetime.now().isoformat()}, f, indent=2)
        if SKLEARN_OK and self.model:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────
    def extract_features(self, r):
        price = r.get('Price', 1)
        bb_u  = r.get('BB_Upper', price * 1.02)
        bb_l  = r.get('BB_Lower', price * 0.98)
        bb_b  = (price - bb_l) / (bb_u - bb_l + 1e-9)
        ema_align = sum([
            1 if price > r.get('EMA9',  price) else 0,
            1 if price > r.get('EMA20', price) else 0,
            1 if price > r.get('EMA50', price) else 0,
        ])
        return [
            float(r.get('RSI', 50)),
            float(r.get('MACD_Hist', 0)),
            float(r.get('Vol_Ratio', 1)),
            float(r.get('ADX', 20)),
            float(bb_b),
            float(r.get('Return_20d', 0)),
            float(ema_align),
            1.0 if price > r.get('VWAP', price) else 0.0,
            float(r.get('Score', 50)) / 100.0
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # SUPERVISED LEARNING — train RandomForest on closed predictions
    # ─────────────────────────────────────────────────────────────────────────
    def train_supervised(self, predictions):
        if not SKLEARN_OK:
            return "scikit-learn not installed"
        closed = [p for p in predictions if p.get('status') in ('WIN_T1','WIN_T2','LOSS')]
        if len(closed) < 8:
            return f"Need 8+ closed predictions to train (have {len(closed)})"

        X, y = [], []
        for p in closed:
            feats = [
                p.get('rsi_at_entry', p.get('score', 50) * 0.6),
                p.get('macd_at_entry', 0.5),
                p.get('vol_at_entry', 1.0),
                p.get('adx_at_entry', 22.0),
                p.get('bb_b_at_entry', 0.5),
                p.get('ret20_at_entry', 3.0),
                p.get('ema_align_at_entry', 2.0),
                p.get('vwap_above_at_entry', 1.0),
                float(p.get('score', 60)) / 100.0
            ]
            X.append(feats)
            y.append(1 if p['status'].startswith('WIN') else 0)

        X, y = np.array(X), np.array(y)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        if len(closed) >= 20:
            self.model = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

        self.model.fit(Xs, y)

        # Feature importances → adjust weights
        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            fi_map = dict(zip(FEATURE_NAMES, fi))
            weight_keys = ['RSI','MACD','Volume','ADX','BB','Momentum','EMA','VWAP','RSI']
            for i, key in enumerate(FEATURE_NAMES):
                if i < len(weight_keys):
                    wk = weight_keys[i]
                    if wk in self.weights:
                        self.weights[wk] = round(0.5 + fi[i] * 5.0, 3)

        accuracy = float(np.mean(np.array(y) == self.model.predict(Xs)))
        self.accuracy['supervised'] = round(accuracy * 100, 1)
        self._save_state()
        return f"Trained on {len(closed)} predictions — training accuracy {accuracy*100:.0f}%"

    # ─────────────────────────────────────────────────────────────────────────
    # UNSUPERVISED LEARNING — cluster stocks into behavioral groups
    # ─────────────────────────────────────────────────────────────────────────
    def cluster_stocks(self, stocks, n_clusters=3):
        if not SKLEARN_OK or len(stocks) < n_clusters * 2:
            return {}
        X = np.array([self.extract_features(s) for s in stocks])
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        clusters = {}
        for i, (s, lbl) in enumerate(zip(stocks, labels)):
            g = int(lbl)
            clusters.setdefault(g, []).append(s['Symbol'])
        # Describe each cluster
        descriptions = {}
        centers = km.cluster_centers_
        cluster_names = ['Momentum Leaders', 'Steady Climbers', 'Consolidating Stocks']
        for g, syms in clusters.items():
            c = centers[g]
            avg_rsi = c[0]; avg_vol = c[2]; avg_ema = c[6]
            if avg_ema >= 2.5 and avg_vol > 1.3:
                desc = "🚀 Momentum Leaders — strong uptrend, high activity"
            elif avg_rsi < 55 and avg_vol < 1.0:
                desc = "😴 Consolidating — taking a breather, watch for breakout"
            else:
                desc = "📈 Steady Climbers — moderate trend, building momentum"
            descriptions[g] = {'name': desc, 'stocks': syms}
        self.cluster_info = {str(k): v for k, v in descriptions.items()}
        self._save_state()
        return self.cluster_info

    # ─────────────────────────────────────────────────────────────────────────
    # REINFORCEMENT LEARNING — update weights on each outcome
    # ─────────────────────────────────────────────────────────────────────────
    def reinforce(self, prediction, won):
        """Epsilon-greedy weight update: reinforce winning setups, penalize losing ones."""
        lr = 0.08  # learning rate
        rsi  = prediction.get('score', 60) * 0.6
        vol  = prediction.get('vol_at_entry', 1.0)
        macd = prediction.get('macd_at_entry', 0)
        adx  = prediction.get('adx_at_entry', 20)
        sign = 1.0 if won else -1.0

        # RSI zone quality
        if 50 < rsi < 65:
            self.weights['RSI'] = round(max(0.3, min(2.0, self.weights['RSI'] + sign * lr)), 3)
        elif rsi > 70:
            self.weights['RSI'] = round(max(0.3, min(2.0, self.weights['RSI'] - lr * 0.5)), 3)
        # Volume confirmation
        if vol > 1.5:
            self.weights['Volume'] = round(max(0.3, min(2.0, self.weights['Volume'] + sign * lr * 0.8)), 3)
        # MACD signal
        if macd > 0:
            self.weights['MACD'] = round(max(0.3, min(2.0, self.weights['MACD'] + sign * lr * 0.7)), 3)
        # ADX trend strength
        if adx > 25:
            self.weights['ADX'] = round(max(0.3, min(2.0, self.weights['ADX'] + sign * lr * 0.6)), 3)
        self._save_state()

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICT WIN PROBABILITY
    # ─────────────────────────────────────────────────────────────────────────
    def predict_win_prob(self, stock_data):
        feats = self.extract_features(stock_data)
        if self.model and self.scaler and SKLEARN_OK:
            try:
                Xs = self.scaler.transform([feats])
                prob = float(self.model.predict_proba(Xs)[0][1])
                return round(prob * 100, 1)
            except: pass
        # Rule-based fallback
        rsi,macd,vol,adx,bb_b,ret20,ema,vwap,sc = feats
        p = 50.0
        if 50 < rsi < 65: p += 10
        elif rsi >= 70:   p -= 8
        elif rsi < 40:    p -= 12
        if macd > 0: p += 8
        if vol > 1.5: p += 8
        elif vol < 0.7: p -= 6
        if adx > 25: p += 7
        if ema == 3: p += 10
        elif ema == 2: p += 5
        if vwap: p += 5
        if ret20 > 8: p += 6
        p += sc * 20
        return round(min(95, max(5, p)), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # ML-ENHANCED SCORE
    # ─────────────────────────────────────────────────────────────────────────
    def enhanced_score(self, raw_score, stock_data):
        """Apply learned weights to adjust raw score."""
        feats = self.extract_features(stock_data)
        rsi,macd,vol,adx,bb_b,ret20,ema,vwap,sc = feats
        w = self.weights
        adj = 0
        if 50 < rsi < 65:   adj += 2 * w['RSI']
        elif rsi >= 70:      adj -= 3 * w['RSI']
        if macd > 0:         adj += 2 * w['MACD']
        if vol > 1.5:        adj += 2 * w['Volume']
        if adx > 25:         adj += 1.5 * w['ADX']
        if ema == 3:         adj += 2 * w['EMA']
        if vwap:             adj += 1 * w['VWAP']
        return round(min(100, max(0, raw_score + adj)), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE PLAIN-ENGLISH SELF NOTES
    # ─────────────────────────────────────────────────────────────────────────
    def generate_notes(self, predictions):
        notes = []
        ts = datetime.now().strftime('%b %d, %Y %H:%M')
        closed = [p for p in predictions if p.get('status') in ('WIN_T1','WIN_T2','LOSS')]
        wins   = [p for p in closed if p.get('status','').startswith('WIN')]
        losses = [p for p in closed if p.get('status') == 'LOSS']
        total  = len(closed)

        if total == 0:
            notes.append({
                'icon': '🌱', 'type': 'info', 'updated': ts,
                'note': "I'm brand new! I haven't closed any predictions yet. "
                        "Run the screener a few times over several days and I'll start learning from the results."
            })
            return notes

        acc = len(wins) / total * 100

        # Overall accuracy note
        emoji = '🏆' if acc >= 65 else '📊' if acc >= 50 else '⚠️'
        notes.append({
            'icon': emoji, 'type': 'accuracy', 'updated': ts,
            'note': f"Out of my last {total} predictions, {len(wins)} were correct and {len(losses)} were wrong. "
                    f"That's a {acc:.0f}% success rate. "
                    + ("I'm doing well! " if acc >= 65 else "There's room to improve. " if acc >= 50
                       else "I'm struggling — market conditions may be unusual. ")
        })

        # RSI analysis
        swing_wins = [p for p in wins if p.get('trade_type') == 'SWING']
        intra_wins = [p for p in wins if p.get('trade_type') == 'INTRADAY']
        if swing_wins or losses:
            notes.append({
                'icon': '📈', 'type': 'pattern', 'updated': ts,
                'note': f"My swing trades won {len(swing_wins)} times out of {len([p for p in closed if p.get('trade_type')=='SWING'])} total. "
                        f"My intraday trades won {len(intra_wins)} times out of {len([p for p in closed if p.get('trade_type')=='INTRADAY'])} total. "
                        + ("I'm better at swing trades! " if len(swing_wins) > len(intra_wins) else
                           "I'm better at intraday trades! " if len(intra_wins) > len(swing_wins) else
                           "I'm equally good at both types.")
            })

        # Weight adjustments note
        changed = {k: v for k, v in self.weights.items() if abs(v - 1.0) > 0.05}
        if changed:
            up   = [k for k,v in changed.items() if v > 1.0]
            down = [k for k,v in changed.items() if v < 1.0]
            parts = []
            if up:
                name_map = {'RSI':'Momentum meter','MACD':'Trend direction','Volume':'Trading activity',
                            'ADX':'Trend strength','EMA':'Moving averages','BB':'Price position',
                            'Momentum':'Price growth','VWAP':'Today\'s avg price'}
                parts.append("I give more weight now to: " + ", ".join(name_map.get(k,k) for k in up))
            if down:
                parts.append("I reduced weight on: " + ", ".join(name_map.get(k,k) for k in down))
            if parts:
                notes.append({
                    'icon': '⚙️', 'type': 'weights', 'updated': ts,
                    'note': "Based on past results, I've updated my internal scoring. " + " | ".join(parts) + ". "
                            "This means my scores now reflect what actually worked in the past."
                })

        # P&L insight
        if wins:
            avg_win = np.mean([p.get('profit_loss_pct', 0) for p in wins])
            notes.append({
                'icon': '💰', 'type': 'money', 'updated': ts,
                'note': f"When my predictions are correct, you'd make an average of +{avg_win:.1f}% profit. "
                        f"On ₹1,00,000 that's an average profit of ₹{avg_win*1000:.0f} per winning trade."
            })
        if losses:
            avg_loss = abs(np.mean([p.get('profit_loss_pct', 0) for p in losses]))
            notes.append({
                'icon': '🛡️', 'type': 'risk', 'updated': ts,
                'note': f"When my predictions miss, the stop loss limits the damage to an average of -{avg_loss:.1f}%. "
                        f"On ₹1,00,000 that's an average loss of ₹{avg_loss*1000:.0f} — manageable if wins are larger than losses."
            })

        # Cluster insights
        if self.cluster_info:
            for g, info in self.cluster_info.items():
                syms = info.get('stocks', [])[:5]
                if syms:
                    notes.append({
                        'icon': '🔬', 'type': 'cluster', 'updated': ts,
                        'note': f"Pattern Group — {info.get('name','Group')}: "
                                f"I grouped {', '.join(syms)} together because they show similar price behavior. "
                                "Stocks in the same group often move together."
                    })

        # Model type note
        if total >= 20:
            notes.append({
                'icon': '🤖', 'type': 'model', 'updated': ts,
                'note': f"I've now trained a Gradient Boosting model on {total} predictions. "
                        "This is more accurate than my initial rule-based approach. "
                        "The more predictions close, the smarter I get!"
            })
        elif total >= 8:
            notes.append({
                'icon': '🤖', 'type': 'model', 'updated': ts,
                'note': f"I've trained a Random Forest model on {total} predictions. "
                        f"At 20+ predictions I'll upgrade to a more powerful model. Currently at {total}/20."
            })

        self.notes = notes
        self._save_state()
        return notes

ML = SwingEdgeML()
