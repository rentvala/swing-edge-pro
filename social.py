"""
SwingEdge Pro — Social Media & News Sentiment Engine
Sources: Reddit (r/IndianStockMarket, r/DalalStreetTalks), Google News RSS,
         Economic Times RSS, Moneycontrol headlines
Note: Twitter/X requires official API keys (add TWITTER_BEARER_TOKEN to env to enable)
"""
import re, os, json, requests
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False

HEADERS = {'User-Agent': 'SwingEdgePro/4.0 (educational project)'}
TIMEOUT = 12

# NSE stock → keywords mapping
STOCK_KEYWORDS = {
    'RELIANCE': ['reliance','RIL'], 'TCS': ['TCS','tata consultancy'],
    'HDFCBANK': ['hdfc bank','HDFCBANK'], 'INFY': ['infosys','INFY'],
    'ICICIBANK': ['icici bank','ICICIBANK'], 'SBIN': ['SBI','state bank'],
    'BAJFINANCE': ['bajaj finance','BAJFINANCE'], 'ASIANPAINT': ['asian paints','ASIANPAINT'],
    'WIPRO': ['wipro','WIPRO'], 'AXISBANK': ['axis bank','AXISBANK'],
    'MARUTI': ['maruti','MSIL'], 'SUNPHARMA': ['sun pharma','SUNPHARMA'],
    'TITAN': ['titan','TITAN'], 'ADANIENT': ['adani enterprises','ADANIENT'],
    'ADANIPORTS': ['adani ports','ADANIPORTS'], 'COALINDIA': ['coal india','COALINDIA'],
    'CIPLA': ['cipla','CIPLA'], 'DRREDDY': ["dr reddy","DRREDDY"],
    'TATASTEEL': ['tata steel','TATASTEEL'], 'HINDALCO': ['hindalco','HINDALCO'],
    'BHARTIARTL': ['airtel','bharti airtel','BHARTIARTL'], 'LT': ['larsen','L&T','LTIM'],
    'APOLLOHOSP': ['apollo hospital','APOLLOHOSP'], 'NESTLEIND': ['nestle','NESTLEIND'],
    'PIDILITIND': ['pidilite','PIDILITIND'], 'DIVISLAB': ['divi','DIVISLAB'],
    'DMART': ['dmart','avenue supermart'], 'IRCTC': ['IRCTC','indian railway'],
    'PAYTM': ['paytm','one97'], 'ITC': ['ITC','cigarette'],
    'ULTRACEMCO': ['ultratech cement','ULTRACEMCO'], 'ONGC': ['ONGC','oil india'],
    'POWERGRID': ['power grid','POWERGRID'], 'NTPC': ['NTPC','ntpc'],
    'KOTAKBANK': ['kotak bank','KOTAKBANK'], 'BAJAJ-AUTO': ['bajaj auto','BAJAJAUTO'],
    'DABUR': ['dabur','DABUR'], 'HEROMOTOCO': ['hero moto','HEROMOTOCO'],
    'M&M': ['mahindra','M&M'], 'TATACONSUM': ['tata consumer','TATACONSUM'],
}

BUY_WORDS  = ['buy','bullish','breakout','surge','rally','boom','rocket','📈','🚀','moon','all time high','ath','accumulate','strong','recommend','positive']
SELL_WORDS = ['sell','bearish','crash','dump','fall','drop','short','avoid','weak','negative','overvalued','📉','correction','caution']

# ── DATA SOURCES ───────────────────────────────────────────────────────────────

def get_reddit_posts():
    posts = []
    subs = ['IndianStockMarket', 'DalalStreetTalks', 'IndiaInvestments', 'NSEIndia']
    for sub in subs:
        try:
            url = f'https://www.reddit.com/r/{sub}/hot.json?limit=50&t=day'
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                for child in data.get('data', {}).get('children', []):
                    p = child.get('data', {})
                    text = (p.get('title','') + ' ' + p.get('selftext',''))[:800]
                    posts.append({
                        'source': f'Reddit/r/{sub}',
                        'text': text,
                        'score': p.get('score', 0),
                        'url': 'https://reddit.com' + p.get('permalink','')
                    })
        except Exception as e:
            pass
    return posts

def get_news_rss():
    posts = []
    feeds = [
        ('Google News NSE', 'https://news.google.com/rss/search?q=NSE+india+stock+market&hl=en-IN&gl=IN&ceid=IN:en'),
        ('Google News Nifty', 'https://news.google.com/rss/search?q=Nifty50+stock+market&hl=en-IN&gl=IN&ceid=IN:en'),
        ('ET Markets', 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms'),
    ]
    if not FEEDPARSER_OK:
        return posts
    for name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                text = entry.get('title','') + ' ' + entry.get('summary','')[:400]
                posts.append({'source': name, 'text': text, 'score': 10, 'url': entry.get('link','')})
        except: pass
    return posts

def get_twitter_posts():
    """Placeholder — X/Twitter requires API keys."""
    bearer = os.environ.get('TWITTER_BEARER_TOKEN','')
    if not bearer:
        return [{'source': 'Twitter/X', 'text': '', 'score': 0,
                 'note': 'Add TWITTER_BEARER_TOKEN env variable to enable Twitter feed'}]
    # If key provided:
    try:
        headers = {'Authorization': f'Bearer {bearer}'}
        params  = {'query': 'NSE stock India -is:retweet lang:en', 'max_results': 50,
                   'tweet.fields': 'public_metrics,created_at'}
        r = requests.get('https://api.twitter.com/2/tweets/search/recent',
                         headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            tweets = r.json().get('data', [])
            return [{'source': 'Twitter/X', 'text': t.get('text',''),
                     'score': t.get('public_metrics',{}).get('like_count',0),
                     'url': f"https://twitter.com/i/web/status/{t['id']}"}
                    for t in tweets]
    except: pass
    return []

# ── SENTIMENT ANALYSIS ─────────────────────────────────────────────────────────

def analyze_sentiment(text):
    """Returns (buy_score, sell_score, polarity)"""
    text_l = text.lower()
    buy_score  = sum(2 if w in text_l else 0 for w in BUY_WORDS)
    sell_score = sum(2 if w in text_l else 0 for w in SELL_WORDS)
    polarity   = 0.0
    if TEXTBLOB_OK:
        try:
            polarity = TextBlob(text).sentiment.polarity
        except: pass
    if polarity > 0.1:   buy_score += 1
    elif polarity < -0.1: sell_score += 1
    return buy_score, sell_score, round(polarity, 3)

def extract_mentions(posts):
    """Count bullish/bearish mentions per stock across all posts."""
    mentions = defaultdict(lambda: {'bull': 0, 'bear': 0, 'posts': [], 'total_score': 0, 'polarity': 0.0})
    
    for post in posts:
        text = post.get('text', '')
        if not text.strip():
            continue
        text_l = text.lower()
        bs, ss, pol = analyze_sentiment(text)
        
        for sym, keywords in STOCK_KEYWORDS.items():
            found = any(kw.lower() in text_l for kw in keywords)
            if found:
                post_weight = max(1, post.get('score', 1))
                mentions[sym]['bull']       += bs * post_weight
                mentions[sym]['bear']       += ss * post_weight
                mentions[sym]['total_score']+= post.get('score', 0)
                mentions[sym]['polarity']   += pol
                snippet = text[:120].replace('\n',' ') + '...'
                if len(mentions[sym]['posts']) < 3:
                    mentions[sym]['posts'].append({
                        'source': post.get('source',''),
                        'snippet': snippet,
                        'url': post.get('url','')
                    })
    return mentions

# ── LEADERBOARD ────────────────────────────────────────────────────────────────

def build_leaderboard():
    all_posts = []
    all_posts.extend(get_reddit_posts())
    all_posts.extend(get_news_rss())
    all_posts.extend(get_twitter_posts())
    twitter_note = next((p.get('note') for p in all_posts if 'note' in p), None)
    all_posts = [p for p in all_posts if p.get('text','').strip()]
    
    if not all_posts:
        return {'buy': [], 'sell': [], 'note': 'Could not fetch social data — check your internet connection',
                'twitter_note': twitter_note, 'total_posts': 0}
    
    mentions = extract_mentions(all_posts)
    
    # Score each stock
    scored = []
    for sym, data in mentions.items():
        bull = data['bull']; bear = data['bear']
        total = bull + bear
        if total == 0: continue
        net_sentiment = (bull - bear) / (total + 1)
        avg_polarity = data['polarity'] / max(1, len(data['posts']))
        buy_rank  = net_sentiment + avg_polarity * 0.5 + data['total_score'] * 0.001
        sell_rank = -net_sentiment - avg_polarity * 0.5 + data['total_score'] * 0.001
        scored.append({
            'symbol': sym, 'buy_rank': buy_rank, 'sell_rank': sell_rank,
            'bull_mentions': bull, 'bear_mentions': bear,
            'sentiment_pct': round(net_sentiment * 100, 1),
            'posts': data['posts'],
            'total_community_score': data['total_score']
        })
    
    scored.sort(key=lambda x: x['buy_rank'], reverse=True)
    top_buy  = scored[:5]
    scored.sort(key=lambda x: x['sell_rank'], reverse=True)
    top_sell = scored[:5]
    
    return {
        'buy':  top_buy, 'sell': top_sell,
        'note': f"Analyzed {len(all_posts)} posts from Reddit, News RSS" +
                (" and Twitter/X" if not twitter_note else " (Twitter/X not configured)"),
        'twitter_note': twitter_note,
        'total_posts': len(all_posts),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
    }
