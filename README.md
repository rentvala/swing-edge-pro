# SwingEdge Pro 🚀
### NSE Swing & Intraday Trading Intelligence — macOS App

---

## ⚡ Quickest Way to Start (macOS)

### Just double-click `SwingEdgePro.command` in Finder!

That's it. It will:
1. ✅ Auto-detect your Python installation
2. ✅ Create a virtual environment (first run only, ~30 sec)
3. ✅ Install all dependencies automatically
4. ✅ Start the server
5. ✅ Open your browser at http://localhost:5050

**First run:** Takes ~1–2 minutes (installing packages)
**Every run after:** Opens in ~5 seconds

---

## 🛠 If double-click doesn't work (one-time fix)

macOS may block `.command` files from unknown sources. Fix it once:

```bash
# Open Terminal and run:
chmod +x /path/to/SwingEdgePro/SwingEdgePro.command
```

Or right-click the file → Open → Click "Open" in the security dialog.

---

## 📁 Project Files

```
SwingEdgePro/
├── SwingEdgePro.command    ← ⭐ DOUBLE-CLICK THIS to launch
├── app.py                  ← Flask backend + screener engine
├── templates/
│   └── index.html          ← Full dark UI (single-page app)
├── data/
│   └── swingEdgePro_schema_fixed.json
├── requirements.txt
└── README.md
```

---

## 📊 Scoring System (0–100)

| Factor          | Max Points | Signal                                      |
|----------------|-----------|---------------------------------------------|
| RSI             | 15         | 50–65 = bullish zone                        |
| MACD            | 15         | Histogram positive + above signal line      |
| EMA Stack       | 15         | Price > EMA9 > EMA20 > EMA50 = perfect bull |
| Volume          | 15         | >1.5x avg + price up = breakout             |
| Bollinger %B    | 10         | Mid-upper band = momentum                   |
| ADX             | 10         | >25 = strong trending market                |
| 20-day Momentum | 10         | >8% = strong momentum                       |
| VWAP            | 10         | Price above VWAP = institutional buying     |

---

## ⚙️ Requirements
- Python 3.9+ (download from python.org if missing)
- Internet connection (for live NSE data from Yahoo Finance)
- macOS 12+
