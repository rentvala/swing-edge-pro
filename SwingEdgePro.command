#!/bin/bash
cd "$(dirname "$0")"
clear
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║   SwingEdge Pro v5.0 — AI-Powered NSE Trading Intelligence   ║"
echo "  ║   ML Learning · Buy Now Tab · Plain English · Market Buzz    ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  📂 $(pwd)"
if ! command -v python3 &>/dev/null; then
  echo "  ❌ Python3 not found. Get it at https://www.python.org/downloads/"
  read -p "  Press Enter..."; exit 1
fi
echo "  ✅ $(python3 --version)"
[ ! -d ".venv" ] && echo "  📦 First run — setting up (1-2 min)..." && python3 -m venv .venv
source .venv/bin/activate
echo "  📦 Installing packages..."
pip install -r requirements.txt -q --disable-pip-version-check
[ $? -ne 0 ] && echo "  ❌ Install failed" && read -p "Press Enter..." && exit 1
echo "  ✅ Ready!"
EXISTING=$(lsof -ti:5050 2>/dev/null)
[ -n "$EXISTING" ] && kill -9 $EXISTING 2>/dev/null && sleep 1
(sleep 4 && open "http://localhost:5050") &
echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║  🚀  Opening http://localhost:5050 in 4 seconds…             ║"
echo "  ║  📊  Click 'Refresh' to load today's AI-ranked NSE stocks    ║"
echo "  ║  🤖  ML model trains itself after each prediction closes     ║"
echo "  ║  ⭐  'Buy Now' tab shows market-hour-aware best picks         ║"
echo "  ║  Press Ctrl+C to stop                                        ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""
python3 app.py
echo ""; read -p "  Stopped. Press Enter to close..."
