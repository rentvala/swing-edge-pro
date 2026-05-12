#!/bin/bash
# SwingEdge Pro — macOS Quick Start
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "  ███████╗██╗    ██╗██╗███╗   ██╗ ██████╗     ███████╗██████╗  ██████╗ "
echo "  ██╔════╝██║    ██║██║████╗  ██║██╔════╝     ██╔════╝██╔══██╗██╔════╝ "
echo "  ███████╗██║ █╗ ██║██║██╔██╗ ██║██║  ███╗    █████╗  ██║  ██║██║  ███╗"
echo "  ╚════██║██║███╗██║██║██║╚██╗██║██║   ██║    ██╔══╝  ██║  ██║██║   ██║"
echo "  ███████║╚███╔███╔╝██║██║ ╚████║╚██████╔╝    ███████╗██████╔╝╚██████╔╝"
echo "  ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚══════╝╚═════╝  ╚═════╝ "
echo ""
echo "  SwingEdge Pro — NSE Swing & Intraday Trading Intelligence"
echo "  ─────────────────────────────────────────────────────────"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "🚀 Starting SwingEdge Pro on http://localhost:5050"
echo "   → Open your browser and go to: http://localhost:5050"
echo "   → Press Ctrl+C to stop the server"
echo ""

# Open browser after 2 seconds (macOS)
(sleep 2 && open http://localhost:5050) &

# Start Flask
python3 app.py
