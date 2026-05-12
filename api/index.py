"""
SwingEdge Pro — Vercel Entry Point
Routes all requests to the Flask app.
"""
import sys, os

# Make project root the working dir so Flask finds /templates correctly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from app import app

# Vercel looks for `handler`
handler = app
