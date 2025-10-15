#!/bin/bash
# Startup script for Thumbscore.io backend with proper environment variables

cd "/Users/nicvenettacci/Desktop/Thumbnail Lab/thumbnail-lab/python-service"

# Set environment variables
export YOUTUBE_API_KEY="AIzaSyCL6s5QZWeLMTqAXxkviGAjSUy4iinRjng"
export SUPABASE_URL="https://eubfjhegyvivesqgpvlh.supabase.co"
export SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV1YmZqaGVneXZpdmVzcWdwdmxoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAyMTc3MzIsImV4cCI6MjA3NTc5MzczMn0.FIcWYZgmRtI3HUgbQuILSR9ji2Vp1FTRxiCf36ZKpxI"

# Kill any existing server
pkill -f "uvicorn" || true
sleep 2

# Start the server
echo "🚀 Starting Thumbscore.io backend server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > server.log 2>&1 &

# Wait for server to start
sleep 5

# Test server health
echo "🔍 Testing server health..."
curl -s http://localhost:8000/ | head -3

echo "✅ Server started! Check server.log for details."
echo "📊 Server status: http://localhost:8000/"
echo "🔧 FAISS status: http://localhost:8000/internal/faiss-status"
