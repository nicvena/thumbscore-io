# Thumbscore Python Backend

AI-powered YouTube thumbnail scoring service built with FastAPI.

## Quick Start

This service automatically starts with Railway using the Procfile.

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `RESEND_API_KEY`
- `JWT_SECRET`

## API Endpoints

- `GET /health` - Health check
- `POST /v1/score` - Score thumbnails
- `GET /docs` - API documentation