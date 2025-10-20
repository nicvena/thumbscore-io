-- Email Subscribers Table for Thumbscore.io
-- Run this in your Supabase SQL editor

CREATE TABLE email_subscribers (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  subscribed_at TIMESTAMP DEFAULT NOW(),
  source VARCHAR(50) DEFAULT 'free_user_signup',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Add index for faster email lookups
CREATE INDEX idx_email_subscribers_email ON email_subscribers(email);

-- Add index for source filtering
CREATE INDEX idx_email_subscribers_source ON email_subscribers(source);

-- Optional: Add RLS (Row Level Security) if needed
-- ALTER TABLE email_subscribers ENABLE ROW LEVEL SECURITY;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL ON email_subscribers TO authenticated;
-- GRANT ALL ON email_subscribers TO service_role;
