-- Create waitlist table in Supabase
-- Run this SQL in your Supabase SQL Editor

CREATE TABLE waitlist (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT NOT NULL,
  plan TEXT NOT NULL CHECK (plan IN ('creator', 'pro')),
  max_price TEXT NOT NULL,
  interests TEXT[] DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
  
  -- Prevent duplicate email + plan combinations
  UNIQUE(email, plan)
);

-- Create index for better query performance
CREATE INDEX idx_waitlist_email ON waitlist(email);
CREATE INDEX idx_waitlist_plan ON waitlist(plan);
CREATE INDEX idx_waitlist_created_at ON waitlist(created_at);

-- Enable Row Level Security (RLS)
ALTER TABLE waitlist ENABLE ROW LEVEL SECURITY;

-- Policy to allow service role to do everything
CREATE POLICY "Service role can manage waitlist" ON waitlist
  FOR ALL USING (auth.role() = 'service_role');

-- Policy to allow authenticated users to insert their own entries
CREATE POLICY "Users can insert their waitlist entries" ON waitlist
  FOR INSERT WITH CHECK (true);

-- Policy to allow reading waitlist counts (for public display)
CREATE POLICY "Anyone can read waitlist counts" ON waitlist
  FOR SELECT USING (true);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = timezone('utc'::text, now());
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_waitlist_updated_at
  BEFORE UPDATE ON waitlist
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data (optional)
-- INSERT INTO waitlist (email, plan, max_price, interests) VALUES
-- ('test@example.com', 'creator', '$19', ARRAY['API access', 'Team accounts']);