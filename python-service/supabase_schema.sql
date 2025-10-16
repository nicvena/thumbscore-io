-- ============================================================================
-- Thumbscore.io Credit-Based System Schema
-- ============================================================================

-- Users table (if not already exists)
create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  email text unique,
  device_id text unique, -- for anonymous users
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Credit wallet system
create table if not exists public.credits (
  user_id uuid references public.users(id) on delete cascade,
  plan text not null default 'free',
  monthly_quota int not null default 1,          -- analyses that can call GPT Vision
  max_thumbnails_per_analysis int not null default 3, -- max thumbnails per analysis
  used_this_cycle int not null default 0,
  cycle_start date not null default date_trunc('month', now()),
  updated_at timestamptz default now(),
  primary key (user_id)
);

-- Score cache to avoid re-paying for identical analyses
create table if not exists public.score_cache (
  cache_key text primary key,                    -- sha256(image+title+niche+version)
  payload jsonb not null,
  created_at timestamptz default now(),
  expires_at timestamptz default (now() + interval '30 days')
);

-- Simple IP rate limiting
create table if not exists public.rate_limits (
  ip text,
  window_start timestamptz,
  hits int default 1,
  created_at timestamptz default now(),
  primary key (ip, window_start)
);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Monthly credit reset function
create or replace function public.reset_credits_monthly()
returns void language sql as $$
  update public.credits
  set used_this_cycle = 0,
      cycle_start = date_trunc('month', now()),
      updated_at = now()
  where cycle_start < date_trunc('month', now());
$$;

-- Clean up expired cache entries
create or replace function public.cleanup_expired_cache()
returns void language sql as $$
  delete from public.score_cache
  where expires_at < now();
$$;

-- Clean up old rate limit entries (older than 24 hours)
create or replace function public.cleanup_rate_limits()
returns void language sql as $$
  delete from public.rate_limits
  where window_start < (now() - interval '24 hours');
$$;

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Indexes for performance
create index if not exists idx_credits_user_id on public.credits(user_id);
create index if not exists idx_credits_plan on public.credits(plan);
create index if not exists idx_score_cache_created_at on public.score_cache(created_at);
create index if not exists idx_score_cache_expires_at on public.score_cache(expires_at);
create index if not exists idx_rate_limits_window_start on public.rate_limits(window_start);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS
alter table public.users enable row level security;
alter table public.credits enable row level security;
alter table public.score_cache enable row level security;
alter table public.rate_limits enable row level security;

-- Policies (basic - can be enhanced later)
create policy "Users can view own data" on public.users
  for select using (auth.uid() = id or device_id = current_setting('request.jwt.claims', true)::json->>'device_id');

create policy "Users can view own credits" on public.credits
  for select using (auth.uid() = user_id or user_id in (select id from public.users where device_id = current_setting('request.jwt.claims', true)::json->>'device_id'));

create policy "Score cache is readable by all" on public.score_cache
  for select using (true);

create policy "Rate limits are readable by all" on public.rate_limits
  for select using (true);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert default plans if they don't exist
insert into public.users (id, email, device_id) 
values 
  ('00000000-0000-0000-0000-000000000000', 'system@thumbscore.io', 'system')
on conflict (id) do nothing;

-- ============================================================================
-- COMMENTS
-- ============================================================================

comment on table public.users is 'User accounts (email-based or anonymous with device_id)';
comment on table public.credits is 'Credit wallet system for controlling GPT Vision API costs';
comment on table public.score_cache is 'Cached analysis results to avoid duplicate API calls';
comment on table public.rate_limits is 'IP-based rate limiting to prevent abuse';

comment on column public.credits.monthly_quota is 'Number of GPT Vision analyses allowed per month';
comment on column public.credits.used_this_cycle is 'Number of analyses used in current billing cycle';
comment on column public.credits.cycle_start is 'Start date of current billing cycle';

comment on column public.score_cache.cache_key is 'SHA256 hash of (image_bytes + title + niche + version)';
comment on column public.score_cache.payload is 'Complete analysis result JSON';
