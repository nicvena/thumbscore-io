-- ============================================================================
-- Thumbscore.io Analytics & Training Data Schema Extension
-- ============================================================================
-- This extends the existing supabase_schema.sql with analytics tables

-- Main analysis logs table for model training and analytics
CREATE TABLE IF NOT EXISTS public.thumbnail_analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- User info
  user_id UUID REFERENCES public.users(id),
  session_id TEXT NOT NULL,
  
  -- Request metadata
  niche TEXT NOT NULL,
  title TEXT,
  thumbnail_index INTEGER,  -- 1, 2, or 3
  
  -- Scores (main metrics)
  final_score DECIMAL(5,2) NOT NULL,
  confidence DECIMAL(5,2),
  tier TEXT,  -- excellent/strong/good/fair/needs_work
  
  -- Component scores (0-100)
  text_clarity DECIMAL(5,2),
  subject_prominence DECIMAL(5,2), 
  contrast_pop DECIMAL(5,2),
  emotion DECIMAL(5,2),
  visual_hierarchy DECIMAL(5,2),
  title_match DECIMAL(5,2),
  power_words DECIMAL(5,2),
  
  -- Detection data
  face_detected BOOLEAN,
  face_size_pct DECIMAL(5,2),
  emotion_detected TEXT,
  word_count INTEGER,
  detected_text TEXT,
  ocr_confidence DECIMAL(5,2),
  saturation DECIMAL(5,4),
  
  -- GPT-4 Vision data
  gpt_summary TEXT,
  gpt_insights JSONB,
  gpt_token_count INTEGER,
  
  -- CTR predictions
  ctr_min DECIMAL(5,2),
  ctr_max DECIMAL(5,2),
  ctr_predicted DECIMAL(5,2),
  
  -- Technical metadata
  processing_time_ms INTEGER,
  scoring_version TEXT DEFAULT 'v1.0',
  model_version TEXT,
  
  -- Image identification (for deduplication)
  image_hash TEXT,
  
  -- Full response (for debugging and reprocessing)
  full_response JSONB,
  
  -- System metadata
  request_ip TEXT,
  user_agent TEXT
);

-- User feedback table (for training labels and quality assurance)
CREATE TABLE IF NOT EXISTS public.user_feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  analysis_id UUID REFERENCES public.thumbnail_analyses(id) ON DELETE CASCADE,
  user_id UUID REFERENCES public.users(id),
  
  -- Feedback ratings
  helpful BOOLEAN,  -- Was the analysis helpful?
  accurate BOOLEAN,  -- Was the score accurate?
  used_winner BOOLEAN,  -- Did they use the recommended thumbnail?
  
  -- Actual performance data (if they report back)
  actual_ctr DECIMAL(5,2),
  actual_views INTEGER,
  actual_impressions INTEGER,
  
  -- Qualitative feedback
  comments TEXT,
  
  -- Metadata
  feedback_type TEXT DEFAULT 'rating', -- rating, performance, comment
  request_ip TEXT
);

-- Model training snapshots (for versioning and A/B testing)
CREATE TABLE IF NOT EXISTS public.model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  version_name TEXT UNIQUE NOT NULL,
  description TEXT,
  
  -- Model configuration
  config JSONB NOT NULL,
  
  -- Performance metrics
  accuracy DECIMAL(5,4),
  avg_confidence DECIMAL(5,2),
  total_analyses INTEGER DEFAULT 0,
  
  -- Status
  is_active BOOLEAN DEFAULT false,
  is_training BOOLEAN DEFAULT false,
  
  -- Training data
  training_data_count INTEGER,
  validation_accuracy DECIMAL(5,4)
);

-- A/B testing framework
CREATE TABLE IF NOT EXISTS public.ab_tests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  test_name TEXT UNIQUE NOT NULL,
  description TEXT,
  
  -- Test configuration
  variant_a_config JSONB,
  variant_b_config JSONB,
  traffic_split DECIMAL(3,2) DEFAULT 0.5, -- 0.5 = 50/50 split
  
  -- Status
  is_active BOOLEAN DEFAULT false,
  start_date TIMESTAMPTZ,
  end_date TIMESTAMPTZ,
  
  -- Results tracking
  variant_a_count INTEGER DEFAULT 0,
  variant_b_count INTEGER DEFAULT 0,
  variant_a_avg_score DECIMAL(5,2),
  variant_b_avg_score DECIMAL(5,2)
);

-- Niche-specific analytics aggregations
CREATE TABLE IF NOT EXISTS public.niche_analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  niche TEXT PRIMARY KEY,
  
  -- Volume metrics
  total_analyses INTEGER DEFAULT 0,
  analyses_last_7d INTEGER DEFAULT 0,
  analyses_last_30d INTEGER DEFAULT 0,
  
  -- Score metrics
  avg_score DECIMAL(5,2),
  median_score DECIMAL(5,2),
  score_std_dev DECIMAL(5,2),
  
  -- Performance metrics
  avg_confidence DECIMAL(5,2),
  avg_processing_time_ms INTEGER,
  
  -- Detection rates
  face_detection_rate DECIMAL(3,2), -- % of thumbnails with faces
  text_detection_rate DECIMAL(3,2), -- % with readable text
  avg_word_count DECIMAL(4,2),
  
  -- User satisfaction
  avg_helpfulness DECIMAL(3,2), -- From user feedback
  feedback_count INTEGER DEFAULT 0,
  
  -- Top insights
  common_strengths JSONB,
  common_weaknesses JSONB,
  recommended_improvements JSONB
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Primary query indexes
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON public.thumbnail_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON public.thumbnail_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_session_id ON public.thumbnail_analyses(session_id);
CREATE INDEX IF NOT EXISTS idx_analyses_niche ON public.thumbnail_analyses(niche);
CREATE INDEX IF NOT EXISTS idx_analyses_final_score ON public.thumbnail_analyses(final_score);
CREATE INDEX IF NOT EXISTS idx_analyses_image_hash ON public.thumbnail_analyses(image_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_scoring_version ON public.thumbnail_analyses(scoring_version);

-- Composite indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_analyses_niche_created_at ON public.thumbnail_analyses(niche, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_score_niche ON public.thumbnail_analyses(final_score, niche);
CREATE INDEX IF NOT EXISTS idx_analyses_confidence ON public.thumbnail_analyses(confidence DESC);

-- Feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_analysis_id ON public.user_feedback(analysis_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON public.user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON public.user_feedback(created_at DESC);

-- Model version indexes
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON public.model_versions(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON public.model_versions(created_at DESC);

-- ============================================================================
-- ANALYTICS VIEWS
-- ============================================================================

-- Daily analytics summary
CREATE OR REPLACE VIEW public.daily_analytics AS
SELECT 
  DATE(created_at) as analysis_date,
  niche,
  COUNT(*) as total_analyses,
  AVG(final_score) as avg_score,
  AVG(confidence) as avg_confidence,
  AVG(processing_time_ms) as avg_processing_time,
  COUNT(DISTINCT user_id) as unique_users,
  COUNT(DISTINCT session_id) as unique_sessions,
  
  -- Detection metrics
  SUM(CASE WHEN face_detected THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as face_detection_rate,
  AVG(CASE WHEN word_count > 0 THEN word_count END) as avg_word_count,
  SUM(CASE WHEN word_count > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as text_detection_rate,
  
  -- Score distribution
  SUM(CASE WHEN final_score >= 80 THEN 1 ELSE 0 END) as excellent_count,
  SUM(CASE WHEN final_score >= 60 AND final_score < 80 THEN 1 ELSE 0 END) as good_count,
  SUM(CASE WHEN final_score < 60 THEN 1 ELSE 0 END) as needs_work_count
  
FROM public.thumbnail_analyses
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), niche
ORDER BY analysis_date DESC, niche;

-- User engagement view
CREATE OR REPLACE VIEW public.user_engagement AS
SELECT 
  user_id,
  COUNT(DISTINCT session_id) as total_sessions,
  COUNT(*) as total_analyses,
  AVG(final_score) as avg_score_received,
  MIN(created_at) as first_analysis,
  MAX(created_at) as last_analysis,
  
  -- Feedback metrics
  (SELECT COUNT(*) FROM public.user_feedback uf WHERE uf.user_id = ta.user_id) as feedback_count,
  (SELECT AVG(CASE WHEN helpful THEN 1.0 ELSE 0.0 END) FROM public.user_feedback uf WHERE uf.user_id = ta.user_id) as helpfulness_rate
  
FROM public.thumbnail_analyses ta
GROUP BY user_id
ORDER BY total_analyses DESC;

-- Score quality metrics view
CREATE OR REPLACE VIEW public.score_quality AS
SELECT 
  scoring_version,
  model_version,
  COUNT(*) as total_analyses,
  AVG(final_score) as avg_score,
  STDDEV(final_score) as score_variance,
  AVG(confidence) as avg_confidence,
  
  -- Consistency metrics
  COUNT(DISTINCT user_id) as unique_users,
  AVG(processing_time_ms) as avg_processing_time,
  
  -- Recent performance
  COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as analyses_last_7d,
  AVG(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN final_score END) as avg_score_last_7d
  
FROM public.thumbnail_analyses
GROUP BY scoring_version, model_version
ORDER BY avg_score DESC;

-- ============================================================================
-- FUNCTIONS FOR ANALYTICS
-- ============================================================================

-- Function to refresh niche analytics (call periodically)
CREATE OR REPLACE FUNCTION public.refresh_niche_analytics()
RETURNS void LANGUAGE SQL AS $$
  INSERT INTO public.niche_analytics (
    niche, total_analyses, analyses_last_7d, analyses_last_30d,
    avg_score, median_score, avg_confidence, avg_processing_time_ms,
    face_detection_rate, text_detection_rate, avg_word_count
  )
  SELECT 
    niche,
    COUNT(*) as total_analyses,
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as analyses_last_7d,
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '30 days' THEN 1 END) as analyses_last_30d,
    AVG(final_score) as avg_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY final_score) as median_score,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time_ms,
    AVG(CASE WHEN face_detected THEN 1.0 ELSE 0.0 END) as face_detection_rate,
    AVG(CASE WHEN word_count > 0 THEN 1.0 ELSE 0.0 END) as text_detection_rate,
    AVG(CASE WHEN word_count > 0 THEN word_count END) as avg_word_count
  FROM public.thumbnail_analyses 
  GROUP BY niche
  ON CONFLICT (niche) DO UPDATE SET
    total_analyses = EXCLUDED.total_analyses,
    analyses_last_7d = EXCLUDED.analyses_last_7d,
    analyses_last_30d = EXCLUDED.analyses_last_30d,
    avg_score = EXCLUDED.avg_score,
    median_score = EXCLUDED.median_score,
    avg_confidence = EXCLUDED.avg_confidence,
    avg_processing_time_ms = EXCLUDED.avg_processing_time_ms,
    face_detection_rate = EXCLUDED.face_detection_rate,
    text_detection_rate = EXCLUDED.text_detection_rate,
    avg_word_count = EXCLUDED.avg_word_count,
    updated_at = NOW();
$$;

-- Function to clean up old analytics data (call monthly)
CREATE OR REPLACE FUNCTION public.cleanup_old_analytics()
RETURNS void LANGUAGE SQL AS $$
  -- Keep detailed logs for 90 days, summarized data forever
  DELETE FROM public.thumbnail_analyses 
  WHERE created_at < NOW() - INTERVAL '90 days';
  
  -- Keep feedback for 1 year
  DELETE FROM public.user_feedback 
  WHERE created_at < NOW() - INTERVAL '1 year';
$$;

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS
ALTER TABLE public.thumbnail_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ab_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.niche_analytics ENABLE ROW LEVEL SECURITY;

-- Policies for thumbnail_analyses (readable by all for analytics, writable by service)
CREATE POLICY "Analytics data is readable by all" ON public.thumbnail_analyses
  FOR SELECT USING (true);

CREATE POLICY "Service can insert analysis data" ON public.thumbnail_analyses
  FOR INSERT WITH CHECK (true);

-- Policies for user_feedback
CREATE POLICY "Users can view own feedback" ON public.user_feedback
  FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert own feedback" ON public.user_feedback
  FOR INSERT WITH CHECK (user_id = auth.uid());

-- Policies for aggregated analytics (readable by all)
CREATE POLICY "Analytics views are public" ON public.niche_analytics
  FOR SELECT USING (true);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE public.thumbnail_analyses IS 'Comprehensive logging of all thumbnail analyses for model training and analytics';
COMMENT ON TABLE public.user_feedback IS 'User feedback and performance data for training labels';
COMMENT ON TABLE public.model_versions IS 'Model version tracking for A/B testing and rollbacks';
COMMENT ON TABLE public.ab_tests IS 'A/B testing framework for scoring algorithm improvements';
COMMENT ON TABLE public.niche_analytics IS 'Aggregated analytics by niche for fast dashboard queries';

COMMENT ON COLUMN public.thumbnail_analyses.image_hash IS 'SHA256 hash of image for deduplication';
COMMENT ON COLUMN public.thumbnail_analyses.full_response IS 'Complete analysis response for debugging and reprocessing';
COMMENT ON COLUMN public.thumbnail_analyses.gpt_insights IS 'Structured insights from GPT-4 Vision for training data';