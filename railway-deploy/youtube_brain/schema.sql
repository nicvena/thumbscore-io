-- YouTube Intelligence Brain Database Schema
-- This file contains the SQL schema for all brain components

-- ============================================================================
-- YOUTUBE VIDEOS TABLE
-- ============================================================================
-- Stores collected YouTube video data for analysis
CREATE TABLE IF NOT EXISTS youtube_videos (
    video_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    channel_title VARCHAR(255) NOT NULL,
    thumbnail_url TEXT NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    view_count BIGINT NOT NULL DEFAULT 0,
    like_count BIGINT NOT NULL DEFAULT 0,
    comment_count BIGINT NOT NULL DEFAULT 0,
    duration VARCHAR(50) NOT NULL,
    category_id VARCHAR(10) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    description TEXT DEFAULT '',
    views_per_hour DECIMAL(10,2) NOT NULL DEFAULT 0,
    engagement_rate DECIMAL(5,4) NOT NULL DEFAULT 0,
    niche VARCHAR(50) NOT NULL,
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_youtube_videos_niche (niche),
    INDEX idx_youtube_videos_published_at (published_at),
    INDEX idx_youtube_videos_views_per_hour (views_per_hour),
    INDEX idx_youtube_videos_channel_id (channel_id)
);

-- ============================================================================
-- VISUAL PATTERNS TABLE
-- ============================================================================
-- Stores discovered visual patterns from successful thumbnails
CREATE TABLE IF NOT EXISTS visual_patterns (
    pattern_id VARCHAR(255) PRIMARY KEY,
    niche VARCHAR(50) NOT NULL,
    cluster_center JSONB NOT NULL, -- CLIP embedding vector
    success_rate DECIMAL(5,4) NOT NULL,
    avg_views_per_hour DECIMAL(10,2) NOT NULL,
    common_features JSONB NOT NULL DEFAULT '{}',
    thumbnail_examples TEXT[] DEFAULT '{}',
    pattern_description TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_visual_patterns_niche (niche),
    INDEX idx_visual_patterns_success_rate (success_rate)
);

-- ============================================================================
-- FEATURE PATTERNS TABLE
-- ============================================================================
-- Stores patterns in individual features (text, color, etc.)
CREATE TABLE IF NOT EXISTS feature_patterns (
    feature_name VARCHAR(100) NOT NULL,
    niche VARCHAR(50) NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    success_threshold DECIMAL(10,4) NOT NULL,
    impact_score DECIMAL(10,2) NOT NULL,
    examples TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    PRIMARY KEY (feature_name, niche),
    INDEX idx_feature_patterns_niche (niche),
    INDEX idx_feature_patterns_impact_score (impact_score)
);

-- ============================================================================
-- MODEL PERFORMANCE TABLE
-- ============================================================================
-- Stores performance metrics for trained niche models
CREATE TABLE IF NOT EXISTS model_performance (
    niche VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    r2_score DECIMAL(5,4) NOT NULL,
    mse DECIMAL(10,4) NOT NULL,
    feature_importance JSONB NOT NULL DEFAULT '{}',
    training_samples INTEGER NOT NULL,
    validation_samples INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_model_performance_r2_score (r2_score)
);

-- ============================================================================
-- VISUAL TRENDS TABLE
-- ============================================================================
-- Stores detected visual trends and their characteristics
CREATE TABLE IF NOT EXISTS visual_trends (
    trend_id VARCHAR(255) PRIMARY KEY,
    niche VARCHAR(50) NOT NULL,
    trend_type VARCHAR(50) NOT NULL, -- "color", "text", "composition", "style"
    trend_strength DECIMAL(5,4) NOT NULL,
    growth_rate DECIMAL(10,4) NOT NULL,
    trend_description TEXT NOT NULL,
    examples TEXT[] DEFAULT '{}',
    predicted_lifespan INTEGER NOT NULL, -- days
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_visual_trends_niche (niche),
    INDEX idx_visual_trends_strength (trend_strength),
    INDEX idx_visual_trends_type (trend_type)
);

-- ============================================================================
-- CREATOR INSIGHTS TABLE
-- ============================================================================
-- Stores personalized insights for YouTube creators
CREATE TABLE IF NOT EXISTS creator_insights (
    channel_id VARCHAR(255) PRIMARY KEY,
    channel_name VARCHAR(255) NOT NULL,
    niche VARCHAR(50) NOT NULL,
    total_videos INTEGER NOT NULL DEFAULT 0,
    avg_performance JSONB NOT NULL DEFAULT '{}',
    best_performing_patterns JSONB NOT NULL DEFAULT '[]',
    improvement_opportunities TEXT[] DEFAULT '{}',
    competitor_analysis JSONB NOT NULL DEFAULT '{}',
    personalized_recommendations TEXT[] DEFAULT '{}',
    performance_trends JSONB NOT NULL DEFAULT '{}',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_creator_insights_niche (niche),
    INDEX idx_creator_insights_last_updated (last_updated)
);

-- ============================================================================
-- BRAIN STATUS TABLE
-- ============================================================================
-- Stores overall brain status and statistics
CREATE TABLE IF NOT EXISTS brain_status (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL, -- "initializing", "ready", "error"
    components JSONB NOT NULL DEFAULT '{}',
    statistics JSONB NOT NULL DEFAULT '{}',
    last_update TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT,
    
    INDEX idx_brain_status_last_update (last_update)
);

-- ============================================================================
-- BRAIN SCORING LOGS TABLE
-- ============================================================================
-- Stores logs of brain scoring for analysis and debugging
CREATE TABLE IF NOT EXISTS brain_scoring_logs (
    id SERIAL PRIMARY KEY,
    thumbnail_id VARCHAR(255) NOT NULL,
    niche VARCHAR(50) NOT NULL,
    brain_score DECIMAL(5,4) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    pattern_matches JSONB NOT NULL DEFAULT '[]',
    trend_alignment DECIMAL(5,4) NOT NULL,
    model_predictions JSONB,
    explanations TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_brain_scoring_logs_niche (niche),
    INDEX idx_brain_scoring_logs_created_at (created_at),
    INDEX idx_brain_scoring_logs_score (brain_score)
);

-- ============================================================================
-- TREND ALERTS TABLE
-- ============================================================================
-- Stores trend alerts for creators
CREATE TABLE IF NOT EXISTS trend_alerts (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255),
    niche VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- "emerging", "peaking", "declining"
    trend_id VARCHAR(255) NOT NULL,
    urgency VARCHAR(20) NOT NULL, -- "low", "medium", "high"
    recommendation TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_trend_alerts_channel_id (channel_id),
    INDEX idx_trend_alerts_niche (niche),
    INDEX idx_trend_alerts_urgency (urgency),
    INDEX idx_trend_alerts_is_read (is_read)
);

-- ============================================================================
-- PERFORMANCE INSIGHTS TABLE
-- ============================================================================
-- Stores detailed performance insights for creators
CREATE TABLE IF NOT EXISTS performance_insights (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    current_value DECIMAL(10,4) NOT NULL,
    benchmark_value DECIMAL(10,4) NOT NULL,
    percentile DECIMAL(5,2) NOT NULL,
    trend_direction VARCHAR(20) NOT NULL, -- "up", "down", "stable"
    recommendation TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_performance_insights_channel_id (channel_id),
    INDEX idx_performance_insights_metric_name (metric_name),
    INDEX idx_performance_insights_percentile (percentile)
);

-- ============================================================================
-- BRAIN ANALYTICS TABLE
-- ============================================================================
-- Stores aggregated analytics for brain performance
CREATE TABLE IF NOT EXISTS brain_analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    niche VARCHAR(50) NOT NULL,
    total_predictions INTEGER NOT NULL DEFAULT 0,
    avg_confidence DECIMAL(5,4) NOT NULL DEFAULT 0,
    avg_brain_score DECIMAL(5,4) NOT NULL DEFAULT 0,
    pattern_match_rate DECIMAL(5,4) NOT NULL DEFAULT 0,
    trend_alignment_rate DECIMAL(5,4) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(date, niche),
    INDEX idx_brain_analytics_date (date),
    INDEX idx_brain_analytics_niche (niche)
);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update brain status
CREATE OR REPLACE FUNCTION update_brain_status(
    p_status VARCHAR(50),
    p_components JSONB,
    p_statistics JSONB,
    p_error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO brain_status (status, components, statistics, error_message)
    VALUES (p_status, p_components, p_statistics, p_error_message)
    ON CONFLICT DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Function to clean old data
CREATE OR REPLACE FUNCTION clean_old_brain_data() RETURNS VOID AS $$
BEGIN
    -- Clean old scoring logs (keep 30 days)
    DELETE FROM brain_scoring_logs 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Clean old trend alerts (keep 7 days)
    DELETE FROM trend_alerts 
    WHERE created_at < NOW() - INTERVAL '7 days' AND is_read = TRUE;
    
    -- Clean old analytics (keep 1 year)
    DELETE FROM brain_analytics 
    WHERE date < CURRENT_DATE - INTERVAL '1 year';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- View for top performing patterns
CREATE OR REPLACE VIEW top_patterns AS
SELECT 
    niche,
    pattern_id,
    pattern_description,
    success_rate,
    avg_views_per_hour,
    created_at
FROM visual_patterns
WHERE success_rate > 0.7
ORDER BY success_rate DESC, avg_views_per_hour DESC;

-- View for trending patterns
CREATE OR REPLACE VIEW trending_patterns AS
SELECT 
    niche,
    trend_id,
    trend_type,
    trend_description,
    trend_strength,
    growth_rate,
    confidence,
    predicted_lifespan
FROM visual_trends
WHERE trend_strength > 0.5 AND confidence > 0.6
ORDER BY trend_strength DESC, growth_rate DESC;

-- View for creator performance summary
CREATE OR REPLACE VIEW creator_performance_summary AS
SELECT 
    ci.channel_id,
    ci.channel_name,
    ci.niche,
    ci.total_videos,
    ci.avg_performance->>'avg_views_per_hour' as avg_views_per_hour,
    ci.avg_performance->>'avg_engagement_rate' as avg_engagement_rate,
    ci.performance_trends->>'views_trend' as views_trend,
    ci.performance_trends->>'engagement_trend' as engagement_trend,
    ci.last_updated
FROM creator_insights ci
ORDER BY (ci.avg_performance->>'avg_views_per_hour')::DECIMAL DESC;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert initial brain status
INSERT INTO brain_status (status, components, statistics) VALUES (
    'initializing',
    '{"data_collector": false, "pattern_miner": false, "niche_models": false, "trend_detector": false, "insights_engine": false}',
    '{"total_patterns": 0, "total_trends": 0, "trained_niches": []}'
) ON CONFLICT DO NOTHING;
