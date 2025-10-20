#!/usr/bin/env python3
"""
Analytics Logger for Thumbscore.io

Comprehensive logging of all thumbnail analyses to Supabase for:
1. Model training data collection
2. Analytics and user behavior tracking  
3. Quality monitoring and A/B testing
4. Performance optimization insights
"""

import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class AnalyticsLogger:
    """
    Handles comprehensive logging of thumbnail analyses to Supabase
    for model training, analytics, and quality monitoring
    """
    
    def __init__(self):
        """Initialize Supabase client for analytics logging"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("[ANALYTICS] Supabase credentials missing - analytics logging disabled")
            self.client: Optional[Client] = None
            self.enabled = False
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                self.enabled = True
                logger.info("[ANALYTICS] Analytics logger initialized successfully")
            except Exception as e:
                logger.error(f"[ANALYTICS] Failed to initialize Supabase client: {e}")
                self.client = None
                self.enabled = False
    
    def _generate_image_hash(self, image_data: bytes) -> str:
        """Generate SHA256 hash of image for deduplication"""
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    def _extract_gpt_data(self, full_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract GPT-4 Vision data from response"""
        gpt_data = {}
        
        # Try to find GPT summary data
        if 'gpt_summary' in full_response:
            gpt_summary = full_response['gpt_summary']
            if isinstance(gpt_summary, dict):
                gpt_data['summary'] = gpt_summary.get('winner_summary', '')
                gpt_data['insights'] = gpt_summary.get('insights', [])
                gpt_data['token_count'] = gpt_summary.get('token_count', 0)
            elif isinstance(gpt_summary, str):
                gpt_data['summary'] = gpt_summary
                gpt_data['insights'] = []
                gpt_data['token_count'] = 0
        
        # Also check metadata for GPT summaries
        if 'metadata' in full_response and 'gpt_summaries' in full_response['metadata']:
            gpt_summaries = full_response['metadata']['gpt_summaries']
            if gpt_summaries:
                # Take the first available summary
                first_summary = next(iter(gpt_summaries.values()))
                if not gpt_data.get('summary'):
                    gpt_data['summary'] = first_summary.get('winner_summary', '')
                if not gpt_data.get('insights'):
                    gpt_data['insights'] = first_summary.get('insights', [])
        
        return gpt_data
    
    def _extract_detections(self, thumbnail_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detection data from thumbnail analysis"""
        detections = {}
        
        # Face detection
        face_boxes = thumbnail_data.get('face_boxes', [])
        detections['face_detected'] = len(face_boxes) > 0
        
        # Try to get face size from various possible locations
        detections['face_size_pct'] = 0
        if 'subscores' in thumbnail_data:
            subject_prominence = thumbnail_data['subscores'].get('subject_prominence', 0)
            detections['face_size_pct'] = subject_prominence
        elif 'subScores' in thumbnail_data:
            subject_prominence = thumbnail_data['subScores'].get('subjectProminence', 0)
            detections['face_size_pct'] = subject_prominence
        
        # OCR/Text detection
        detected_text = ''
        word_count = 0
        ocr_confidence = 0
        
        # Try different locations for OCR data
        if 'ocr_highlights' in thumbnail_data:
            ocr_highlights = thumbnail_data['ocr_highlights']
            if ocr_highlights:
                detected_text = ocr_highlights[0].get('text', '')
                ocr_confidence = ocr_highlights[0].get('confidence', 0) * 100
        
        # Check for text in other locations
        if not detected_text and 'detected_text' in thumbnail_data:
            detected_text = thumbnail_data['detected_text']
        
        if detected_text:
            word_count = len(detected_text.split())
        
        detections['detected_text'] = detected_text[:500]  # Truncate for storage
        detections['word_count'] = word_count
        detections['ocr_confidence'] = ocr_confidence
        
        # Color/Contrast data
        detections['saturation'] = thumbnail_data.get('saturation', 0)
        if 'subscores' in thumbnail_data:
            detections['contrast_score'] = thumbnail_data['subscores'].get('contrast_pop', 0)
        elif 'subScores' in thumbnail_data:
            detections['contrast_score'] = thumbnail_data['subScores'].get('contrastColorPop', 0)
        else:
            detections['contrast_score'] = 0
        
        # Emotion detection
        detections['emotion'] = 'neutral'  # Default
        if face_boxes:
            for face in face_boxes:
                if 'emotion' in face:
                    detections['emotion'] = face['emotion']
                    break
        
        return detections
    
    def log_analysis(
        self,
        user_id: Optional[str],
        session_id: str,
        niche: str,
        title: Optional[str],
        thumbnail_index: int,
        thumbnail_data: Dict[str, Any],
        image_data: bytes,
        processing_time_ms: int,
        request_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """
        Log comprehensive thumbnail analysis data to Supabase
        
        Args:
            user_id: User identifier (UUID or None for anonymous)
            session_id: Session identifier for grouping related analyses
            niche: Content niche (gaming, business, etc.)
            title: Video title
            thumbnail_index: Position in comparison (1, 2, or 3)
            thumbnail_data: Complete analysis result from scoring system
            image_data: Raw image bytes for hash generation
            processing_time_ms: Time taken for analysis
            request_ip: Client IP address
            user_agent: Client user agent string
            
        Returns:
            analysis_id (UUID) if successful, None if failed
        """
        
        if not self.enabled or not self.client:
            return None
        
        try:
            # Generate image hash for deduplication
            image_hash = self._generate_image_hash(image_data)
            
            # Extract core score data
            final_score = thumbnail_data.get('ctr_score', 0)
            confidence = thumbnail_data.get('confidence', 0)
            tier = thumbnail_data.get('tier', 'unknown')
            
            # Extract component scores
            subscores = thumbnail_data.get('subscores', {})
            if not subscores:
                subscores = thumbnail_data.get('subScores', {})
            
            # Extract detection data
            detections = self._extract_detections(thumbnail_data)
            
            # Extract GPT data
            gpt_data = self._extract_gpt_data(thumbnail_data)
            
            # Extract CTR predictions
            ctr_prediction = thumbnail_data.get('ctr_prediction', {})
            if isinstance(ctr_prediction, str):
                # Handle cases where ctr_prediction is just a string like "73%"
                try:
                    ctr_value = float(ctr_prediction.replace('%', ''))
                    ctr_min = max(0, ctr_value - 5)
                    ctr_max = min(100, ctr_value + 5)
                    ctr_predicted = ctr_value
                except:
                    ctr_min = ctr_max = ctr_predicted = None
            else:
                ctr_min = ctr_prediction.get('ctr_min')
                ctr_max = ctr_prediction.get('ctr_max')
                ctr_predicted = ctr_prediction.get('ctr_predicted', final_score)
            
            # Prepare row for insertion
            row = {
                'user_id': user_id,
                'session_id': session_id,
                'niche': niche,
                'title': title,
                'thumbnail_index': thumbnail_index,
                
                # Main scores
                'final_score': float(final_score) if final_score else 0,
                'confidence': float(confidence) if confidence else 0,
                'tier': tier,
                
                # Component scores
                'text_clarity': float(subscores.get('clarity', subscores.get('text_clarity', 0))),
                'subject_prominence': float(subscores.get('subject_prominence', subscores.get('subjectProminence', 0))),
                'contrast_pop': float(subscores.get('contrast_pop', subscores.get('contrastColorPop', 0))),
                'emotion': float(subscores.get('emotion', 0)),
                'visual_hierarchy': float(subscores.get('hierarchy', subscores.get('visualHierarchy', 0))),
                'title_match': float(subscores.get('title_match', subscores.get('clickIntentMatch', 0))),
                'power_words': float(subscores.get('power_words', subscores.get('powerWords', 0))),
                
                # Detection data
                'face_detected': detections.get('face_detected', False),
                'face_size_pct': float(detections.get('face_size_pct', 0)),
                'emotion_detected': detections.get('emotion', 'neutral'),
                'word_count': int(detections.get('word_count', 0)),
                'detected_text': detections.get('detected_text', ''),
                'ocr_confidence': float(detections.get('ocr_confidence', 0)),
                'saturation': float(detections.get('saturation', 0)),
                
                # GPT-4 data
                'gpt_summary': gpt_data.get('summary', ''),
                'gpt_insights': gpt_data.get('insights', []),
                'gpt_token_count': int(gpt_data.get('token_count', 0)),
                
                # CTR predictions
                'ctr_min': float(ctr_min) if ctr_min is not None else None,
                'ctr_max': float(ctr_max) if ctr_max is not None else None,
                'ctr_predicted': float(ctr_predicted) if ctr_predicted is not None else float(final_score),
                
                # Technical metadata
                'processing_time_ms': int(processing_time_ms),
                'scoring_version': thumbnail_data.get('scoring_version', 'v1.0'),
                'model_version': thumbnail_data.get('model_version', 'v1.0'),
                'image_hash': image_hash,
                
                # Request metadata
                'request_ip': request_ip,
                'user_agent': user_agent,
                
                # Full response for debugging
                'full_response': thumbnail_data
            }
            
            # Remove None values to avoid database issues
            row = {k: v for k, v in row.items() if v is not None}
            
            # Insert into database
            result = self.client.table('thumbnail_analyses').insert(row).execute()
            
            if result.data and len(result.data) > 0:
                analysis_id = result.data[0]['id']
                logger.info(f"[ANALYTICS] Logged analysis: {analysis_id} (score: {final_score}, niche: {niche})")
                return analysis_id
            else:
                logger.warning("[ANALYTICS] Failed to insert analysis - no data returned")
                return None
                
        except Exception as e:
            logger.error(f"[ANALYTICS] Error logging analysis: {e}")
            return None
    
    def log_feedback(
        self,
        analysis_id: str,
        user_id: str,
        helpful: Optional[bool] = None,
        accurate: Optional[bool] = None,
        used_winner: Optional[bool] = None,
        actual_ctr: Optional[float] = None,
        actual_views: Optional[int] = None,
        actual_impressions: Optional[int] = None,
        comments: Optional[str] = None,
        feedback_type: str = 'rating',
        request_ip: Optional[str] = None
    ) -> bool:
        """
        Log user feedback for training labels and quality assurance
        
        Args:
            analysis_id: UUID of the original analysis
            user_id: User providing feedback
            helpful: Was the analysis helpful?
            accurate: Was the score accurate?
            used_winner: Did they use the recommended thumbnail?
            actual_ctr: Actual CTR achieved (if available)
            actual_views: Actual view count
            actual_impressions: Actual impression count
            comments: Free-form comments
            feedback_type: Type of feedback (rating, performance, comment)
            request_ip: Client IP address
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.enabled or not self.client:
            return False
        
        try:
            row = {
                'analysis_id': analysis_id,
                'user_id': user_id,
                'helpful': helpful,
                'accurate': accurate,
                'used_winner': used_winner,
                'actual_ctr': float(actual_ctr) if actual_ctr is not None else None,
                'actual_views': int(actual_views) if actual_views is not None else None,
                'actual_impressions': int(actual_impressions) if actual_impressions is not None else None,
                'comments': comments,
                'feedback_type': feedback_type,
                'request_ip': request_ip
            }
            
            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}
            
            result = self.client.table('user_feedback').insert(row).execute()
            
            if result.data:
                logger.info(f"[ANALYTICS] Logged feedback for analysis {analysis_id}")
                return True
            else:
                logger.warning(f"[ANALYTICS] Failed to log feedback for analysis {analysis_id}")
                return False
                
        except Exception as e:
            logger.error(f"[ANALYTICS] Error logging feedback: {e}")
            return False
    
    def get_niche_analytics(self, niche: Optional[str] = None, days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Get analytics data for dashboard
        
        Args:
            niche: Specific niche to filter by (optional)
            days: Number of days to look back
            
        Returns:
            Analytics data dictionary or None if failed
        """
        
        if not self.enabled or not self.client:
            return None
        
        try:
            # Build query
            query = self.client.table('thumbnail_analyses').select('*')
            
            if niche:
                query = query.eq('niche', niche)
            
            # Filter by date range
            from datetime import datetime, timedelta
            since = (datetime.now() - timedelta(days=days)).isoformat()
            query = query.gte('created_at', since)
            
            result = query.execute()
            
            if result.data:
                # Calculate analytics
                data = result.data
                total_analyses = len(data)
                
                if total_analyses == 0:
                    return {'total_analyses': 0}
                
                # Basic metrics
                scores = [item['final_score'] for item in data if item['final_score']]
                confidences = [item['confidence'] for item in data if item['confidence']]
                processing_times = [item['processing_time_ms'] for item in data if item['processing_time_ms']]
                
                analytics = {
                    'total_analyses': total_analyses,
                    'unique_users': len(set(item['user_id'] for item in data if item['user_id'])),
                    'unique_sessions': len(set(item['session_id'] for item in data)),
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                    'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                    'face_detection_rate': sum(1 for item in data if item.get('face_detected')) / total_analyses,
                    'text_detection_rate': sum(1 for item in data if item.get('word_count', 0) > 0) / total_analyses,
                    'avg_word_count': sum(item.get('word_count', 0) for item in data) / total_analyses
                }
                
                return analytics
            else:
                return {'total_analyses': 0}
                
        except Exception as e:
            logger.error(f"[ANALYTICS] Error fetching analytics: {e}")
            return None
    
    def get_recent_analyses(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent analyses for monitoring"""
        
        if not self.enabled or not self.client:
            return []
        
        try:
            result = self.client.table('thumbnail_analyses').select('*').order('created_at', desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"[ANALYTICS] Error fetching recent analyses: {e}")
            return []
    
    def refresh_analytics(self) -> bool:
        """Refresh aggregated analytics (call periodically)"""
        
        if not self.enabled or not self.client:
            return False
        
        try:
            # Call the stored procedure to refresh niche analytics
            self.client.rpc('refresh_niche_analytics').execute()
            logger.info("[ANALYTICS] Refreshed aggregated analytics")
            return True
        except Exception as e:
            logger.warning(f"[ANALYTICS] Could not refresh analytics: {e}")
            return False

# Global instance
analytics_logger = AnalyticsLogger()