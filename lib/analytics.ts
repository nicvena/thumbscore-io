// Analytics configuration and tracking utilities
declare global {
  interface Window {
    gtag: (...args: any[]) => void;
    dataLayer: any[];
  }
}

// Google Analytics 4 Configuration
export const GA_TRACKING_ID = process.env.NEXT_PUBLIC_GA_ID || 'G-XXXXXXXXXX';

// Initialize Google Analytics
export const initGA = () => {
  if (typeof window !== 'undefined' && GA_TRACKING_ID !== 'G-XXXXXXXXXX') {
    try {
      // Load Google Analytics script
      const script = document.createElement('script');
      script.async = true;
      script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_TRACKING_ID}`;
      document.head.appendChild(script);

      // Initialize gtag
      window.dataLayer = window.dataLayer || [];
      window.gtag = function() {
        window.dataLayer.push(arguments);
      };
      window.gtag('js', new Date());
      window.gtag('config', GA_TRACKING_ID, {
        page_title: document.title,
        page_location: window.location.href,
      });
    } catch (error) {
      console.warn('Failed to initialize Google Analytics:', error);
    }
  } else {
    console.log('Google Analytics not configured - set NEXT_PUBLIC_GA_ID environment variable');
  }
};

// Track page views
export const trackPageView = (url: string, title?: string) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', GA_TRACKING_ID, {
      page_path: url,
      page_title: title || document.title,
    });
  }
};

// Track custom events
export const trackEvent = (eventName: string, parameters?: Record<string, any>) => {
  if (typeof window !== 'undefined' && window.gtag) {
    try {
      window.gtag('event', eventName, {
        event_category: 'user_interaction',
        ...parameters,
      });
    } catch (error) {
      console.warn('Failed to track event:', eventName, error);
    }
  }
};

// Thumbscore-specific event tracking
export const analytics = {
  // User journey tracking
  trackHomepageVisit: () => {
    try {
      trackEvent('homepage_visit', {
        event_category: 'engagement',
        event_label: 'homepage',
      });
    } catch (error) {
      console.warn('Failed to track homepage visit:', error);
    }
  },

  trackUploadStart: (niche: string) => {
    try {
      trackEvent('upload_start', {
        event_category: 'conversion',
        event_label: 'upload_flow',
        custom_parameter_1: niche,
      });
    } catch (error) {
      console.warn('Failed to track upload start:', error);
    }
  },

  trackThumbnailUpload: (fileCount: number, niche: string, fileSizes: number[]) => {
    try {
      trackEvent('thumbnail_upload', {
        event_category: 'conversion',
        event_label: 'file_upload',
        custom_parameter_1: niche,
        custom_parameter_2: fileCount,
        custom_parameter_3: Math.round(fileSizes.reduce((a, b) => a + b, 0) / 1024 / 1024), // Total MB
      });
    } catch (error) {
      console.warn('Failed to track thumbnail upload:', error);
    }
  },

  trackAnalysisComplete: (results: {
    winnerScore: number;
    niche: string;
    processingTime: number;
    thumbnailCount: number;
  }) => {
    try {
      trackEvent('analysis_complete', {
        event_category: 'conversion',
        event_label: 'analysis_success',
        custom_parameter_1: results.niche,
        custom_parameter_2: results.thumbnailCount,
        custom_parameter_3: Math.round(results.processingTime),
        value: results.winnerScore,
      });
    } catch (error) {
      console.warn('Failed to track analysis complete:', error);
    }
  },

  trackResultsPageView: (winnerScore: number, niche: string) => {
    trackEvent('results_page_view', {
      event_category: 'engagement',
      event_label: 'results_view',
      custom_parameter_1: niche,
      value: winnerScore,
    });
  },

  trackPowerWordsAnalysis: (score: number, wordCount: number, niche: string) => {
    trackEvent('power_words_analysis', {
      event_category: 'feature_usage',
      event_label: 'power_words',
      custom_parameter_1: niche,
      custom_parameter_2: wordCount,
      value: score,
    });
  },

  trackVisualOverlayUsage: (overlayType: string, niche: string) => {
    trackEvent('visual_overlay_usage', {
      event_category: 'feature_usage',
      event_label: 'visual_overlays',
      custom_parameter_1: overlayType,
      custom_parameter_2: niche,
    });
  },

  trackAutoFixClick: (issueType: string, niche: string) => {
    trackEvent('auto_fix_click', {
      event_category: 'feature_usage',
      event_label: 'auto_fix',
      custom_parameter_1: issueType,
      custom_parameter_2: niche,
    });
  },

  trackNicheSelectorUsage: (niche: string) => {
    trackEvent('niche_selection', {
      event_category: 'user_preference',
      event_label: 'niche_choice',
      custom_parameter_1: niche,
    });
  },

  // Error tracking
  trackError: (errorType: string, errorMessage: string, context?: string) => {
    trackEvent('error_occurred', {
      event_category: 'error',
      event_label: errorType,
      custom_parameter_1: errorMessage,
      custom_parameter_2: context,
    });
  },

  // Performance tracking
  trackPerformance: (metric: string, value: number, context?: string) => {
    trackEvent('performance_metric', {
      event_category: 'performance',
      event_label: metric,
      custom_parameter_1: context,
      value: Math.round(value),
    });
  },

  // User engagement tracking
  trackTimeOnPage: (page: string, timeSpent: number) => {
    trackEvent('time_on_page', {
      event_category: 'engagement',
      event_label: page,
      value: Math.round(timeSpent),
    });
  },

  trackScrollDepth: (page: string, depth: number) => {
    trackEvent('scroll_depth', {
      event_category: 'engagement',
      event_label: page,
      value: depth,
    });
  },

  // Conversion tracking
  trackConversion: (conversionType: string, value?: number, niche?: string) => {
    trackEvent('conversion', {
      event_category: 'conversion',
      event_label: conversionType,
      custom_parameter_1: niche,
      value: value,
    });
  },
};

// Performance monitoring utilities
export const performanceMonitor = {
  startTimer: (label: string) => {
    if (typeof window !== 'undefined' && window.performance) {
      window.performance.mark(`${label}-start`);
    }
  },

  endTimer: (label: string) => {
    if (typeof window !== 'undefined' && window.performance) {
      window.performance.mark(`${label}-end`);
      window.performance.measure(label, `${label}-start`, `${label}-end`);
      
      const measure = window.performance.getEntriesByName(label)[0];
      if (measure) {
        analytics.trackPerformance(label, measure.duration);
      }
    }
  },

  measurePageLoad: () => {
    if (typeof window !== 'undefined' && window.performance) {
      window.addEventListener('load', () => {
        const navigation = window.performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (navigation) {
          analytics.trackPerformance('page_load_time', navigation.loadEventEnd - navigation.fetchStart);
          analytics.trackPerformance('dom_content_loaded', navigation.domContentLoadedEventEnd - navigation.fetchStart);
          analytics.trackPerformance('first_paint', navigation.responseEnd - navigation.fetchStart);
        }
      });
    }
  },
};

// Error boundary analytics
export const errorAnalytics = {
  trackReactError: (error: Error, errorInfo: any) => {
    analytics.trackError('react_error', error.message, errorInfo.componentStack);
  },

  trackApiError: (endpoint: string, status: number, message: string) => {
    analytics.trackError('api_error', `${endpoint}: ${status} - ${message}`, 'api');
  },

  trackUploadError: (error: string, fileType?: string) => {
    analytics.trackError('upload_error', error, fileType);
  },
};

// User session tracking
export const sessionTracker = {
  startSession: () => {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('thumbscore_session_id', sessionId);
    return sessionId;
  },

  getSessionId: () => {
    return sessionStorage.getItem('thumbscore_session_id');
  },

  trackSessionEvent: (eventType: string, data?: any) => {
    const sessionId = sessionTracker.getSessionId();
    if (sessionId) {
      trackEvent('session_event', {
        event_category: 'session',
        event_label: eventType,
        custom_parameter_1: sessionId,
        ...data,
      });
    }
  },
};

export default analytics;
