'use client';

import { useEffect, useRef, useCallback } from 'react';
import { usePathname } from 'next/navigation';
import { analytics, trackPageView, trackEvent, performanceMonitor, sessionTracker } from '@/lib/analytics';

// Custom hook for analytics tracking
export const useAnalytics = () => {
  const pathname = usePathname();
  const pageLoadTimeRef = useRef<number>(Date.now());
  const scrollDepthRef = useRef<number>(0);
  const maxScrollDepthRef = useRef<number>(0);

  // Track page views
  useEffect(() => {
    trackPageView(pathname);
    pageLoadTimeRef.current = Date.now();
    
    // Track performance metrics
    performanceMonitor.measurePageLoad();
    
    // Start session if not exists
    if (!sessionTracker.getSessionId()) {
      sessionTracker.startSession();
    }
  }, [pathname]);

  // Track scroll depth
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollPercent = Math.round((scrollTop / docHeight) * 100);
      
      if (scrollPercent > maxScrollDepthRef.current) {
        maxScrollDepthRef.current = scrollPercent;
        scrollDepthRef.current = scrollPercent;
        
        // Track milestone scroll depths
        if (scrollPercent >= 25 && scrollPercent < 50) {
          analytics.trackScrollDepth(pathname, 25);
        } else if (scrollPercent >= 50 && scrollPercent < 75) {
          analytics.trackScrollDepth(pathname, 50);
        } else if (scrollPercent >= 75 && scrollPercent < 90) {
          analytics.trackScrollDepth(pathname, 75);
        } else if (scrollPercent >= 90) {
          analytics.trackScrollDepth(pathname, 90);
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [pathname]);

  // Track time on page when component unmounts
  useEffect(() => {
    return () => {
      const timeSpent = Date.now() - pageLoadTimeRef.current;
      analytics.trackTimeOnPage(pathname, timeSpent);
    };
  }, [pathname]);

  // Return analytics functions for manual tracking
  return {
    trackEvent: trackEvent,
    trackUploadStart: analytics.trackUploadStart,
    trackThumbnailUpload: analytics.trackThumbnailUpload,
    trackAnalysisComplete: analytics.trackAnalysisComplete,
    trackResultsPageView: analytics.trackResultsPageView,
    trackPowerWordsAnalysis: analytics.trackPowerWordsAnalysis,
    trackVisualOverlayUsage: analytics.trackVisualOverlayUsage,
    trackAutoFixClick: analytics.trackAutoFixClick,
    trackNicheSelectorUsage: analytics.trackNicheSelectorUsage,
    trackError: analytics.trackError,
    trackConversion: analytics.trackConversion,
    startTimer: performanceMonitor.startTimer,
    endTimer: performanceMonitor.endTimer,
  };
};

// Hook for tracking specific page interactions
export const usePageAnalytics = (pageName: string) => {
  const analytics = useAnalytics();

  const trackPageSpecificEvent = useCallback((eventName: string, data?: any) => {
    analytics.trackEvent(`${pageName}_${eventName}`, {
      event_category: 'page_interaction',
      event_label: pageName,
      ...data,
    });
  }, [analytics, pageName]);

  return {
    ...analytics,
    trackPageSpecificEvent,
  };
};

// Hook for tracking form interactions
export const useFormAnalytics = (formName: string) => {
  const analytics = useAnalytics();

  const trackFormStart = useCallback(() => {
    analytics.trackEvent('form_start', {
      event_category: 'form_interaction',
      event_label: formName,
    });
  }, [analytics, formName]);

  const trackFormSubmit = useCallback((success: boolean, errorMessage?: string) => {
    analytics.trackEvent('form_submit', {
      event_category: 'form_interaction',
      event_label: formName,
      custom_parameter_1: success ? 'success' : 'error',
      custom_parameter_2: errorMessage,
    });
  }, [analytics, formName]);

  const trackFormFieldInteraction = useCallback((fieldName: string, action: string) => {
    analytics.trackEvent('form_field_interaction', {
      event_category: 'form_interaction',
      event_label: formName,
      custom_parameter_1: fieldName,
      custom_parameter_2: action,
    });
  }, [analytics, formName]);

  return {
    trackFormStart,
    trackFormSubmit,
    trackFormFieldInteraction,
  };
};

// Hook for tracking file uploads
export const useUploadAnalytics = () => {
  const analytics = useAnalytics();

  const trackFileSelection = useCallback((fileCount: number, totalSize: number) => {
    analytics.trackEvent('file_selection', {
      event_category: 'file_upload',
      custom_parameter_1: fileCount,
      custom_parameter_2: Math.round(totalSize / 1024 / 1024), // MB
    });
  }, [analytics]);

  const trackUploadProgress = useCallback((progress: number, fileCount: number) => {
    analytics.trackEvent('upload_progress', {
      event_category: 'file_upload',
      custom_parameter_1: fileCount,
      value: Math.round(progress),
    });
  }, [analytics]);

  const trackUploadError = useCallback((error: string, fileType?: string) => {
    analytics.trackError('upload_error', error, fileType);
  }, [analytics]);

  return {
    trackFileSelection,
    trackUploadProgress,
    trackUploadError,
  };
};

export default useAnalytics;
