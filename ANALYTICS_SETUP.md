# Analytics Setup Guide for Thumbscore.io

## Overview
This application includes comprehensive analytics tracking to monitor user behavior, performance metrics, and conversion funnels.

## Features Included

### 📊 **Google Analytics 4 Integration**
- Page view tracking
- Custom event tracking
- User journey analysis
- Conversion funnel monitoring

### 🎯 **Custom Event Tracking**
- **User Journey**: Homepage visits, upload flow, results viewing
- **File Uploads**: File selection, upload progress, success/failure rates
- **Form Interactions**: Field interactions, form completion rates
- **Feature Usage**: Power words analysis, visual overlays, auto-fix clicks
- **Performance**: Page load times, scroll depth, time on page
- **Errors**: React errors, API errors, upload failures

### 🔧 **Analytics Components**
- `AnalyticsProvider`: Initializes Google Analytics
- `AnalyticsErrorBoundary`: Tracks React errors
- `useAnalytics`: React hook for easy tracking
- `useFormAnalytics`: Specialized form tracking
- `useUploadAnalytics`: File upload tracking

## Setup Instructions

### 1. Google Analytics 4 Setup

1. **Create GA4 Property**:
   - Go to [Google Analytics](https://analytics.google.com/)
   - Create a new property for your website
   - Choose "Web" as the platform

2. **Get Measurement ID**:
   - Go to Admin > Data Streams > Web
   - Copy the Measurement ID (starts with `G-`)

3. **Configure Environment Variables**:
   ```bash
   # Create .env.local file in your project root
   echo "NEXT_PUBLIC_GA_ID=G-YOUR-ACTUAL-ID" > .env.local
   ```

### 2. Verify Analytics Integration

1. **Check Console**: Look for GA initialization logs
2. **Test Events**: Use browser dev tools to verify events are firing
3. **Real-time Reports**: Check GA4 Real-time reports for incoming data

## Tracked Events

### 🏠 **Homepage Events**
- `homepage_visit`: When users land on homepage
- `cta_click`: When users click "Test Your Thumbnails" or "FAQ"

### 📤 **Upload Page Events**
- `form_start`: When upload form is initialized
- `file_selection`: When files are selected (count, total size)
- `form_field_interaction`: Field interactions (niche, title, files)
- `form_submit`: Form submission (success/failure)
- `upload_error`: Upload failures with error details

### 📊 **Results Page Events**
- `results_page_view`: When results page loads
- `analysis_complete`: When analysis finishes (score, niche, processing time)
- `power_words_analysis`: Power words feature usage
- `visual_overlay_usage`: Visual overlay interactions
- `auto_fix_click`: Auto-fix button clicks

### ⚡ **Performance Events**
- `page_load_time`: Page load performance
- `scroll_depth`: User scroll behavior (25%, 50%, 75%, 90%)
- `time_on_page`: Time spent on each page

### 🚨 **Error Events**
- `react_error`: React component errors
- `api_error`: API call failures
- `upload_error`: File upload errors

## Custom Analytics Implementation

### Basic Event Tracking
```typescript
import { useAnalytics } from '@/lib/hooks/useAnalytics';

const MyComponent = () => {
  const analytics = useAnalytics();
  
  const handleClick = () => {
    analytics.trackEvent('button_click', {
      event_category: 'engagement',
      event_label: 'my_button',
    });
  };
};
```

### Form Tracking
```typescript
import { useFormAnalytics } from '@/lib/hooks/useAnalytics';

const MyForm = () => {
  const formAnalytics = useFormAnalytics('my_form');
  
  useEffect(() => {
    formAnalytics.trackFormStart();
  }, []);
  
  const handleSubmit = (success: boolean) => {
    formAnalytics.trackFormSubmit(success);
  };
};
```

### Performance Tracking
```typescript
import { useAnalytics } from '@/lib/hooks/useAnalytics';

const MyComponent = () => {
  const analytics = useAnalytics();
  
  useEffect(() => {
    analytics.startTimer('my_operation');
    // ... perform operation
    analytics.endTimer('my_operation');
  }, []);
};
```

## Analytics Dashboard Setup

### 1. **Custom Reports in GA4**
- Create custom reports for key metrics
- Set up conversion goals
- Configure audience segments

### 2. **Key Metrics to Monitor**
- **Conversion Rate**: Upload form completion rate
- **User Engagement**: Time on page, scroll depth
- **Feature Usage**: Power words, visual overlays usage
- **Error Rates**: Upload failures, API errors
- **Performance**: Page load times, processing times

### 3. **Funnel Analysis**
- Homepage → Upload → Results conversion funnel
- Identify drop-off points
- Optimize user experience

## Privacy Considerations

### GDPR Compliance
- Analytics respects user privacy settings
- No personally identifiable information is tracked
- Users can opt-out via browser settings

### Data Retention
- GA4 default retention: 14 months
- Custom events: Follow GA4 retention policies
- No sensitive data is collected

## Troubleshooting

### Common Issues

1. **Events Not Appearing**:
   - Check GA4 Measurement ID is correct
   - Verify `.env.local` file exists
   - Check browser console for errors

2. **Real-time Data Not Showing**:
   - Wait 5-10 minutes for data to appear
   - Check GA4 Real-time reports
   - Verify events are firing in browser dev tools

3. **Performance Issues**:
   - Analytics loads asynchronously
   - Minimal impact on page performance
   - Use browser dev tools to monitor

### Debug Mode
```typescript
// Enable debug mode in development
if (process.env.NODE_ENV === 'development') {
  window.gtag('config', GA_TRACKING_ID, {
    debug_mode: true,
  });
}
```

## Advanced Features

### Custom Dimensions
- Track user niches
- Monitor file types and sizes
- Analyze processing times

### Enhanced Ecommerce
- Track "purchases" (successful analyses)
- Monitor "cart" (file uploads)
- Conversion value tracking

### Audience Building
- Create audiences based on behavior
- Retarget users who didn't complete upload
- Segment by niche preferences

---

## Support

For analytics setup help or custom tracking needs, refer to:
- [Google Analytics 4 Documentation](https://developers.google.com/analytics/devguides/collection/ga4)
- [Next.js Analytics Integration](https://nextjs.org/docs/advanced-features/measuring-performance)
- [React Analytics Best Practices](https://reactjs.org/docs/error-boundaries.html)
