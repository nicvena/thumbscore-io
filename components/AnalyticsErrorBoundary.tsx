'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { errorAnalytics } from '@/lib/analytics';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class AnalyticsErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Track the error with analytics
    try {
      errorAnalytics.trackReactError(error, errorInfo);
    } catch (analyticsError) {
      console.warn('Failed to track error with analytics:', analyticsError);
    }
    
    console.error('AnalyticsErrorBoundary caught an error:', error, errorInfo);
    
    // Log additional debugging information
    console.error('Error details:', {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
    });
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex items-center justify-center p-8">
          <div className="bg-white/5 backdrop-blur-sm rounded-lg p-8 max-w-md mx-auto border border-white/10 text-center">
            <div className="text-6xl mb-4">⚠️</div>
            <h2 className="text-2xl font-bold text-white mb-4">Something went wrong</h2>
            <p className="text-gray-300 mb-6">
              We've encountered an unexpected error. Our team has been notified and is working to fix it.
            </p>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="mb-6 p-4 bg-red-900/20 border border-red-500/30 rounded-lg text-left">
                <p className="text-red-300 text-sm font-mono">
                  <strong>Error:</strong> {this.state.error.message}
                </p>
                <details className="mt-2">
                  <summary className="text-red-300 text-xs cursor-pointer">Stack trace</summary>
                  <pre className="text-red-300 text-xs mt-2 whitespace-pre-wrap">
                    {this.state.error.stack}
                  </pre>
                </details>
              </div>
            )}
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all font-semibold"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default AnalyticsErrorBoundary;
