'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface UsageInfo {
  tier: 'free' | 'creator' | 'pro';
  currentUsage: number;
  monthlyLimit: number;
  canAnalyze: boolean;
  subscriptionStatus: 'active' | 'cancelled' | 'trial' | 'none';
}

interface UsageTrackerProps {
  className?: string;
}

export function UsageTracker({ className = '' }: UsageTrackerProps) {
  const [usage, setUsage] = useState<UsageInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsageInfo();
  }, []);

  const fetchUsageInfo = async () => {
    try {
      const sessionToken = localStorage.getItem('thumbscore_user_session');
      const sessionId = getSessionId();
      const adminOverride = localStorage.getItem('thumbscore_admin_override');
      
      const headers: Record<string, string> = {
        'X-Session-Id': sessionId,
      };
      
      if (sessionToken) {
        headers['X-Session-Token'] = sessionToken;
      }
      
      if (adminOverride === 'true') {
        headers['X-Admin-Override'] = 'true';
      }

      const response = await fetch('/api/usage', { headers });
      
      if (response.ok) {
        const data = await response.json();
        setUsage(data);
      } else {
        console.warn(`Usage API returned ${response.status}, using default limits`);
        // Fallback to default usage to prevent blocking the app
        setUsage({
          tier: 'free',
          currentUsage: 0,
          monthlyLimit: 3,
          canAnalyze: true,
          features: {
            basicScoring: true,
            advancedScoring: false,
            abTestingHistory: false,
            trendAnalysis: false,
            competitorBenchmarking: false,
            customNicheTraining: false,
            apiAccess: false
          },
          subscriptionStatus: 'none'
        });
      }
    } catch (error) {
      console.warn('Failed to fetch usage info, using fallback:', error);
      // Fallback to default usage to prevent blocking the app
      setUsage({
        tier: 'free',
        currentUsage: 0,
        monthlyLimit: 3,
        canAnalyze: true,
        features: {
          basicScoring: true,
          advancedScoring: false,
          abTestingHistory: false,
          trendAnalysis: false,
          competitorBenchmarking: false,
          customNicheTraining: false,
          apiAccess: false
        },
        subscriptionStatus: 'none'
      });
    } finally {
      setLoading(false);
    }
  };

  const getSessionId = () => {
    let sessionId = localStorage.getItem('session-id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('session-id', sessionId);
    }
    return sessionId;
  };

  const getUsageColor = () => {
    if (!usage || usage.monthlyLimit === -1) return 'text-green-400';
    
    const percentage = (usage.currentUsage / usage.monthlyLimit) * 100;
    if (percentage >= 90) return 'text-red-400';
    if (percentage >= 70) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getUsageBarColor = () => {
    if (!usage || usage.monthlyLimit === -1) return 'bg-green-500';
    
    const percentage = (usage.currentUsage / usage.monthlyLimit) * 100;
    if (percentage >= 90) return 'bg-red-500';
    if (percentage >= 70) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getProgressPercentage = () => {
    if (!usage || usage.monthlyLimit === -1) return 0;
    return Math.min(100, (usage.currentUsage / usage.monthlyLimit) * 100);
  };

  if (loading) {
    return (
      <div className={`bg-white/5 rounded-lg p-4 backdrop-blur-sm border border-white/10 ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-white/20 rounded w-3/4 mb-2"></div>
          <div className="h-2 bg-white/20 rounded w-full"></div>
        </div>
      </div>
    );
  }

  if (!usage) return null;

  return (
    <div className={`bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-xl p-6 border border-gray-700/50 ${className}`}>
      <div className="text-center">
        <div className="flex items-center justify-center mb-3">
          <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-400 rounded-full flex items-center justify-center mr-3">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-white">
            {usage.tier.charAt(0).toUpperCase() + usage.tier.slice(1)} Plan
          </h3>
        </div>
        
        {usage.monthlyLimit === -1 ? (
          <div className="text-green-400 font-medium">
            ✓ Unlimited analyses
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-2xl font-bold text-white">
              {usage.currentUsage} / {usage.monthlyLimit}
            </div>
            <div className="text-sm text-gray-400">analyses this month</div>
            
            {/* Simplified Progress Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-500 ${getUsageBarColor()}`}
                style={{ width: `${getProgressPercentage()}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Upgrade prompt for free users */}
        {usage.tier === 'free' && usage.subscriptionStatus === 'none' && (
          <div className="mt-4 pt-4 border-t border-gray-600">
            <p className="text-sm text-gray-300 mb-3">
              Want more analyses?
            </p>
            <Link
              href="/pricing"
              className="inline-block px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-semibold text-sm transition-all duration-200"
            >
              Upgrade to Creator
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}

// Hook for usage tracking in other components
export function useUsageTracking() {
  const [usage, setUsage] = useState<UsageInfo | null>(null);

  const refreshUsage = async () => {
    try {
      const sessionToken = localStorage.getItem('thumbscore_user_session');
      const sessionId = localStorage.getItem('session-id') || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const adminOverride = localStorage.getItem('thumbscore_admin_override');
      localStorage.setItem('session-id', sessionId);
      
      const headers: Record<string, string> = {
        'X-Session-Id': sessionId,
      };
      
      if (sessionToken) {
        headers['X-Session-Token'] = sessionToken;
      }
      
      if (adminOverride === 'true') {
        headers['X-Admin-Override'] = 'true';
      }

      const response = await fetch('/api/usage', { headers });
      
      if (response.ok) {
        const data = await response.json();
        setUsage(data);
        return data;
      } else {
        console.warn(`Usage refresh failed with ${response.status}, keeping current state`);
      }
    } catch (error) {
      console.warn('Failed to refresh usage, keeping current state:', error);
    }
    return null;
  };

  const checkCanAnalyze = () => {
    return usage?.canAnalyze ?? true;
  };

  useEffect(() => {
    refreshUsage();
  }, []); // Remove refreshUsage dependency to prevent infinite re-renders

  return {
    usage,
    refreshUsage,
    checkCanAnalyze,
  };
}