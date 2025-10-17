'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface UsageInfo {
  tier: 'free' | 'creator' | 'pro';
  currentUsage: number;
  monthlyLimit: number;
  canAnalyze: boolean;
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
      // Get usage info from API (we'll create this endpoint)
      const response = await fetch('/api/usage', {
        headers: {
          'X-Session-Id': getSessionId(),
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setUsage(data);
      }
    } catch (error) {
      console.error('Failed to fetch usage info:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSessionId = () => {
    // Simple session ID generation for demo
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
    <div className={`bg-white/5 rounded-lg p-4 backdrop-blur-sm border border-white/10 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-white">
          {usage.tier.charAt(0).toUpperCase() + usage.tier.slice(1)} Plan
        </h3>
        {usage.tier === 'free' && (
          <Link href="/pricing" className="text-xs text-blue-400 hover:text-blue-300">
            Upgrade
          </Link>
        )}
      </div>
      
      {usage.monthlyLimit === -1 ? (
        <div className="text-sm text-green-400">
          ✓ Unlimited analyses
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm text-gray-300">Monthly Usage</span>
            <span className={`text-sm font-medium ${getUsageColor()}`}>
              {usage.currentUsage} / {usage.monthlyLimit}
            </span>
          </div>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${getUsageBarColor()}`}
              style={{ width: `${getProgressPercentage()}%` }}
            ></div>
          </div>
          
          {!usage.canAnalyze && (
            <div className="text-xs text-red-400 mb-2">
              ⚠️ Monthly limit reached
            </div>
          )}
          
          {usage.tier === 'free' && usage.currentUsage >= usage.monthlyLimit * 0.8 && (
            <div className="text-xs text-yellow-400 mb-2">
              ⚡ Running low on analyses
            </div>
          )}
        </>
      )}
      
      {usage.tier === 'free' && (
        <Link 
          href="/pricing" 
          className="block text-center text-xs bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white py-2 px-3 rounded mt-2 hover:opacity-90 transition-opacity"
        >
          Get More Analyses
        </Link>
      )}
    </div>
  );
}

// Hook for usage tracking in other components
export function useUsageTracking() {
  const [usage, setUsage] = useState<UsageInfo | null>(null);

  const refreshUsage = async () => {
    try {
      const sessionId = localStorage.getItem('session-id') || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('session-id', sessionId);
      
      const response = await fetch('/api/usage', {
        headers: {
          'X-Session-Id': sessionId,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setUsage(data);
        return data;
      }
    } catch (error) {
      console.error('Failed to refresh usage:', error);
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