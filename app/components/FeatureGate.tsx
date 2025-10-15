'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { TIER_LIMITS } from '@/lib/user-management';

interface FeatureGateProps {
  feature: keyof typeof TIER_LIMITS.free.features;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  className?: string;
}

interface UserTier {
  tier: 'free' | 'creator' | 'pro';
  features: Record<string, boolean>;
}

export function FeatureGate({ feature, children, fallback, className = '' }: FeatureGateProps) {
  const [userTier, setUserTier] = useState<UserTier | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUserTier();
  }, []);

  const fetchUserTier = async () => {
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
        setUserTier({
          tier: data.tier,
          features: data.features,
        });
      }
    } catch (error) {
      console.error('Failed to fetch user tier:', error);
      // Default to free tier on error
      setUserTier({
        tier: 'free',
        features: TIER_LIMITS.free.features,
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className={`animate-pulse bg-white/5 rounded-lg p-4 ${className}`}>
        <div className="h-4 bg-white/20 rounded w-3/4"></div>
      </div>
    );
  }

  if (!userTier) {
    return null;
  }

  const hasFeature = userTier.features[feature];

  if (hasFeature) {
    return <div className={className}>{children}</div>;
  }

  if (fallback) {
    return <div className={className}>{fallback}</div>;
  }

  // Default upgrade prompt
  return (
    <div className={`bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-500/30 rounded-lg p-6 text-center ${className}`}>
      <div className="text-3xl mb-4">🔒</div>
      <h3 className="text-lg font-semibold text-white mb-2">
        {getFeatureName(feature)} - Premium Feature
      </h3>
      <p className="text-gray-300 mb-4">
        Upgrade to unlock {getFeatureName(feature).toLowerCase()} and boost your thumbnail performance.
      </p>
      <div className="flex gap-3 justify-center">
        <Link
          href="/pricing"
          className="px-6 py-2 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all font-semibold"
        >
          Upgrade Now
        </Link>
        <button 
          onClick={fetchUserTier}
          className="px-6 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
        >
          Refresh
        </button>
      </div>
      <div className="mt-4 text-xs text-gray-400">
        Available in {getRequiredTier(feature)} plan and above
      </div>
    </div>
  );
}

function getFeatureName(feature: keyof typeof TIER_LIMITS.free.features): string {
  const names: Record<typeof feature, string> = {
    basicScoring: 'Basic Scoring',
    advancedScoring: 'Advanced Scoring',
    abTestingHistory: 'A/B Testing History',
    trendAnalysis: 'Trend Analysis',
    competitorBenchmarking: 'Competitor Benchmarking',
    customNicheTraining: 'Custom Niche Training',
    apiAccess: 'API Access',
    whiteLabelReports: 'White-label Reports',
    prioritySupport: 'Priority Support',
  };
  return names[feature] || feature;
}

function getRequiredTier(feature: keyof typeof TIER_LIMITS.free.features): string {
  // Check which tier first includes this feature
  if (TIER_LIMITS.creator.features[feature]) return 'Creator';
  if (TIER_LIMITS.pro.features[feature]) return 'Pro';
  return 'Free';
}

// Convenience components for specific features
export function AdvancedScoringGate({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <FeatureGate feature="advancedScoring" className={className}>
      {children}
    </FeatureGate>
  );
}

export function ABTestingGate({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <FeatureGate feature="abTestingHistory" className={className}>
      {children}
    </FeatureGate>
  );
}

export function TrendAnalysisGate({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <FeatureGate feature="trendAnalysis" className={className}>
      {children}
    </FeatureGate>
  );
}

export function CompetitorBenchmarkingGate({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <FeatureGate feature="competitorBenchmarking" className={className}>
      {children}
    </FeatureGate>
  );
}

export function APIAccessGate({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <FeatureGate feature="apiAccess" className={className}>
      {children}
    </FeatureGate>
  );
}