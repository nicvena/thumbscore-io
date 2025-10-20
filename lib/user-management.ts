// User Management and Usage Tracking System

export interface User {
  id: string;
  email?: string;
  tier: 'free' | 'creator' | 'pro';
  createdAt: Date;
  subscriptionStatus: 'active' | 'cancelled' | 'trial' | 'none';
  trialEndDate?: Date;
  stripeCustomerId?: string;
  stripeSubscriptionId?: string;
}

export interface UsageRecord {
  userId: string;
  month: string; // Format: YYYY-MM
  count: number;
  lastUpdated: Date;
}

export interface TierLimits {
  monthlyAnalyses: number;
  features: {
    basicScoring: boolean;
    advancedScoring: boolean;
    abTestingHistory: boolean;
    trendAnalysis: boolean;
    competitorBenchmarking: boolean;
    customNicheTraining: boolean;
    apiAccess: boolean;
    whiteLabelReports: boolean;
    prioritySupport: boolean;
  };
}

export const TIER_LIMITS: Record<User['tier'], TierLimits> = {
  free: {
    monthlyAnalyses: 1,
    features: {
      basicScoring: true,
      advancedScoring: false,
      abTestingHistory: false,
      trendAnalysis: false,
      competitorBenchmarking: false,
      customNicheTraining: false,
      apiAccess: false,
      whiteLabelReports: false,
      prioritySupport: false,
    }
  },
  creator: {
    monthlyAnalyses: 25,
    features: {
      basicScoring: true,
      advancedScoring: true,
      abTestingHistory: false,
      trendAnalysis: true,
      competitorBenchmarking: false,
      customNicheTraining: false,
      apiAccess: false,
      whiteLabelReports: false,
      prioritySupport: true,
    }
  },
  pro: {
    monthlyAnalyses: 100,
    features: {
      basicScoring: true,
      advancedScoring: true,
      abTestingHistory: true,
      trendAnalysis: true,
      competitorBenchmarking: true,
      customNicheTraining: true,
      apiAccess: true,
      whiteLabelReports: true,
      prioritySupport: true,
    }
  }
};

// Simple in-memory storage for demo purposes
// In production, use a proper database (PostgreSQL, MongoDB, etc.)
class InMemoryUserStore {
  private users: Map<string, User> = new Map();
  private usage: Map<string, UsageRecord> = new Map();

  // User management
  getUser(userId: string): User | null {
    return this.users.get(userId) || null;
  }

  createUser(user: Omit<User, 'id' | 'createdAt'>): User {
    const newUser: User = {
      ...user,
      id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date(),
    };
    this.users.set(newUser.id, newUser);
    return newUser;
  }

  updateUser(userId: string, updates: Partial<User>): User | null {
    const user = this.users.get(userId);
    if (!user) return null;
    
    const updatedUser = { ...user, ...updates };
    this.users.set(userId, updatedUser);
    return updatedUser;
  }

  // Usage tracking
  getCurrentMonthUsage(userId: string): number {
    const currentMonth = new Date().toISOString().slice(0, 7); // YYYY-MM
    const key = `${userId}:${currentMonth}`;
    const record = this.usage.get(key);
    return record?.count || 0;
  }

  incrementUsage(userId: string): number {
    const currentMonth = new Date().toISOString().slice(0, 7); // YYYY-MM
    const key = `${userId}:${currentMonth}`;
    const existing = this.usage.get(key);
    
    const newCount = (existing?.count || 0) + 1;
    const record: UsageRecord = {
      userId,
      month: currentMonth,
      count: newCount,
      lastUpdated: new Date(),
    };
    
    this.usage.set(key, record);
    return newCount;
  }

  // Check if user can make another analysis
  canUserAnalyze(userId: string): { canAnalyze: boolean; reason?: string; currentUsage?: number; limit?: number } {
    const user = this.getUser(userId);
    if (!user) {
      return { canAnalyze: false, reason: 'User not found' };
    }

    const currentUsage = this.getCurrentMonthUsage(userId);
    const limits = TIER_LIMITS[user.tier];
    
    // Unlimited for pro tier
    if (limits.monthlyAnalyses === -1) {
      return { canAnalyze: true, currentUsage };
    }

    // Check if under limit
    if (currentUsage < limits.monthlyAnalyses) {
      return { canAnalyze: true, currentUsage, limit: limits.monthlyAnalyses };
    }

    return { 
      canAnalyze: false, 
      reason: `Monthly limit of ${limits.monthlyAnalyses} analyses reached`, 
      currentUsage, 
      limit: limits.monthlyAnalyses 
    };
  }

  // Get user usage summary
  getUserUsageSummary(userId: string) {
    const user = this.getUser(userId);
    if (!user) return null;

    const currentUsage = this.getCurrentMonthUsage(userId);
    const limits = TIER_LIMITS[user.tier];
    
    return {
      tier: user.tier,
      currentUsage,
      monthlyLimit: limits.monthlyAnalyses,
      features: limits.features,
      canAnalyze: this.canUserAnalyze(userId).canAnalyze,
    };
  }
}

// Singleton instance
export const userStore = new InMemoryUserStore();

// Helper function to get or create anonymous user
export function getOrCreateAnonymousUser(sessionId?: string): User {
  // For demo purposes, create anonymous users
  // In production, you'd want proper session management
  const userId = sessionId || `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  let user = userStore.getUser(userId);
  if (!user) {
    user = userStore.createUser({
      tier: 'free',
      subscriptionStatus: 'none',
    });
  }
  
  return user;
}

// Rate limiting functions
export function checkRateLimit(userId: string): { allowed: boolean; remainingQuota?: number; resetDate?: Date } {
  const usage = userStore.canUserAnalyze(userId);
  
  if (!usage.canAnalyze) {
    const nextMonth = new Date();
    nextMonth.setMonth(nextMonth.getMonth() + 1);
    nextMonth.setDate(1);
    nextMonth.setHours(0, 0, 0, 0);
    
    return {
      allowed: false,
      resetDate: nextMonth,
    };
  }
  
  const remainingQuota = usage.limit === -1 ? -1 : (usage.limit! - (usage.currentUsage || 0));
  
  return {
    allowed: true,
    remainingQuota,
  };
}

// Subscription management helpers
export function upgradeTier(userId: string, newTier: User['tier'], stripeData?: { customerId: string; subscriptionId: string }): User | null {
  const updates: Partial<User> = {
    tier: newTier,
    subscriptionStatus: 'active',
  };

  if (stripeData) {
    updates.stripeCustomerId = stripeData.customerId;
    updates.stripeSubscriptionId = stripeData.subscriptionId;
  }

  return userStore.updateUser(userId, updates);
}

export function startTrial(userId: string, tier: User['tier']): User | null {
  const trialEndDate = new Date();
  trialEndDate.setDate(trialEndDate.getDate() + 14); // 14-day trial

  return userStore.updateUser(userId, {
    tier,
    subscriptionStatus: 'trial',
    trialEndDate,
  });
}