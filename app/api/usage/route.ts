import { NextRequest, NextResponse } from 'next/server';
import { userStore, getOrCreateAnonymousUser } from '@/lib/user-management';

export async function GET(request: NextRequest) {
  try {
    // Get user session
    const sessionId = request.headers.get('x-session-id') || undefined;
    const user = getOrCreateAnonymousUser(sessionId);
    
    // Get usage summary
    const summary = userStore.getUserUsageSummary(user.id);
    
    if (!summary) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      tier: summary.tier,
      currentUsage: summary.currentUsage,
      monthlyLimit: summary.monthlyLimit,
      canAnalyze: summary.canAnalyze,
      features: summary.features,
    });

  } catch (error) {
    console.error('[Usage API] Error:', error);
    return NextResponse.json(
      { 
        error: 'Failed to fetch usage info',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}