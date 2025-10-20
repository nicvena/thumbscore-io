import { NextRequest, NextResponse } from 'next/server';
import { userStore, getOrCreateAnonymousUser, upgradeTier } from '@/lib/user-management';
import { getStripeCustomerByEmail } from '@/lib/auth';

export async function GET(request: NextRequest) {
  try {
    // Check if user is authenticated via session token
    const sessionToken = request.headers.get('x-session-token');
    let tier: 'free' | 'creator' | 'pro' = 'free';
    let subscriptionStatus: 'active' | 'cancelled' | 'trial' | 'none' = 'none';

    if (sessionToken) {
      try {
        // Verify session and get user info
        const verifyResponse = await fetch(`${request.nextUrl.origin}/api/auth/verify-session`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionToken })
        });

        if (verifyResponse.ok) {
          const sessionData = await verifyResponse.json();
          const email = sessionData.email;
          
          // Check Stripe subscription status
          const stripeData = await getStripeCustomerByEmail(email);
          
          if (stripeData && stripeData.status === 'active') {
            tier = stripeData.plan as 'creator' | 'pro';
            subscriptionStatus = 'active';
            
            // Update user in our system if they have a subscription
            const userId = `user_${email}`;
            const existingUser = userStore.getUser(userId);
            
            if (!existingUser || existingUser.tier !== tier) {
              upgradeTier(userId, tier, {
                customerId: stripeData.customerId,
                subscriptionId: stripeData.subscriptionId || ''
              });
            }
          }
        }
      } catch (error) {
        console.log('Session verification failed, using anonymous user');
      }
    }

    // Get or create user (use email-based ID for authenticated users)
    const sessionId = request.headers.get('x-session-id') || undefined;
    const userId = sessionToken ? `user_${sessionId}` : sessionId;
    
    if (!userStore.getUser(userId)) {
      userStore.createUser({
        id: userId,
        tier,
        subscriptionStatus,
        email: sessionToken ? 'authenticated@user.com' : undefined
      });
    }

    const user = userStore.getUser(userId);
    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    // Get usage summary
    const summary = userStore.getUserUsageSummary(userId);
    
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
      subscriptionStatus: user.subscriptionStatus,
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