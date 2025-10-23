import { NextRequest, NextResponse } from 'next/server';
import { userStore, getOrCreateAnonymousUser, upgradeTier, User } from '@/lib/user-management';
import { getStripeCustomerByEmail } from '@/lib/auth';

export async function GET(request: NextRequest) {
  try {
    // Admin override check
    const adminOverride = request.headers.get('x-admin-override');
    if (adminOverride === 'true') {
      return NextResponse.json({
        tier: 'pro',
        currentUsage: 0,
        monthlyLimit: -1, // Unlimited
        canAnalyze: true,
        features: {
          basicScoring: true,
          advancedScoring: true,
          abTestingHistory: true,
          trendAnalysis: true,
          competitorBenchmarking: true,
          customNicheTraining: true,
          apiAccess: true
        },
        subscriptionStatus: 'active',
      });
    }

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
                subscriptionId: (stripeData.subscription as any)?.id || ''
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
    
    // Get or create user with specific ID
    let currentUser = userId ? userStore.getUser(userId) : null;
    if (!currentUser && userId) {
      // Create user with specific ID by directly adding to the store
      const newUser: User = {
        id: userId,
        tier,
        subscriptionStatus,
        email: sessionToken ? 'authenticated@user.com' : undefined,
        createdAt: new Date(),
      };
      // Access the private users map directly (we need to modify the class)
      (userStore as any).users.set(userId, newUser);
      currentUser = newUser;
    }
    if (!currentUser) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    // Get usage summary
    const summary = userId ? userStore.getUserUsageSummary(userId) : null;
    
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
      subscriptionStatus: currentUser.subscriptionStatus,
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