import { NextResponse } from 'next/server'
import { verifyToken, createUserSession, getStripeCustomerByEmail } from '@/lib/auth'

export async function POST(req: Request) {
  try {
    const { token } = await req.json()

    if (!token) {
      return NextResponse.json({ error: 'Token is required' }, { status: 400 })
    }

    // Verify the magic link token
    const payload = await verifyToken(token)
    
    if (payload.type !== 'magic_link') {
      return NextResponse.json({ error: 'Invalid token type' }, { status: 400 })
    }

    const email = payload.email as string

    // Check Stripe subscription status (but allow sign-in regardless)
    const stripeData = await getStripeCustomerByEmail(email)
    
    // Create user session for all users (free and paid)
    const sessionToken = await createUserSession({
      email,
      stripeCustomerId: stripeData?.customerId,
      subscriptionStatus: (stripeData?.status as any) || 'none',
      plan: stripeData?.plan || 'free',
      expiresAt: Date.now() + (7 * 24 * 60 * 60 * 1000) // 7 days
    })

    return NextResponse.json({
      success: true,
      sessionToken,
      email,
      plan: stripeData?.plan || 'free',
      subscriptionStatus: stripeData?.status || 'none'
    })

  } catch (error) {
    console.error('Error verifying magic link:', error)
    return NextResponse.json({ 
      error: 'Invalid or expired token',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 400 })
  }
}
