import { NextResponse } from 'next/server'
import { verifyToken, getStripeCustomerByEmail } from '@/lib/auth'

export async function POST(req: Request) {
  try {
    const { sessionToken } = await req.json()

    if (!sessionToken) {
      return NextResponse.json({ error: 'Session token is required' }, { status: 400 })
    }

    // Verify the session token
    const payload = await verifyToken(sessionToken)
    const email = payload.email as string

    // Check current Stripe subscription status
    const stripeData = await getStripeCustomerByEmail(email)
    
    if (!stripeData || stripeData.status !== 'active') {
      return NextResponse.json({ 
        error: 'Subscription no longer active' 
      }, { status: 403 })
    }

    return NextResponse.json({
      success: true,
      email,
      plan: stripeData.plan,
      subscriptionStatus: stripeData.status
    })

  } catch (error) {
    console.error('Error verifying session:', error)
    return NextResponse.json({ 
      error: 'Invalid or expired session' 
    }, { status: 401 })
  }
}
