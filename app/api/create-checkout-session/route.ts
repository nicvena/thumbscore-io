import { NextResponse } from 'next/server'
import { createCheckoutSession } from '@/lib/stripe'

export async function POST(req: Request) {
  try {
    const { priceId, email } = await req.json()

    if (!priceId) {
      return NextResponse.json({ error: 'Price ID is required' }, { status: 400 })
    }

    const session = await createCheckoutSession(priceId, email)

    return NextResponse.json({ url: session.url })
  } catch (error) {
    console.error('Error creating checkout session:', error)
    return NextResponse.json({ error: 'Failed to create checkout session' }, { status: 500 })
  }
}
