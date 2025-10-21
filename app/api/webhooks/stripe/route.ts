import { NextResponse } from 'next/server'
import Stripe from 'stripe'
import { addToEmailList } from '@/lib/email'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-06-20'
})

export async function POST(req: Request) {
  const body = await req.text()
  const signature = req.headers.get('stripe-signature')!

  let event

  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET!
    )
  } catch (err) {
    console.error('Webhook signature verification failed:', err)
    return NextResponse.json({ error: 'Invalid signature' }, { status: 400 })
  }

  // Handle successful subscription
  if (event.type === 'checkout.session.completed') {
    const session = event.data.object as any
    
    if (session.customer_email) {
      // Add customer to email list
      await addToEmailList(session.customer_email)
      console.log('Customer added to email list:', session.customer_email)
    }
  }

  // Handle subscription created
  if (event.type === 'customer.subscription.created') {
    const subscription = event.data.object as any
    console.log('New subscription created:', subscription.id)
  }

  return NextResponse.json({ received: true })
}
