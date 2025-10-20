import Stripe from 'stripe'

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2025-09-30.clover'
})

export async function createCheckoutSession(priceId: string, customerEmail?: string) {
  const session = await stripe.checkout.sessions.create({
    mode: 'subscription',
    payment_method_types: ['card'],
    line_items: [{ price: priceId, quantity: 1 }],
    success_url: `${process.env.NEXT_PUBLIC_APP_URL}/success`,
    cancel_url: `${process.env.NEXT_PUBLIC_APP_URL}/pricing`,
    customer_email: customerEmail,
    // Collect email if not provided
    customer_creation: 'always',
    // Add metadata for tracking
    metadata: {
      source: 'upgrade_prompt',
      price_id: priceId
    }
  })

  return session
}
