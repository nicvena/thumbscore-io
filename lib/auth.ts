import { SignJWT, jwtVerify } from 'jose'
import { v4 as uuidv4 } from 'uuid'

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || 'your-secret-key-change-in-production')

export interface UserSession {
  email: string
  stripeCustomerId?: string
  subscriptionStatus?: 'active' | 'canceled' | 'past_due' | 'incomplete'
  plan?: 'creator' | 'pro'
  expiresAt: number
}

export async function createMagicLinkToken(email: string): Promise<string> {
  const token = await new SignJWT({ 
    email,
    type: 'magic_link',
    jti: uuidv4() // Unique token ID
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('15m') // Magic links expire in 15 minutes
    .sign(JWT_SECRET)

  return token
}

export async function createUserSession(userData: UserSession): Promise<string> {
  const token = await new SignJWT(userData as any)
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('7d') // Sessions last 7 days
    .sign(JWT_SECRET)

  return token
}

export async function verifyToken(token: string): Promise<any> {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET)
    return payload
  } catch (error) {
    throw new Error('Invalid or expired token')
  }
}

export async function getStripeCustomerByEmail(email: string) {
  // Admin access for nicvenettacci@gmail.com
  if (email === 'nicvenettacci@gmail.com') {
    return {
      customerId: 'admin_user',
      subscription: { status: 'active' },
      plan: 'pro' as const,
      status: 'active'
    }
  }

  const stripe = (await import('stripe')).default
  const stripeInstance = new stripe(process.env.STRIPE_SECRET_KEY!, {
    apiVersion: '2025-09-30.clover'
  })

  try {
    // Search for customer by email
    const customers = await stripeInstance.customers.list({
      email: email,
      limit: 1
    })

    if (customers.data.length === 0) {
      return null
    }

    const customer = customers.data[0]
    
    // Get active subscriptions
    const subscriptions = await stripeInstance.subscriptions.list({
      customer: customer.id,
      status: 'active',
      limit: 1
    })

    const activeSubscription = subscriptions.data[0]
    
    return {
      customerId: customer.id,
      subscription: activeSubscription,
      plan: activeSubscription ? getPlanFromPriceId(activeSubscription.items.data[0].price.id) : null,
      status: activeSubscription?.status || 'none'
    }
  } catch (error) {
    console.error('Error fetching Stripe customer:', error)
    return null
  }
}

function getPlanFromPriceId(priceId: string): 'creator' | 'pro' | null {
  const creatorPriceIds = [
    'price_1SK6IQ1LSC8aAl4yNLG3TXDL', // Creator Monthly
    'price_1SK6PO1LSC8aAl4yziu3VCVt'  // Creator Annual
  ]
  
  const proPriceIds = [
    'price_1SK6Hi1LSC8aAl4ySqs5CDZK', // Pro Monthly
    'price_1SK6Q51LSC8aAl4yziu3VCVt'  // Pro Annual
  ]

  if (creatorPriceIds.includes(priceId)) return 'creator'
  if (proPriceIds.includes(priceId)) return 'pro'
  return null
}
