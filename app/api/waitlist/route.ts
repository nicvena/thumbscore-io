import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

let supabase: any = null

function getSupabaseClient() {
  if (!supabase && supabaseUrl && supabaseKey && supabaseUrl !== 'https://your-project.supabase.co') {
    supabase = createClient(supabaseUrl, supabaseKey)
  }
  return supabase
}

// Fallback in-memory store if Supabase not configured
const waitlistStore: Array<{
  email: string
  plan: string
  maxPrice: string
  interests: string[]
  joinedAt: string
}> = []

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const { email, plan, maxPrice, interests } = body
    
    if (!email || !plan) {
      return NextResponse.json(
        { error: 'Email and plan are required' },
        { status: 400 }
      )
    }
    
    const supabaseClient = getSupabaseClient()
    
    if (supabaseClient) {
      // Use Supabase for storage
      const { data: existingEntry, error: checkError } = await supabaseClient
        .from('waitlist')
        .select('id')
        .eq('email', email)
        .eq('plan', plan)
        .single()
      
      if (existingEntry) {
        return NextResponse.json(
          { error: 'Already on waitlist for this plan' },
          { status: 409 }
        )
      }
      
      // Insert new waitlist entry
      const { data, error } = await supabaseClient
        .from('waitlist')
        .insert({
          email,
          plan,
          max_price: maxPrice || '$19',
          interests: interests || []
        })
        .select()
        .single()
      
      if (error) {
        console.error('Supabase insert error:', error)
        console.error('Error details:', JSON.stringify(error, null, 2))
        
        // If Supabase fails, continue with email anyway and use fallback storage
        console.log('Falling back to memory storage due to Supabase error')
        const entry = {
          email,
          plan,
          maxPrice: maxPrice || '$19',
          interests: interests || [],
          joinedAt: new Date().toISOString()
        }
        waitlistStore.push(entry)
        console.log(`Fallback: Added to memory storage: ${email}`)
      } else {
        console.log(`Successfully added to Supabase: ${email}`)
      }
      
      console.log(`Waitlist signup: ${email} - ${plan} plan`)
      
    } else {
      // Fallback to in-memory storage
      const existingEntry = waitlistStore.find(entry => entry.email === email && entry.plan === plan)
      if (existingEntry) {
        return NextResponse.json(
          { error: 'Already on waitlist for this plan' },
          { status: 409 }
        )
      }
      
      const entry = {
        email,
        plan,
        maxPrice: maxPrice || '$19',
        interests: interests || [],
        joinedAt: new Date().toISOString()
      }
      
      waitlistStore.push(entry)
      console.log(`Waitlist signup (memory): ${email} - ${plan} plan`)
    }
    
    // Send notifications
    await notifyWaitlistJoin(email, plan, maxPrice, interests)
    
    return NextResponse.json({ success: true })
    
  } catch (error) {
    console.error('Waitlist error:', error)
    return NextResponse.json(
      { error: 'Failed to join waitlist' },
      { status: 500 }
    )
  }
}

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const plan = searchParams.get('plan')
    
    const supabaseClient = getSupabaseClient()
    
    if (supabaseClient) {
      // Use Supabase for count
      let query = supabaseClient
        .from('waitlist')
        .select('id', { count: 'exact', head: true })
      
      if (plan) {
        query = query.eq('plan', plan)
      }
      
      const { count, error } = await query
      
      if (error) {
        console.error('Supabase count error:', error)
        return NextResponse.json({ count: 0 })
      }
      
      return NextResponse.json({ count: count || 0 })
      
    } else {
      // Fallback to in-memory storage
      if (plan) {
        const count = waitlistStore.filter(entry => entry.plan === plan).length
        return NextResponse.json({ count })
      } else {
        const count = waitlistStore.length
        return NextResponse.json({ count })
      }
    }
    
  } catch (error) {
    console.error('GET waitlist error:', error)
    return NextResponse.json({ count: 0 })
  }
}

async function notifyWaitlistJoin(
  email: string,
  plan: string,
  maxPrice: string,
  interests: string[]
) {
  // Email notifications disabled for now - focusing on waitlist collection
  return
}