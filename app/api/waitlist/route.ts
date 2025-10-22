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
    console.log('=== WAITLIST API DEBUG ===')
    console.log('Environment check:')
    console.log('- RESEND_API_KEY present:', !!process.env.RESEND_API_KEY)
    console.log('- SUPABASE_URL:', process.env.NEXT_PUBLIC_SUPABASE_URL)
    console.log('- SUPABASE_SERVICE_KEY present:', !!process.env.SUPABASE_SERVICE_ROLE_KEY)
    
    const body = await req.json()
    console.log('Request body:', { email: body.email, plan: body.plan, maxPrice: body.maxPrice })
    
    const { email, plan, maxPrice, interests } = body
    
    if (!email || !plan) {
      console.error('Missing required fields:', { email: !!email, plan: !!plan })
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
        return NextResponse.json(
          { error: 'Failed to join waitlist' },
          { status: 500 }
        )
      }
      
      console.log(`New waitlist signup (Supabase): ${email} for ${plan} plan`)
      
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
      console.log(`New waitlist signup (memory): ${email} for ${plan} plan`)
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
  try {
    console.log(`Attempting to send email to ${email}`)
    console.log(`Resend API key present: ${!!process.env.RESEND_API_KEY}`)
    
    // Send confirmation email to user
    const { Resend } = await import('resend')
    const resend = new Resend(process.env.RESEND_API_KEY)
    
    if (!process.env.RESEND_API_KEY) {
      console.error('No Resend API key - skipping email notification')
      return
    }
    
    const emailResult = await resend.emails.send({
      from: 'Thumbscore <noreply@resend.dev>',
      to: email,
      subject: "You're on the Thumbscore Waitlist! 🎯",
      html: `
        <div style='font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;'>
          <h1 style='color: #9333ea; font-size: 28px; margin-bottom: 20px;'>You are on the Waitlist!</h1>
          <p style='font-size: 16px; line-height: 1.5; margin-bottom: 16px;'>
            Thanks for joining the <strong>${plan.charAt(0).toUpperCase() + plan.slice(1)} Plan</strong> waitlist.
          </p>
          <p style='font-size: 16px; line-height: 1.5; margin-bottom: 16px;'>
            We are finalizing features based on feedback from creators like you.
          </p>
          <div style='background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;'>
            <p style='margin: 0; font-size: 14px; color: #374151;'>
              <strong>Expected launch:</strong> 2-3 weeks<br>
              <strong>Your plan preference:</strong> ${plan.charAt(0).toUpperCase() + plan.slice(1)}<br>
              <strong>Price point:</strong> ${maxPrice}/month
            </p>
          </div>
          <p style='font-size: 16px; line-height: 1.5; margin-bottom: 32px;'>
            We will email you as soon as spots open.
          </p>
          <p style='margin-top: 32px;'>
            <a href='https://thumbscore.io' style='display: inline-block; padding: 12px 24px; background: #9333ea; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;'>
              Back to Thumbscore
            </a>
          </p>
          <p style='color: #6b7280; font-size: 14px; margin-top: 32px;'>
            Questions? Just reply to this email.
          </p>
        </div>
      `
    })
    
    console.log(`Email send result:`, emailResult)
    
    if (emailResult.error) {
      console.error(`Email send failed:`, emailResult.error)
    } else {
      console.log(`Confirmation email sent successfully to ${email}, ID: ${emailResult.data?.id}`)
    }
    
  } catch (error) {
    console.error('Email notification failed with error:', error)
    console.error('Error details:', JSON.stringify(error, null, 2))
  }
}