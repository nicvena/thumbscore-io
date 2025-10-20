import { NextResponse } from 'next/server'
import { createMagicLinkToken, getStripeCustomerByEmail } from '@/lib/auth'
import { addToEmailList } from '@/lib/email'

export async function POST(req: Request) {
  try {
    const { email } = await req.json()

    if (!email) {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 })
    }

    // Check if user has an active Stripe subscription or is admin
    const stripeData = await getStripeCustomerByEmail(email)
    
    // Allow admin access for nicvenettacci@gmail.com
    if (email === 'nicvenettacci@gmail.com') {
      // Admin user - proceed with magic link
    } else if (!stripeData || stripeData.status !== 'active') {
      return NextResponse.json({ 
        error: 'No active subscription found for this email. Please upgrade to continue.' 
      }, { status: 404 })
    }

    // Create magic link token
    const token = await createMagicLinkToken(email)

    // Send magic link email
    try {
      const magicLink = `${process.env.NEXT_PUBLIC_APP_URL}/auth/verify?token=${token}`
      
      // Use Resend to send the magic link email
      const { Resend } = await import('resend')
      const resend = new Resend(process.env.RESEND_API_KEY)
      
      console.log('Sending email to:', email)
      console.log('Resend API Key available:', !!process.env.RESEND_API_KEY)
      
      const emailResult = await resend.emails.send({
        from: 'Thumbscore <onboarding@resend.dev>',
        to: email,
        subject: 'Your Thumbscore Magic Link - Sign In',
        html: `
          <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #0a0f25, #1a1832); padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
              <h1 style="color: #00C8C8; font-size: 28px; margin: 0;">Thumbscore.io</h1>
              <p style="color: #66B2FF; margin: 8px 0 0 0;">AI-Powered YouTube Thumbnail Optimization</p>
            </div>
            
            <div style="padding: 30px; background: white; border-radius: 8px;">
              <h2 style="color: #1f2937; text-align: center; margin-bottom: 20px;">Welcome Back! 🎯</h2>
              
              <p style="color: #4b5563; text-align: center; margin-bottom: 30px;">
                Click the button below to sign in to your Thumbscore account and access unlimited AI-powered thumbnail analyses.
              </p>
              
              <div style="text-align: center; margin: 30px 0;">
                <a href="${magicLink}" style="display: inline-block; background: linear-gradient(135deg, #8B5CF6, #3B82F6); color: white; padding: 16px 32px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">
                  ✨ Sign In to Thumbscore
                </a>
              </div>
              
              <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #1f2937; text-align: center; margin-bottom: 15px;">What you get with Thumbscore:</h3>
                <ul style="color: #4b5563; margin: 0; padding-left: 20px;">
                  <li style="margin-bottom: 8px;"><strong>🎯 AI-Powered Analysis:</strong> Get detailed insights on your thumbnails</li>
                  <li style="margin-bottom: 8px;"><strong>📊 Performance Scoring:</strong> Know which thumbnail will perform best</li>
                  <li style="margin-bottom: 8px;"><strong>🚀 Optimization Tips:</strong> Actionable recommendations for improvement</li>
                  <li><strong>⚡ Unlimited Access:</strong> Analyze as many thumbnails as you need</li>
                </ul>
              </div>
              
              <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 15px; margin: 20px 0; text-align: center;">
                <p style="color: #92400e; margin: 0; font-size: 14px;">
                  🔒 <strong>Security:</strong> This link expires in 15 minutes and can only be used once.
                </p>
              </div>
              
              <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                <p style="color: #6b7280; font-size: 14px; text-align: center; margin-bottom: 10px;">
                  If the button doesn't work, copy and paste this link:
                </p>
                <p style="color: #3b82f6; font-size: 12px; text-align: center; word-break: break-all; background: #f8fafc; padding: 10px; border-radius: 4px; margin: 0;">
                  ${magicLink}
                </p>
              </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px;">
              <p>If you didn't request this sign-in link, you can safely ignore this email.</p>
              <p>© 2024 Thumbscore.io - AI-Powered YouTube Thumbnail Optimization</p>
            </div>
          </div>
        `
      })

      console.log('Email result:', emailResult)
      console.log(`Magic link sent to ${email}`)
      
      return NextResponse.json({ 
        success: true, 
        message: 'Magic link sent to your email' 
      })
      
    } catch (emailError) {
      console.error('Error sending magic link email:', emailError)
      return NextResponse.json({ 
        error: 'Failed to send magic link. Please try again.' 
      }, { status: 500 })
    }

  } catch (error) {
    console.error('Error in send-magic-link:', error)
    return NextResponse.json({ 
      error: 'Something went wrong. Please try again.' 
    }, { status: 500 })
  }
}
