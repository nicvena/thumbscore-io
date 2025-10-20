import { Resend } from 'resend'
import { createClient } from '@supabase/supabase-js'

const resend = new Resend(process.env.RESEND_API_KEY)
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

export async function addToEmailList(email: string) {
  try {
    // Save to Supabase database
    const { error } = await supabase
      .from('email_subscribers')
      .insert({ 
        email, 
        subscribed_at: new Date().toISOString(),
        source: 'free_user_signup'
      })

    if (error) {
      console.error('Error saving email to database:', error)
      return { success: false, error }
    }

    console.log('Email saved to database:', email)
    
    // Send welcome email
    await resend.emails.send({
      from: 'Thumbscore <hello@thumbscore.io>',
      to: email,
      subject: 'Welcome to Thumbscore!',
      html: `
        <h1>Welcome to Thumbscore!</h1>
        <p>Thanks for trying our AI thumbnail analysis. Ready to unlock unlimited insights?</p>
        <a href="${process.env.NEXT_PUBLIC_APP_URL}/pricing">Upgrade Now</a>
      `
    })
    
    return { success: true }
  } catch (error) {
    console.error('Error adding email to list:', error)
    return { success: false, error }
  }
}
