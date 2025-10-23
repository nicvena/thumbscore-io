import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const { email } = await req.json()
    
    if (!email) {
      return NextResponse.json({ error: 'Email required' }, { status: 400 })
    }
    
    console.log('Testing email send to:', email)
    console.log('Resend API key present:', !!process.env.RESEND_API_KEY)
    
    const { Resend } = await import('resend')
    const resend = new Resend(process.env.RESEND_API_KEY)
    
    const result = await resend.emails.send({
      from: 'Test <noreply@resend.dev>',
      to: email,
      subject: 'Test Email from Thumbscore',
      html: '<h1>Test Email</h1><p>If you receive this, email is working!</p>'
    })
    
    console.log('Email send result:', result)
    
    return NextResponse.json({ 
      success: true, 
      result: result,
      message: 'Email sent successfully' 
    })
    
  } catch (error) {
    console.error('Email test failed:', error)
    return NextResponse.json({ 
      error: 'Email test failed', 
      details: JSON.stringify(error, null, 2)
    }, { status: 500 })
  }
}