import { NextResponse } from 'next/server'
import { addToEmailList } from '@/lib/email'

export async function POST(req: Request) {
  try {
    const { email } = await req.json()

    if (!email) {
      return NextResponse.json({ error: 'Email is required' }, { status: 400 })
    }

    const result = await addToEmailList(email)

    if (result.success) {
      return NextResponse.json({ success: true })
    } else {
      return NextResponse.json({ error: 'Failed to add email' }, { status: 500 })
    }
  } catch (error) {
    console.error('Error adding email to list:', error)
    return NextResponse.json({ error: 'Failed to add email' }, { status: 500 })
  }
}
