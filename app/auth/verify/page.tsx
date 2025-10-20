'use client'

import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'

export default function VerifyAuthPage() {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [message, setMessage] = useState('')
  const router = useRouter()
  const searchParams = useSearchParams()

  useEffect(() => {
    const verifyToken = async () => {
      const token = searchParams.get('token')
      
      if (!token) {
        setStatus('error')
        setMessage('Invalid or missing token')
        return
      }

      try {
        const response = await fetch('/api/auth/verify-magic-link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token })
        })

        const data = await response.json()

        if (response.ok) {
          setStatus('success')
          setMessage('Successfully signed in!')
          
          // Store user session
          localStorage.setItem('thumbscore_user_session', data.sessionToken)
          localStorage.setItem('thumbscore_user_email', data.email)
          localStorage.setItem('thumbscore_user_plan', data.plan)
          
          // Redirect to upload page after 2 seconds
          setTimeout(() => {
            router.push('/upload')
          }, 2000)
        } else {
          setStatus('error')
          setMessage(data.error || 'Invalid or expired token')
        }
      } catch (error) {
        setStatus('error')
        setMessage('Something went wrong. Please try again.')
      }
    }

    verifyToken()
  }, [searchParams, router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center p-4">
      <div className="max-w-md w-full text-center">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Verifying Access
          </h1>
        </div>

        {/* Status Card */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl p-8 border border-gray-700/50">
          {status === 'loading' && (
            <div className="space-y-4">
              <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto"></div>
              <p className="text-gray-300">Verifying your magic link...</p>
            </div>
          )}

          {status === 'success' && (
            <div className="space-y-4">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white">Welcome Back!</h2>
              <p className="text-gray-300">{message}</p>
              <p className="text-sm text-gray-400">Redirecting to upload page...</p>
            </div>
          )}

          {status === 'error' && (
            <div className="space-y-4">
              <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white">Verification Failed</h2>
              <p className="text-gray-300">{message}</p>
              <div className="space-y-3">
                <Link 
                  href="/signin" 
                  className="block w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-semibold transition-all duration-200"
                >
                  Try Again
                </Link>
                <Link 
                  href="/" 
                  className="block w-full py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
                >
                  Back to Home
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
