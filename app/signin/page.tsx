'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function SignInPage() {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email) return

    setIsLoading(true)
    setMessage('')

    try {
      const response = await fetch('/api/auth/send-magic-link', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })

      const data = await response.json()

      if (response.ok) {
        setMessage('Check your email for a magic link to sign in!')
      } else {
        setMessage(data.error || 'Something went wrong. Please try again.')
      }
    } catch (error) {
      setMessage('Something went wrong. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-6 inline-block transition-colors duration-200">
            ← Back to Home
          </Link>
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Sign In
          </h1>
          <p className="text-gray-300">
            Enter your email to receive a magic link
          </p>
        </div>

        {/* Sign In Form */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl p-8 border border-gray-700/50">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !email}
              className="w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
                  Sending Magic Link...
                </div>
              ) : (
                'Send Magic Link'
              )}
            </button>
          </form>

          {/* Message */}
          {message && (
            <div className={`mt-6 p-4 rounded-lg text-center ${
              message.includes('Check your email')
                ? 'bg-green-900/50 text-green-300 border border-green-700/50' 
                : 'bg-red-900/50 text-red-300 border border-red-700/50'
            }`}>
              {message}
            </div>
          )}

          {/* Help Text */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-400">
              Don't have an account?{' '}
              <Link href="/pricing" className="text-blue-400 hover:text-blue-300 transition-colors">
                Upgrade to get started
              </Link>
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-8 text-center">
          <h3 className="text-lg font-semibold text-white mb-4">What you get with paid plans:</h3>
          <div className="grid grid-cols-1 gap-3 text-sm text-gray-300">
            <div className="flex items-center justify-center">
              <span className="text-green-400 mr-2">✓</span>
              Unlimited thumbnail analyses
            </div>
            <div className="flex items-center justify-center">
              <span className="text-green-400 mr-2">✓</span>
              Advanced AI insights & explanations
            </div>
            <div className="flex items-center justify-center">
              <span className="text-green-400 mr-2">✓</span>
              Priority processing
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
