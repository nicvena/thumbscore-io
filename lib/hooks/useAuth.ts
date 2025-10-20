'use client'

import { useState, useEffect } from 'react'

export interface User {
  email: string
  plan: 'creator' | 'pro' | null
  subscriptionStatus: string
  isAuthenticated: boolean
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    try {
      const sessionToken = localStorage.getItem('thumbscore_user_session')
      const email = localStorage.getItem('thumbscore_user_email')
      const plan = localStorage.getItem('thumbscore_user_plan')

      if (sessionToken && email) {
        // Verify session is still valid
        const response = await fetch('/api/auth/verify-session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionToken })
        })

        if (response.ok) {
          const data = await response.json()
          setUser({
            email: data.email,
            plan: data.plan,
            subscriptionStatus: data.subscriptionStatus,
            isAuthenticated: true
          })
        } else {
          // Session invalid, clear local storage
          clearAuth()
        }
      }
    } catch (error) {
      console.error('Error checking auth status:', error)
      clearAuth()
    } finally {
      setLoading(false)
    }
  }

  const signOut = () => {
    clearAuth()
    setUser(null)
  }

  const clearAuth = () => {
    localStorage.removeItem('thumbscore_user_session')
    localStorage.removeItem('thumbscore_user_email')
    localStorage.removeItem('thumbscore_user_plan')
  }

  const hasUnlimitedAccess = () => {
    return user?.isAuthenticated && user?.subscriptionStatus === 'active'
  }

  const getPlanDisplayName = () => {
    if (!user?.plan) return 'Free'
    if (user.email === 'nicvenettacci@gmail.com') return 'Admin'
    return user.plan === 'creator' ? 'Creator' : 'Pro'
  }

  return {
    user,
    loading,
    signOut,
    hasUnlimitedAccess,
    getPlanDisplayName,
    checkAuthStatus
  }
}
