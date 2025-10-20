'use client'

export function useFreeCredits() {
  const key = 'thumbscore_free_uses'
  const used = typeof window !== 'undefined' ? Number(localStorage.getItem(key) || 0) : 0

  function increment() {
    const newCount = used + 1
    localStorage.setItem(key, String(newCount))
  }

  const isUsedUp = used >= 1
  const remaining = Math.max(0, 1 - used)

  return { used, remaining, isUsedUp, increment }
}
