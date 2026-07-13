import { create } from 'zustand'
import type { User } from '@/types/api'
import { clearToken, getToken } from '@/services/api/client'
import { fetchCurrentUser } from '@/services/api/auth'

interface AuthState {
  user: User | null
  loading: boolean
  initialized: boolean
  setUser: (user: User | null) => void
  bootstrap: () => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: false,
  initialized: false,

  setUser: (user) => set({ user }),

  bootstrap: async () => {
    if (!getToken()) {
      set({ user: null, initialized: true })
      return
    }

    set({ loading: true })
    try {
      const user = await fetchCurrentUser()
      set({ user, initialized: true, loading: false })
    } catch {
      clearToken()
      set({ user: null, initialized: true, loading: false })
    }
  },

  logout: () => {
    clearToken()
    set({ user: null })
  },
}))
