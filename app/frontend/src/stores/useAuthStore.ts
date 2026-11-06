import { create } from 'zustand'
import type { User } from '@/types/api'
import { clearToken, getToken } from '@/services/api/client'
import { fetchCurrentUser } from '@/services/api/auth'
import { useChatStore } from '@/stores/useChatStore'

function clearChatForAccountSwitch() {
  useChatStore.getState().resetChat()
  useChatStore.setState({ conversations: [] })
}

interface AuthState {
  user: User | null
  loading: boolean
  initialized: boolean
  setUser: (user: User | null) => void
  bootstrap: () => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  loading: false,
  initialized: false,

  setUser: (user) => {
    const current = get().user
    if (current?.id !== user?.id) {
      clearChatForAccountSwitch()
    }
    set({ user })
  },

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
    clearChatForAccountSwitch()
    set({ user: null })
  },
}))
