import { create } from 'zustand'
import type { AgentRoute, Citation, Conversation } from '@/types/api'
import type { AgentStep, AgentStepStatus } from '@/types/agentSteps'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  citations?: Citation[]
  route?: AgentRoute
  interrupted?: boolean
  agentSteps?: AgentStep[]
  timestamp: number
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  messages: Message[]
  isGenerating: boolean
  agentSteps: AgentStep[]
  hitlPending: boolean
  hitlMessage: string | null
  setConversations: (conversations: Conversation[]) => void
  setActiveConversationId: (id: string | null) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  updateMessage: (id: string, patch: Partial<Message>) => void
  setGenerating: (value: boolean) => void
  resetAgentSteps: () => void
  upsertAgentStep: (step: {
    id: string
    label: string
    status: AgentStepStatus
    category?: string
    shortLabel?: string
  }) => void
  setHitlPending: (value: boolean, message?: string | null) => void
  resetChat: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  messages: [],
  isGenerating: false,
  agentSteps: [],
  hitlPending: false,
  hitlMessage: null,

  setConversations: (conversations) => set({ conversations }),
  setActiveConversationId: (id) => set({ activeConversationId: id }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  updateMessage: (id, patch) =>
    set((state) => ({
      messages: state.messages.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg)),
    })),
  setGenerating: (value) => set({ isGenerating: value }),
  resetAgentSteps: () => set({ agentSteps: [] }),
  upsertAgentStep: (step) =>
    set((state) => {
      const existingIndex = state.agentSteps.findIndex((item) => item.id === step.id)
      if (existingIndex >= 0) {
        const agentSteps = [...state.agentSteps]
        agentSteps[existingIndex] = { ...agentSteps[existingIndex], ...step }
        return { agentSteps }
      }
      return { agentSteps: [...state.agentSteps, step] }
    }),
  setHitlPending: (value, message = null) => set({ hitlPending: value, hitlMessage: message }),
  resetChat: () =>
    set({
      messages: [],
      activeConversationId: null,
      isGenerating: false,
      agentSteps: [],
      hitlPending: false,
      hitlMessage: null,
    }),
}))
