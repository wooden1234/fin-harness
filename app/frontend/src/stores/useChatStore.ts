import { create } from 'zustand'
import type { AnswerChartSpec, AgentRoute, Citation, Conversation } from '@/types/api'
import type { AgentStep, AgentStepDetail, AgentStepStatus, AgentTodo } from '@/types/agentSteps'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  imagePreviewUrl?: string
  attachmentId?: string
  citations?: Citation[]
  route?: AgentRoute
  interrupted?: boolean
  agentSteps?: AgentStep[]
  agentTodos?: AgentTodo[]
  followUps?: string[]
  charts?: AnswerChartSpec[]
  timestamp: number
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  messages: Message[]
  isGenerating: boolean
  agentSteps: AgentStep[]
  agentTodos: AgentTodo[]
  hitlPending: boolean
  hitlMessage: string | null
  sourcesOpen: boolean
  activeMessageId: string | null
  activeCitationIndex: number | null
  setConversations: (conversations: Conversation[]) => void
  setActiveConversationId: (id: string | null) => void
  setMessages: (messages: Message[]) => void
  addMessage: (message: Message) => void
  updateMessage: (id: string, patch: Partial<Message>) => void
  setGenerating: (value: boolean) => void
  resetAgentSteps: () => void
  setAgentTodos: (todos: AgentTodo[]) => void
  resetAgentTodos: () => void
  upsertAgentStep: (step: {
    id: string
    label: string
    status: AgentStepStatus
    category?: string
    shortLabel?: string
    detail?: AgentStepDetail
  }) => void
  setHitlPending: (value: boolean, message?: string | null) => void
  openSources: (messageId: string, citationIndex?: number) => void
  closeSources: () => void
  selectCitation: (messageId: string, citationIndex: number) => void
  resetChat: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  messages: [],
  isGenerating: false,
  agentSteps: [],
  agentTodos: [],
  hitlPending: false,
  hitlMessage: null,
  sourcesOpen: false,
  activeMessageId: null,
  activeCitationIndex: null,

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
  setAgentTodos: (agentTodos) => set({ agentTodos }),
  resetAgentTodos: () => set({ agentTodos: [] }),
  upsertAgentStep: (step) =>
    set((state) => {
      const existingIndex = state.agentSteps.findIndex((item) => item.id === step.id)
      if (existingIndex >= 0) {
        const agentSteps = [...state.agentSteps]
        const previous = agentSteps[existingIndex]
        agentSteps[existingIndex] = {
          ...previous,
          ...step,
          // 避免后续无 detail 的进度事件把可展开摘要冲掉
          detail: step.detail ?? previous.detail,
        }
        return { agentSteps }
      }
      return { agentSteps: [...state.agentSteps, step] }
    }),
  setHitlPending: (value, message = null) => set({ hitlPending: value, hitlMessage: message }),
  openSources: (messageId, citationIndex = 0) =>
    set({
      sourcesOpen: true,
      activeMessageId: messageId,
      activeCitationIndex: citationIndex,
    }),
  closeSources: () =>
    set({
      sourcesOpen: false,
      activeCitationIndex: null,
    }),
  selectCitation: (messageId, citationIndex) =>
    set({
      sourcesOpen: true,
      activeMessageId: messageId,
      activeCitationIndex: citationIndex,
    }),
  resetChat: () =>
    set({
      messages: [],
      activeConversationId: null,
      isGenerating: false,
      agentSteps: [],
      agentTodos: [],
      hitlPending: false,
      hitlMessage: null,
      sourcesOpen: false,
      activeMessageId: null,
      activeCitationIndex: null,
    }),
}))
