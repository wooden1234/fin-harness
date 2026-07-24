export interface User {
  id: number
  username: string
  email: string
  status: string
  created_at: string
  last_login?: string | null
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface Conversation {
  id: number
  conversation_id: string
  title: string
  status: string
  dialogue_type?: string
  created_at: string
  updated_at?: string
}

export interface ChatMessageRecord {
  id: number
  sender: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
}

export interface Citation {
  source: string
  snippet: string
  page?: number
}

export type AgentRoute = 'faq' | 'account' | 'general'
