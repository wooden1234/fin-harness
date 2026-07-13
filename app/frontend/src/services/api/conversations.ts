import { apiFetch } from './client'
import type { ChatMessageRecord, Conversation } from '@/types/api'

export async function createConversation(title?: string): Promise<{ conversation_id: string; id?: number }> {
  return apiFetch('/api/conversations', {
    method: 'POST',
    body: JSON.stringify(title ? { title } : {}),
  })
}

export async function listConversations(): Promise<Conversation[]> {
  return apiFetch<Conversation[]>('/api/conversations')
}

export async function fetchConversationMessages(
  conversationId: string | number,
): Promise<ChatMessageRecord[]> {
  return apiFetch<ChatMessageRecord[]>(`/api/conversations/${conversationId}/messages`)
}

export async function deleteConversation(conversationId: string | number): Promise<void> {
  await apiFetch(`/api/conversations/${conversationId}`, { method: 'DELETE' })
}

export async function renameConversation(
  conversationId: string | number,
  name: string,
): Promise<void> {
  await apiFetch(`/api/conversations/${conversationId}/name`, {
    method: 'PUT',
    body: JSON.stringify({ name }),
  })
}
