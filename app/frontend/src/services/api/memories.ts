import { apiFetch } from './client'

export interface MemoryItem {
  id: string
  memory_key: string
  value: unknown
  display_text: string
  confidence: number
  status: string
}

export interface MemoryProfile {
  user_id: number
  tenant_id: string
  preferences: MemoryItem[]
}

export interface MemorySyncResponse {
  items: MemoryItem[]
  deleted_ids: string[]
  next_cursor: string | null
}

export async function fetchMemoryProfile(): Promise<MemoryProfile> {
  return apiFetch<MemoryProfile>('/api/memories/profile')
}

export async function syncMemories(since?: string): Promise<MemorySyncResponse> {
  const query = since ? '?since=' + encodeURIComponent(since) : ''
  return apiFetch<MemorySyncResponse>('/api/memories/sync' + query)
}
