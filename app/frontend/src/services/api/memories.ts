import { apiFetch } from './client'

export interface MemoryCandidate {
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
  preferences: MemoryCandidate[]
}

export interface MemorySyncResponse {
  items: MemoryCandidate[]
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

export async function listMemoryCandidates(): Promise<MemoryCandidate[]> {
  return apiFetch<MemoryCandidate[]>('/api/memories/candidates')
}

export async function confirmMemoryCandidate(id: string): Promise<MemoryCandidate> {
  return apiFetch<MemoryCandidate>('/api/memories/' + id + '/confirm', { method: 'POST' })
}

export async function rejectMemoryCandidate(id: string): Promise<MemoryCandidate> {
  return apiFetch<MemoryCandidate>('/api/memories/' + id + '/reject', { method: 'POST' })
}
