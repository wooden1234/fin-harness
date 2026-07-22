import { apiFetch } from './client'

export interface MemoryCandidate {
  id: string
  memory_key: string
  value: unknown
  display_text: string
  confidence: number
  status: string
}

export async function listMemoryCandidates(): Promise<MemoryCandidate[]> {
  return apiFetch<MemoryCandidate[]>('/api/memories/candidates')
}

export async function confirmMemoryCandidate(id: string): Promise<MemoryCandidate> {
  return apiFetch<MemoryCandidate>(\`/api/memories/\${id}/confirm\`, { method: 'POST' })
}

export async function rejectMemoryCandidate(id: string): Promise<MemoryCandidate> {
  return apiFetch<MemoryCandidate>(\`/api/memories/\${id}/reject\`, { method: 'POST' })
}
