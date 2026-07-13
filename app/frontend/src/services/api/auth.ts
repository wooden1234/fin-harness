import { apiFetch, setToken } from './client'
import type { TokenResponse, User } from '@/types/api'

export async function registerUser(payload: {
  username: string
  email: string
  password: string
}): Promise<User> {
  return apiFetch<User>('/api/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function loginUser(payload: {
  email: string
  password: string
}): Promise<TokenResponse> {
  const token = await apiFetch<TokenResponse>('/api/token', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  setToken(token.access_token)
  return token
}

export async function fetchCurrentUser(): Promise<User> {
  return apiFetch<User>('/api/users/me')
}
