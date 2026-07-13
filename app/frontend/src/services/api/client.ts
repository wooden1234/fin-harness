const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export function getToken(): string | null {
  return localStorage.getItem('fin_agent_token')
}

export function setToken(token: string): void {
  localStorage.setItem('fin_agent_token', token)
}

export function clearToken(): void {
  localStorage.removeItem('fin_agent_token')
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken()
  const headers = new Headers(options.headers)

  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? payload.message ?? detail
      if (Array.isArray(detail)) {
        detail = detail.map((item) => item.msg ?? JSON.stringify(item)).join('; ')
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(response.status, String(detail))
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}
