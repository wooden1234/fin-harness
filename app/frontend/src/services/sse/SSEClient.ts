import type { AgentSSEEvent } from '@/types/events'

export interface SSEClientOptions {
  url: string
  headers?: Record<string, string>
  body?: FormData
  onEvent: (event: AgentSSEEvent) => void
  onConversationId?: (conversationId: string) => void
  onError?: (error: Error) => void
  onComplete?: () => void
}

export class SSEClient {
  private options: SSEClientOptions
  private cancelled = false

  constructor(options: SSEClientOptions) {
    this.options = options
  }

  async start(): Promise<void> {
    this.cancelled = false

    try {
      await this.connect()
    } catch (error) {
      this.options.onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }

  cancel(): void {
    this.cancelled = true
  }

  private async connect(): Promise<void> {
    const response = await fetch(this.options.url, {
      method: 'POST',
      headers: this.options.headers,
      body: this.options.body,
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const conversationId = response.headers.get('X-Conversation-ID')
    if (conversationId) {
      this.options.onConversationId?.(conversationId)
    }

    if (!response.body) {
      throw new Error('Response body is null')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (!this.cancelled) {
      const { done, value } = await reader.read()

      if (done) {
        this.options.onComplete?.()
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue

        const data = line.slice(6).trim()
        if (!data) continue

        try {
          const event = JSON.parse(data) as AgentSSEEvent
          this.options.onEvent(event)
        } catch (error) {
          console.error('Failed to parse SSE event:', data, error)
        }
      }
    }
  }
}
