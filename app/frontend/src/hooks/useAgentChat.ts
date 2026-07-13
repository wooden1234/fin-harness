import { useRef } from 'react'
import { SSEClient } from '@/services/sse/SSEClient'
import { createConversation } from '@/services/api/conversations'
import { getToken } from '@/services/api/client'
import { useChatStore } from '@/stores/useChatStore'
import type { AgentSSEEvent } from '@/types/events'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
const PERSISTED_CONVERSATION_ID_RE = /^\d+$/

function buildAgentForm(fields: Record<string, string>): FormData {
  const form = new FormData()
  Object.entries(fields).forEach(([key, value]) => {
    if (value) form.append(key, value)
  })
  return form
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export function useAgentChat() {
  const clientRef = useRef<SSEClient | null>(null)
  const {
    activeConversationId,
    setActiveConversationId,
    addMessage,
    updateMessage,
    setGenerating,
    resetAgentSteps,
    upsertAgentStep,
    agentSteps,
    setHitlPending,
  } = useChatStore()

  const ensureConversationId = async (): Promise<string> => {
    const currentConversationId = useChatStore.getState().activeConversationId
    if (currentConversationId && PERSISTED_CONVERSATION_ID_RE.test(currentConversationId)) {
      return currentConversationId
    }

    const created = await createConversation()
    const conversationId = String(created.conversation_id ?? created.id ?? '')
    if (!conversationId) {
      throw new Error('创建会话失败：缺少 conversation_id')
    }

    setActiveConversationId(conversationId)
    return conversationId
  }

  const runStream = async (
    endpoint: '/api/agent/query' | '/api/agent/resume',
    fields: Record<string, string>,
  ) => {
    let contentBuffer = ''
    let assistantMessageId = ''

    const handleEvent = (event: AgentSSEEvent) => {
      if (event.type === 'step') {
        upsertAgentStep({
          id: event.id,
          label: event.label,
          status: event.status,
          category: event.category,
          shortLabel: event.short_label,
        })
      }

      if (event.type === 'token') {
        contentBuffer += event.content

        if (!assistantMessageId) {
          assistantMessageId = `assistant-${Date.now()}`
          addMessage({
            id: assistantMessageId,
            role: 'assistant',
            content: contentBuffer,
            timestamp: Date.now(),
          })
        } else {
          updateMessage(assistantMessageId, { content: contentBuffer })
        }
      }

      if (event.type === 'meta') {
        if (assistantMessageId) {
          updateMessage(assistantMessageId, {
            route: event.route,
            riskLevel: event.risk_level,
          })
        }
      }

      if (event.type === 'done') {
        if (!assistantMessageId && event.content) {
          assistantMessageId = `assistant-${Date.now()}`
          contentBuffer = event.content
          addMessage({
            id: assistantMessageId,
            role: 'assistant',
            content: contentBuffer,
            timestamp: Date.now(),
          })
        }
        if (assistantMessageId) {
          updateMessage(assistantMessageId, {
            content: contentBuffer || event.content || '',
            citations: event.citations,
            route: event.route,
            riskLevel: event.risk_level,
            agentSteps: [...useChatStore.getState().agentSteps],
          })
        }
        resetAgentSteps()
        setGenerating(false)
        setHitlPending(false)
      }

      if (event.type === 'interrupt') {
        setHitlPending(true, event.message ?? '该问题已升级人工处理，请稍候或输入补充说明后恢复。')
        if (assistantMessageId) {
          updateMessage(assistantMessageId, {
            content: contentBuffer || event.message || '已转人工客服，请稍候…',
            interrupted: true,
          })
        } else {
          addMessage({
            id: `assistant-hitl-${Date.now()}`,
            role: 'assistant',
            content: event.message || '已转人工客服，请稍候…',
            interrupted: true,
            timestamp: Date.now(),
          })
        }
        setGenerating(false)
        resetAgentSteps()
      }

      if (event.type === 'error') {
        if (!assistantMessageId) {
          addMessage({
            id: `assistant-error-${Date.now()}`,
            role: 'assistant',
            content: `抱歉，服务暂时不可用：${event.message}`,
            timestamp: Date.now(),
          })
        } else {
          updateMessage(assistantMessageId, {
            content: contentBuffer || `抱歉，服务暂时不可用：${event.message}`,
          })
        }
        setGenerating(false)
        setHitlPending(false)
        resetAgentSteps()
      }
    }

    const client = new SSEClient({
      url: `${API_BASE_URL}${endpoint}`,
      headers: authHeaders(),
      body: buildAgentForm(fields),
      onConversationId: (conversationId) => {
        const currentConversationId = useChatStore.getState().activeConversationId
        if (!currentConversationId) {
          setActiveConversationId(conversationId)
        }
      },
      onEvent: handleEvent,
      onError: (error) => {
        addMessage({
          id: `assistant-error-${Date.now()}`,
          role: 'assistant',
          content: `网络错误：${error.message}`,
          timestamp: Date.now(),
        })
        setGenerating(false)
        setHitlPending(false)
        resetAgentSteps()
      },
      onComplete: () => {
        const stillGenerating = useChatStore.getState().isGenerating
        if (stillGenerating) {
          setGenerating(false)
          resetAgentSteps()
        }
      },
    })

    clientRef.current = client
    await client.start()
    clientRef.current = null
  }

  const sendQuery = async (query: string) => {
    const trimmed = query.trim()
    if (!trimmed) return

    try {
      const conversationId = await ensureConversationId()

      addMessage({
        id: `user-${Date.now()}`,
        role: 'user',
        content: trimmed,
        timestamp: Date.now(),
      })

      setGenerating(true)
      resetAgentSteps()
      setHitlPending(false)

      const fields: Record<string, string> = {
        query: trimmed,
        conversation_id: conversationId,
      }

      await runStream('/api/agent/query', fields)
    } catch (error) {
      addMessage({
        id: `assistant-error-${Date.now()}`,
        role: 'assistant',
        content: error instanceof Error ? error.message : '创建会话失败',
        timestamp: Date.now(),
      })
      setGenerating(false)
      setHitlPending(false)
      resetAgentSteps()
    }
  }

  const resumeAgent = async (humanInput: string) => {
    if (!activeConversationId) return

    const trimmed = humanInput.trim()
    if (!trimmed) return

    addMessage({
      id: `user-${Date.now()}`,
      role: 'user',
      content: `[人工补充] ${trimmed}`,
      timestamp: Date.now(),
    })

    setGenerating(true)
    resetAgentSteps()
    setHitlPending(false)

    await runStream('/api/agent/resume', {
      conversation_id: activeConversationId,
      query: trimmed,
    })
  }

  const cancelStream = () => {
    clientRef.current?.cancel()
    clientRef.current = null
    setGenerating(false)
    resetAgentSteps()
  }

  return { sendQuery, resumeAgent, cancelStream, agentSteps }
}
