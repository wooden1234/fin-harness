import { Fragment, useEffect, useRef, useState } from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { HitlBanner } from './HitlBanner'
import { AgentStepsPanel } from './AgentStepsPanel'
import { useChatStore } from '@/stores/useChatStore'
import { useAgentChat } from '@/hooks/useAgentChat'
import { MemoryCandidateBanner } from './MemoryCandidateBanner'
import {
  confirmMemoryCandidate,
  listMemoryCandidates,
  rejectMemoryCandidate,
  type MemoryCandidate,
} from '@/services/api/memories'

const quickPrompts = [
  '信用卡年费怎么收？',
  '帮我查一下账户余额',
  '我要转账 5 万元',
]

export function ChatView() {
  const { messages, isGenerating, hitlPending, hitlMessage, agentSteps } = useChatStore()
  const { sendQuery, resumeAgent, cancelStream } = useAgentChat()
  const [input, setInput] = useState('')
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isGenerating, hitlPending, agentSteps])

  useEffect(() => {
    void listMemoryCandidates().then(setCandidates).catch(() => setCandidates([]))
  }, [messages.length])

  const decideCandidate = async (id: string, confirm: boolean) => {
    try {
      if (confirm) await confirmMemoryCandidate(id)
      else await rejectMemoryCandidate(id)
      setCandidates((items) => items.filter((item) => item.id !== id))
    } catch {
      // 失败时保留候选，下一次请求会重新获取。
    }
  }

  const handleSend = () => {
    const text = input.trim()
    if (!text || isGenerating) return
    setInput('')
    void sendQuery(text)
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center px-6 py-12">
            <div className="w-16 h-16 rounded-2xl bg-brand-navy text-brand-gold flex items-center justify-center text-2xl font-bold mb-6">
              FA
            </div>
            <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
              金融 Multi-Agent 智能客服
            </h2>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-8 text-center max-w-md">
              Supervisor 三路由 · FAQ / Account / General · Compliance 后置审查 · L4 HITL
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full max-w-3xl">
              {quickPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => void sendQuery(prompt)}
                  disabled={isGenerating}
                  className="text-left p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 hover:border-brand-gold dark:hover:border-brand-gold transition-colors text-sm"
                >
                  <span className="font-medium text-slate-800 dark:text-slate-200">{prompt}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto px-4 py-8">
            {messages.map((msg, index) => {
              const showStepsBefore =
                isGenerating &&
                index === messages.length - 1 &&
                msg.role === 'assistant'
              return (
                <Fragment key={msg.id}>
                  {showStepsBefore && (
                    <AgentStepsPanel steps={agentSteps} isGenerating={isGenerating} />
                  )}
                  <ChatMessage message={msg} />
                </Fragment>
              )
            })}
            {isGenerating &&
              (messages.length === 0 || messages[messages.length - 1].role !== 'assistant') && (
                <AgentStepsPanel steps={agentSteps} isGenerating={isGenerating} />
              )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {hitlPending && hitlMessage && (
        <HitlBanner
          message={hitlMessage}
          onResume={(text) => void resumeAgent(text)}
          disabled={isGenerating}
        />
      )}

      <MemoryCandidateBanner
        candidates={candidates}
        onConfirm={(id) => void decideCandidate(id, true)}
        onReject={(id) => void decideCandidate(id, false)}
      />

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={handleSend}
        onCancel={cancelStream}
        disabled={hitlPending}
        isGenerating={isGenerating}
        placeholder={
          hitlPending
            ? '请使用上方人工恢复面板输入补充说明'
            : '输入金融相关问题…'
        }
      />
    </div>
  )
}
