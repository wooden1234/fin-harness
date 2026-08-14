import { Bot, Globe, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '@/stores/useChatStore'
import { useChatStore } from '@/stores/useChatStore'
import type { Citation } from '@/types/api'
import type { AgentStepDetail } from '@/types/agentSteps'
import { citationFaviconUrl, citationHostname } from '@/utils/citations'
import { StepDetailCard } from './StepDetailCard'
import { AnswerChart } from './AnswerChart'
import { FollowUpChips } from './FollowUpChips'

/** 后端质量门附加的资料说明，前端不展示。 */
const MATERIAL_NOTES_SECTION_RE = /\n*###\s*资料说明\s*\n[\s\S]*$/
/** 历史消息里可能残留的 [1][2] 角标，展示时剥离。 */
const INLINE_CITATION_MARKERS_RE = /\[\d+\]/g
const DATA_SOURCE_CATEGORIES = new Set(['weather', 'market', 'web', 'research', 'financial', 'knowledge'])

function displayAssistantContent(content: string): string {
  return content
    .replace(MATERIAL_NOTES_SECTION_RE, '')
    .replace(INLINE_CITATION_MARKERS_RE, '')
    .trimEnd()
}

function FaviconStack({ citations }: { citations: Citation[] }) {
  const preview = citations.slice(0, 5)
  return (
    <span className="inline-flex items-center">
      {preview.map((citation, index) => {
        const favicon = citationFaviconUrl(citation)
        const host = citationHostname(citation)
        return (
          <span
            key={`${host}-${index}`}
            className="w-5 h-5 rounded-full border border-white dark:border-slate-900 bg-slate-100 dark:bg-slate-800 flex items-center justify-center overflow-hidden -ml-1.5 first:ml-0"
            style={{ zIndex: preview.length - index }}
            title={host}
          >
            {favicon ? (
              <img
                src={favicon}
                alt={host}
                className="w-full h-full object-cover"
                loading="lazy"
                referrerPolicy="no-referrer"
              />
            ) : (
              <Globe size={10} className="text-slate-500" />
            )}
          </span>
        )
      })}
    </span>
  )
}

export function ChatMessage({
  message,
  onFollowUp,
  followUpDisabled,
}: {
  message: Message
  onFollowUp?: (
    text: string,
    options?: { attachmentId?: string; imagePreviewUrl?: string },
  ) => void
  followUpDisabled?: boolean
}) {
  const isUser = message.role === 'user'
  const openSources = useChatStore((state) => state.openSources)
  const citations = message.citations
  const todos = message.agentTodos ?? []
  const steps = message.agentSteps ?? []
  const followUps = message.followUps ?? []
  const charts = message.charts ?? []
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const visibleSteps = steps.filter((step) => step.status !== 'error')
  const dataSourceCount = visibleSteps.filter(
    (step) => step.category !== undefined && DATA_SOURCE_CATEGORIES.has(step.category),
  ).length
  // 与过程面板一致：短题不摊开 todos，只保留步骤时间线。
  const showTodosInDetails =
    todos.length > 2 || dataSourceCount > 2
  const timelineItems: Array<{
    id: string
    label: string
    detail?: AgentStepDetail
  }> = [
    ...(showTodosInDetails
      ? todos.map((todo) => ({ id: todo.id, label: todo.content }))
      : []),
    ...visibleSteps.map((step) => ({
      id: step.id,
      label: step.label,
      detail: step.detail,
    })),
  ]
  const hasAnalysisDetails = timelineItems.length > 0
  const analysisSummary =
    dataSourceCount > 0 ? `${dataSourceCount} 条资料` : timelineItems.length > 0 ? '快速推理' : ''

  useEffect(() => {
    if (!lightboxOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setLightboxOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightboxOpen])

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-brand-navy text-brand-gold flex items-center justify-center mr-3 shrink-0 mt-1">
          <Bot size={16} />
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {!isUser && message.interrupted && (
          <div className="flex flex-wrap gap-2 mb-2">
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300">
              人工介入
            </span>
          </div>
        )}

        {!isUser && hasAnalysisDetails && (
          <details className="mb-2 text-xs text-slate-500 dark:text-slate-400">
            <summary className="cursor-pointer select-none hover:text-slate-700 dark:hover:text-slate-200">
              已快速推理{analysisSummary ? ` · ${analysisSummary}` : ''}
            </summary>
            <ul className="mt-2 space-y-2 border-l border-slate-200 pl-3 dark:border-slate-700">
              {timelineItems.map((item) => (
                <li key={item.id}>
                  <div>{item.label}</div>
                  {item.detail && <StepDetailCard detail={item.detail} />}
                </li>
              ))}
            </ul>
          </details>
        )}

        {!isUser && citations && citations.length > 0 && (
          <button
            type="button"
            onClick={() => openSources(message.id, 0)}
            className="mb-2 inline-flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
          >
            <span>已阅读 {citations.length} 个来源</span>
            <FaviconStack citations={citations} />
          </button>
        )}

        {isUser && message.imagePreviewUrl ? (
          <button
            type="button"
            onClick={() => setLightboxOpen(true)}
            className="mb-2 block overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm hover:opacity-95 transition-opacity"
            title="点击查看大图"
          >
            <img
              src={message.imagePreviewUrl}
              alt="用户上传"
              className="h-20 w-20 object-cover"
            />
          </button>
        ) : null}

        <div
          className={`text-[15px] leading-relaxed ${
            isUser
              ? 'bg-brand-navy text-white px-5 py-3 rounded-3xl rounded-tr-md'
              : 'text-slate-800 dark:text-slate-200 prose prose-sm dark:prose-invert max-w-none'
          }`}
        >
          {isUser ? (
            message.content ? (
              <div className="whitespace-pre-wrap">{message.content}</div>
            ) : null
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {displayAssistantContent(message.content)}
            </ReactMarkdown>
          )}
        </div>

        {!isUser &&
          charts.map((chart, index) => (
            <AnswerChart key={`${message.id}-chart-${index}`} chart={chart} />
          ))}

        {!isUser && onFollowUp && followUps.length > 0 && (
          <FollowUpChips
            items={followUps}
            onSelect={(text) => onFollowUp(text)}
            disabled={followUpDisabled}
          />
        )}
      </div>

      {lightboxOpen && message.imagePreviewUrl ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setLightboxOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="查看图片"
        >
          <button
            type="button"
            className="absolute right-4 top-4 rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
            onClick={() => setLightboxOpen(false)}
            title="关闭"
          >
            <X size={20} />
          </button>
          <img
            src={message.imagePreviewUrl}
            alt="用户上传大图"
            className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      ) : null}
    </div>
  )
}
