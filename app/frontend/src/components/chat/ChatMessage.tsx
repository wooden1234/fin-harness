import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot, Globe } from 'lucide-react'
import type { Components } from 'react-markdown'
import type { Message } from '@/stores/useChatStore'
import { useChatStore } from '@/stores/useChatStore'
import type { Citation } from '@/types/api'
import { citationFaviconUrl, citationHostname } from '@/utils/citations'

const routeLabels = {
  faq: 'FAQ 知识库',
  account: '账户查询',
  general: '通用对话',
} as const

const CITATION_TOKEN_RE = /(\[\d+\])/g
const CITATION_INDEX_RE = /^\[(\d+)\]$/

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

function renderTextWithCitations(
  text: string,
  citations: Citation[] | undefined,
  onSelect: (index: number) => void,
): ReactNode[] {
  if (!citations?.length) return [text]
  return text.split(CITATION_TOKEN_RE).map((part, index) => {
    const match = part.match(CITATION_INDEX_RE)
    if (!match) return <span key={`t-${index}`}>{part}</span>
    const citationIndex = Number(match[1]) - 1
    if (citationIndex < 0 || citationIndex >= citations.length) {
      return <span key={`t-${index}`}>{part}</span>
    }
    return (
      <button
        key={`c-${index}`}
        type="button"
        onClick={(event) => {
          event.preventDefault()
          onSelect(citationIndex)
        }}
        className="mx-0.5 inline-flex items-center justify-center min-w-[1.1rem] h-4 px-1 rounded-full bg-slate-200/90 dark:bg-slate-700 text-[10px] font-semibold text-slate-600 dark:text-slate-200 align-super hover:bg-brand-navy hover:text-white transition-colors"
        aria-label={`查看来源 ${citationIndex + 1}`}
      >
        {citationIndex + 1}
      </button>
    )
  })
}

function mapChildrenWithCitations(
  children: ReactNode,
  citations: Citation[] | undefined,
  onSelect: (index: number) => void,
): ReactNode {
  if (!citations?.length) return children
  return (
    <>
      {Array.isArray(children)
        ? children.map((child, index) => {
            if (typeof child === 'string') {
              return (
                <span key={index}>
                  {renderTextWithCitations(child, citations, onSelect)}
                </span>
              )
            }
            return child
          })
        : typeof children === 'string'
          ? renderTextWithCitations(children, citations, onSelect)
          : children}
    </>
  )
}

export function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const openSources = useChatStore((state) => state.openSources)
  const selectCitation = useChatStore((state) => state.selectCitation)
  const citations = message.citations

  const markdownComponents: Components | undefined = citations?.length
    ? {
        p: ({ children }) => (
          <p>{mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}</p>
        ),
        li: ({ children }) => (
          <li>{mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}</li>
        ),
        td: ({ children }) => (
          <td>{mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}</td>
        ),
        strong: ({ children }) => (
          <strong>
            {mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}
          </strong>
        ),
        em: ({ children }) => (
          <em>
            {mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}
          </em>
        ),
        span: ({ children }) => (
          <span>
            {mapChildrenWithCitations(children, citations, (index) => selectCitation(message.id, index))}
          </span>
        ),
      }
    : undefined

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-brand-navy text-brand-gold flex items-center justify-center mr-3 shrink-0 mt-1">
          <Bot size={16} />
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {!isUser && (message.route || message.interrupted) && (
          <div className="flex flex-wrap gap-2 mb-2">
            {message.route && (
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                {routeLabels[message.route]}
              </span>
            )}
            {message.interrupted && (
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300">
                人工介入
              </span>
            )}
          </div>
        )}

        {!isUser && message.agentSteps && message.agentSteps.length > 0 && (
          <details className="mb-2 text-xs text-slate-500 dark:text-slate-400">
            <summary className="cursor-pointer select-none hover:text-slate-700 dark:hover:text-slate-200">
              已完成分析，查询 {message.agentSteps.length} 个数据源
            </summary>
            <ul className="mt-2 space-y-1 pl-3 border-l border-slate-200 dark:border-slate-700">
              {message.agentSteps.map((step) => (
                <li key={step.id}>{step.label}</li>
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

        <div
          className={`text-[15px] leading-relaxed ${
            isUser
              ? 'bg-brand-navy text-white px-5 py-3 rounded-3xl rounded-tr-md'
              : 'text-slate-800 dark:text-slate-200 prose prose-sm dark:prose-invert max-w-none'
          }`}
        >
          {isUser ? (
            message.content
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  )
}
