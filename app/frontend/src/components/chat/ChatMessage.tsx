import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Bot } from 'lucide-react'
import type { Message } from '@/stores/useChatStore'
import { CitationsPanel } from './CitationsPanel'

const routeLabels = {
  faq: 'FAQ 知识库',
  account: '账户查询',
  general: '通用对话',
} as const

export function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user'

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
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          )}
        </div>

        {!isUser && message.citations && message.citations.length > 0 && (
          <CitationsPanel citations={message.citations} />
        )}
      </div>
    </div>
  )
}
