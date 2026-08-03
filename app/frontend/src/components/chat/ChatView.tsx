import { Fragment, useEffect, useRef, useState } from 'react'
import { Flame, LineChart, Sparkles } from 'lucide-react'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { HitlBanner } from './HitlBanner'
import { AgentStepsPanel } from './AgentStepsPanel'
import { useChatStore } from '@/stores/useChatStore'
import { useAgentChat } from '@/hooks/useAgentChat'
import {
  fetchHotBoard,
  type HotBoardPanel,
  type HotBoardPanelId,
  type HotBoardResponse,
} from '@/services/api/hotBoard'

// stale 数据返回后，等后台刷新大概跑完再悄悄拉一次最新结果。
const HOT_BOARD_STALE_RETRY_MS = 6000
import caiceLogo from '@/assets/caice-zhida-logo.png'

const PANEL_ICONS: Record<HotBoardPanelId, typeof Flame> = {
  hot_discuss: Flame,
  finance_lookup: LineChart,
  market_view: Sparkles,
}

const PANEL_TITLES: Record<HotBoardPanelId, string> = {
  hot_discuss: '热门讨论',
  finance_lookup: '财务查数',
  market_view: '市场研判',
}

const FALLBACK_PANELS: HotBoardPanel[] = [
  {
    id: 'hot_discuss',
    title: '热门讨论',
    items: [
      '信用卡年费怎么收？',
      '什么是 T+1 交易制度？',
      '如何筛选高股息 A 股？',
    ],
  },
  {
    id: 'finance_lookup',
    title: '财务查数',
    items: [
      '腾讯 2024 年营业收入是多少？',
      '贵州茅台近五年净利润怎么看？',
      '宁德时代毛利率是多少？',
    ],
  },
  {
    id: 'market_view',
    title: '市场研判',
    items: [
      '今天市场情绪怎么样？',
      '算力概念股近期表现如何？',
      '白酒板块估值贵不贵？',
    ],
  },
]

export function ChatView() {
  const { messages, isGenerating, hitlPending, hitlMessage, agentSteps, agentTodos } = useChatStore()
  const { sendQuery, resumeAgent, cancelStream } = useAgentChat()
  const [input, setInput] = useState('')
  const [panels, setPanels] = useState<HotBoardPanel[]>(FALLBACK_PANELS)
  const [hotAsOf, setHotAsOf] = useState('')
  const [hotLoading, setHotLoading] = useState(false)
  const [hotRefreshing, setHotRefreshing] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isGenerating, hitlPending, agentSteps, agentTodos])

  useEffect(() => {
    if (messages.length > 0) return
    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | undefined

    const applyBoard = (board: HotBoardResponse) => {
      if (cancelled || !board.panels?.length) return
      setPanels(board.panels)
      setHotAsOf(board.as_of)
    }

    setHotLoading(true)
    void fetchHotBoard()
      .then((board) => {
        applyBoard(board)
        // 后端用 stale-while-revalidate：拿到的是昨日缓存或本地兜底，
        // 后台已经在异步重新生成，这里延迟悄悄拉一次最新结果，不打断当前展示。
        if (!cancelled && board.source === 'stale') {
          setHotRefreshing(true)
          retryTimer = setTimeout(() => {
            void fetchHotBoard()
              .then(applyBoard)
              .catch(() => undefined)
              .finally(() => {
                if (!cancelled) setHotRefreshing(false)
              })
          }, HOT_BOARD_STALE_RETRY_MS)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPanels(FALLBACK_PANELS)
          setHotAsOf('')
        }
      })
      .finally(() => {
        if (!cancelled) setHotLoading(false)
      })
    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [messages.length])

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
          <div className="relative h-full overflow-hidden">
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(30,58,95,0.12),_transparent_55%),radial-gradient(ellipse_at_bottom_right,_rgba(201,162,39,0.14),_transparent_45%)] dark:bg-[radial-gradient(ellipse_at_top,_rgba(45,90,135,0.28),_transparent_55%),radial-gradient(ellipse_at_bottom_right,_rgba(201,162,39,0.12),_transparent_40%)]"
            />
            <div
              aria-hidden
              className="pointer-events-none absolute -top-16 left-1/2 h-56 w-56 -translate-x-1/2 rounded-full bg-brand-gold/15 blur-3xl animate-soft-pulse"
            />

            <div className="relative h-full flex flex-col items-center justify-center px-6 py-12">
              <div className="animate-fade-up flex flex-col items-center text-center">
                <img
                  src={caiceLogo}
                  alt="财智"
                  className="w-20 h-20 rounded-[1.35rem] object-cover shadow-lg shadow-brand-navy/15 mb-6 ring-1 ring-brand-navy/10"
                />
                <h2 className="font-display text-3xl md:text-4xl font-bold text-brand-navy dark:text-brand-gold tracking-wide">
                  Hi，我是小财
                </h2>
                <p className="mt-3 text-sm md:text-[15px] text-slate-500 dark:text-slate-400 max-w-md leading-relaxed">
                  面向投研与客服场景的智能助手，帮你查规则、看财报、读市场。
                </p>
              </div>

              <div className="animate-fade-up-delay mt-10 w-full max-w-4xl">
                <div className="mb-3 flex items-center justify-between px-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                      今日热榜
                    </span>
                    {hotAsOf ? (
                      <span className="text-[11px] text-slate-400 tabular-nums">
                        {hotAsOf}
                      </span>
                    ) : null}
                    {hotLoading ? (
                      <span className="text-[11px] text-slate-400">加载中…</span>
                    ) : hotRefreshing ? (
                      <span className="text-[11px] text-slate-400">后台更新中…</span>
                    ) : null}
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {panels.map(({ id, items }) => {
                    const Icon = PANEL_ICONS[id] ?? Flame
                    const title = PANEL_TITLES[id] ?? id
                    return (
                      <section
                        key={id}
                        className="rounded-2xl border border-slate-200/80 dark:border-slate-700/80 bg-white/90 dark:bg-slate-900/80 backdrop-blur shadow-sm px-4 py-4"
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-brand-navy/8 dark:bg-brand-gold/10 text-brand-navy dark:text-brand-gold">
                            <Icon size={14} />
                          </span>
                          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                            {title}
                          </h3>
                        </div>
                        <ol className="space-y-0.5">
                          {items.map((topic, index) => (
                            <li key={`${id}-${topic}`}>
                              <button
                                type="button"
                                onClick={() => void sendQuery(topic)}
                                disabled={isGenerating}
                                className="w-full flex items-center gap-2.5 rounded-xl px-1.5 py-2 text-left hover:bg-slate-50 dark:hover:bg-slate-800/80 transition-colors disabled:opacity-60"
                              >
                                <span
                                  className={`w-4 shrink-0 text-xs font-semibold tabular-nums ${
                                    id === 'hot_discuss' && index < 3
                                      ? 'text-brand-gold'
                                      : 'text-slate-400'
                                  }`}
                                >
                                  {index + 1}
                                </span>
                                <span className="text-sm text-slate-700 dark:text-slate-200 leading-snug">
                                  {topic}
                                </span>
                              </button>
                            </li>
                          ))}
                        </ol>
                      </section>
                    )
                  })}
                </div>
              </div>

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
                    <AgentStepsPanel
                      steps={agentSteps}
                      todos={agentTodos}
                      isGenerating={isGenerating}
                    />
                  )}
                  <ChatMessage message={msg} />
                </Fragment>
              )
            })}
            {isGenerating &&
              (messages.length === 0 || messages[messages.length - 1].role !== 'assistant') && (
                <AgentStepsPanel
                  steps={agentSteps}
                  todos={agentTodos}
                  isGenerating={isGenerating}
                />
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
            : '有问题，尽管问…'
        }
      />
    </div>
  )
}
