import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Loader2 } from 'lucide-react'
import type { AgentStep, AgentStepDetail, AgentTodo } from '@/types/agentSteps'
import { useChatStore } from '@/stores/useChatStore'
import { StepDetailCard } from './StepDetailCard'

const DATA_SOURCE_CATEGORIES = new Set([
  'weather',
  'market',
  'web',
  'research',
  'financial',
  'knowledge',
])

const SHORT_QUESTION_TODO_HIDE_THRESHOLD = 2

function hasExpandableDetail(detail?: AgentStepDetail): boolean {
  if (!detail) return false
  return Boolean(
    detail.query ||
      detail.display_text ||
      detail.error ||
      (detail.columns?.length && detail.rows?.length),
  )
}

function StatusIcon({ status }: { status: AgentStep['status'] }) {
  if (status === 'running') {
    return (
      <span className="relative z-[1] flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-brand-light/40 bg-white dark:bg-slate-900">
        <Loader2 size={12} className="animate-spin text-brand-light" />
      </span>
    )
  }
  if (status === 'done') {
    return (
      <span className="relative z-[1] flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand-light text-white">
        <Check size={12} strokeWidth={3} />
      </span>
    )
  }
  return (
    <span className="relative z-[1] flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white dark:border-slate-600 dark:bg-slate-900" />
  )
}

function StepLine({
  label,
  status,
  animate,
  detail,
  defaultOpen,
  isLast,
}: {
  label: string
  status: AgentStep['status']
  animate?: boolean
  detail?: AgentStepDetail
  defaultOpen?: boolean
  isLast?: boolean
}) {
  const expandable = hasExpandableDetail(detail)
  const [open, setOpen] = useState(Boolean(defaultOpen && expandable))

  useEffect(() => {
    if (expandable) {
      setOpen(Boolean(defaultOpen))
    }
  }, [defaultOpen, expandable])

  const titleColor =
    status === 'running'
      ? 'text-slate-800 dark:text-slate-100'
      : status === 'done'
        ? 'text-slate-700 dark:text-slate-200'
        : 'text-slate-400 dark:text-slate-500'

  return (
    <div className={`relative pl-8 ${animate ? 'animate-step-in' : ''}`}>
      {!isLast && (
        <span
          aria-hidden
          className="absolute left-[9px] top-5 bottom-0 w-px bg-slate-200 dark:bg-slate-700"
        />
      )}
      <div className="absolute left-0 top-0.5">
        <StatusIcon status={status} />
      </div>
      {expandable ? (
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className={`flex w-full items-center gap-2 text-left text-sm leading-relaxed ${titleColor}`}
        >
          <span className="flex-1 font-medium">{label}</span>
          <ChevronDown
            size={14}
            className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`}
          />
        </button>
      ) : (
        <p className={`text-sm leading-relaxed ${titleColor}`}>{label}</p>
      )}
      {expandable && open && detail && (
        <div className="mt-1 pb-3">
          <StepDetailCard detail={detail} />
        </div>
      )}
    </div>
  )
}

function shouldShowTodos(steps: AgentStep[], todos: AgentTodo[]): boolean {
  if (todos.length === 0) return false
  if (todos.length > SHORT_QUESTION_TODO_HIDE_THRESHOLD) return true
  const dataSourceCount = steps.filter(
    (step) =>
      step.status !== 'error' &&
      step.category !== undefined &&
      DATA_SOURCE_CATEGORIES.has(step.category),
  ).length
  return dataSourceCount > SHORT_QUESTION_TODO_HIDE_THRESHOLD
}

export function AgentStepsPanel({
  steps,
  todos,
  isGenerating,
  answerStarted = false,
}: {
  steps: AgentStep[]
  todos: AgentTodo[]
  isGenerating: boolean
  answerStarted?: boolean
}) {
  const generationStartedAt = useChatStore((state) => state.generationStartedAt)
  const seenStepIds = useRef<Set<string>>(new Set())
  const [now, setNow] = useState(() => Date.now())
  const [userExpanded, setUserExpanded] = useState<boolean | null>(null)

  useEffect(() => {
    steps.forEach((step) => seenStepIds.current.add(step.id))
  }, [steps])

  useEffect(() => {
    if (!isGenerating) {
      seenStepIds.current.clear()
      setUserExpanded(null)
    }
  }, [isGenerating])

  useEffect(() => {
    if (!isGenerating || !generationStartedAt) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [isGenerating, generationStartedAt])

  const collapsedByPhase = !isGenerating || answerStarted
  const expanded = userExpanded ?? !collapsedByPhase

  const showTodos = shouldShowTodos(steps, todos)
  const visibleSteps = steps.filter((step) => step.status !== 'error' && step.category !== 'answer')

  const displayLines = useMemo(() => {
    const lines: Array<{
      id: string
      label: string
      status: AgentStep['status']
      animate?: boolean
      detail?: AgentStepDetail
    }> = []

    if (isGenerating && visibleSteps.length === 0 && !showTodos) {
      lines.push({
        id: '__analyzing',
        label: '正在理解问题…',
        status: 'running',
      })
    }

    if (showTodos) {
      todos.forEach((todo) => {
        lines.push({
          id: todo.id,
          label: todo.content,
          status:
            todo.status === 'completed'
              ? 'done'
              : todo.status === 'in_progress'
                ? 'running'
                : 'pending',
        })
      })
    }

    visibleSteps.forEach((step) => {
      lines.push({
        id: step.id,
        label:
          step.status === 'done' && step.shortLabel ? step.shortLabel : step.label,
        status: step.status,
        animate: isGenerating && !seenStepIds.current.has(step.id),
        detail: step.detail,
      })
    })
    return lines
  }, [isGenerating, visibleSteps, showTodos, todos])

  const normalizedLines = useMemo(() => {
    let lastRunningIndex = -1
    for (let i = displayLines.length - 1; i >= 0; i -= 1) {
      if (displayLines[i].status === 'running') {
        lastRunningIndex = i
        break
      }
    }
    return displayLines.map((line, index) =>
      line.status === 'running' && lastRunningIndex !== -1 && index !== lastRunningIndex
        ? { ...line, status: 'done' as const }
        : line,
    )
  }, [displayLines])

  const dataSourceCount = visibleSteps.filter(
    (step) => step.category !== undefined && DATA_SOURCE_CATEGORIES.has(step.category),
  ).length

  if (!isGenerating && normalizedLines.length === 0) return null
  if (isGenerating && normalizedLines.length === 0 && collapsedByPhase) return null

  const elapsedSec = generationStartedAt
    ? Math.max(0, Math.floor((now - generationStartedAt) / 1000))
    : 0
  const sourceHint = dataSourceCount > 0 ? `${dataSourceCount} 条资料` : '快速推理'
  const headerTitle = isGenerating && !collapsedByPhase ? '小财正在为你生成答案' : '已快速推理'
  const headerMeta =
    isGenerating && !collapsedByPhase
      ? `快速推理中${elapsedSec > 0 ? ` · 思考 ${elapsedSec}秒` : ''}${
          dataSourceCount > 0 ? ` · ${sourceHint}` : ''
        }`
      : sourceHint

  const lastDetailIndex = (() => {
    for (let i = normalizedLines.length - 1; i >= 0; i -= 1) {
      if (hasExpandableDetail(normalizedLines[i].detail)) return i
    }
    return -1
  })()

  return (
    <div className="mb-3 w-full min-w-0">
      <button
        type="button"
        onClick={() => setUserExpanded(!expanded)}
        className="flex w-full items-start gap-2 rounded-xl px-0 py-1 text-left hover:opacity-90"
      >
        <div className="min-w-0 flex-1">
          {isGenerating && !collapsedByPhase ? (
            <>
              <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
                {headerTitle}
              </p>
              <p className="mt-0.5 text-xs text-slate-400 dark:text-slate-500">{headerMeta}</p>
            </>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {headerTitle}
              {headerMeta ? ` · ${headerMeta}` : ''}
            </p>
          )}
        </div>
        <ChevronDown
          size={16}
          className={`mt-0.5 shrink-0 text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
        />
      </button>
      <div
        className={`grid transition-[grid-template-rows] duration-300 ease-out ${
          expanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
        }`}
      >
        <div className="overflow-hidden">
          <div className="mt-2 space-y-1">
            {normalizedLines.map((line, index) => (
              <StepLine
                key={line.id}
                label={line.label}
                status={line.status}
                animate={line.animate}
                detail={line.detail}
                defaultOpen={index === lastDetailIndex}
                isLast={index === normalizedLines.length - 1}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
