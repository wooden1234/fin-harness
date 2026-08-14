import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, ChevronDown, Circle, Loader2 } from 'lucide-react'
import type { AgentStep, AgentStepDetail, AgentTodo } from '@/types/agentSteps'
import { StepDetailCard } from './StepDetailCard'

const DATA_SOURCE_CATEGORIES = new Set([
  'weather',
  'market',
  'web',
  'research',
  'financial',
  'knowledge',
])

/** 短题：资料源步骤不多时不展示 todos，避免「执行计划/执行记录」双栏。 */
const SHORT_QUESTION_TODO_HIDE_THRESHOLD = 2

function StatusIcon({ status }: { status: AgentStep['status'] }) {
  if (status === 'running') {
    return <Loader2 size={13} className="animate-spin text-brand-gold shrink-0" />
  }
  if (status === 'done') {
    return <CheckCircle2 size={13} className="text-emerald-500 shrink-0" />
  }
  return <Circle size={13} className="text-slate-300 dark:text-slate-600 shrink-0" />
}

function hasExpandableDetail(detail?: AgentStepDetail): boolean {
  if (!detail) return false
  return Boolean(
    detail.query ||
      detail.display_text ||
      (detail.columns?.length && detail.rows?.length),
  )
}

function StepLine({
  label,
  status,
  animate,
  detail,
  defaultOpen,
}: {
  label: string
  status: AgentStep['status']
  animate?: boolean
  detail?: AgentStepDetail
  defaultOpen?: boolean
}) {
  const expandable = hasExpandableDetail(detail)
  const [open, setOpen] = useState(Boolean(defaultOpen && expandable))
  const isRunning = status === 'running'
  const isDone = status === 'done'

  useEffect(() => {
    if (defaultOpen && expandable) {
      setOpen(true)
    }
  }, [defaultOpen, expandable])

  const colorClass = isRunning
    ? 'text-slate-800 dark:text-slate-100'
    : isDone
      ? 'text-slate-600 dark:text-slate-300'
      : 'text-slate-400 dark:text-slate-500'

  return (
    <div className={animate ? 'animate-step-in' : undefined}>
      {expandable ? (
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className={`w-full text-left text-sm leading-relaxed flex items-center gap-2 ${colorClass}`}
        >
          <StatusIcon status={status} />
          <span className="flex-1">{label}</span>
          <ChevronDown
            size={14}
            className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`}
          />
        </button>
      ) : (
        <p className={`text-sm leading-relaxed flex items-center gap-2 ${colorClass}`}>
          <StatusIcon status={status} />
          <span>{label}</span>
        </p>
      )}
      {expandable && open && detail && (
        <div className="ml-[21px]">
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
}: {
  steps: AgentStep[]
  todos: AgentTodo[]
  isGenerating: boolean
}) {
  const seenStepIds = useRef<Set<string>>(new Set())

  useEffect(() => {
    steps.forEach((step) => seenStepIds.current.add(step.id))
  }, [steps])

  useEffect(() => {
    if (!isGenerating) {
      seenStepIds.current.clear()
    }
  }, [isGenerating])

  const showTodos = shouldShowTodos(steps, todos)
  if (!isGenerating && steps.length === 0 && !showTodos) return null

  const hasProblemAnalysis = steps.some((step) => step.id === 'problem_analysis')
  const displayLines: Array<{
    id: string
    label: string
    status: AgentStep['status']
    animate?: boolean
    detail?: AgentStepDetail
  }> = []

  if (isGenerating && steps.length === 0 && !hasProblemAnalysis) {
    displayLines.push({
      id: '__analyzing',
      label: '正在理解问题…',
      status: 'running',
    })
  }

  if (showTodos) {
    todos.forEach((todo) => {
      displayLines.push({
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

  steps.forEach((step) => {
    // 无结果/失败步骤不展示，避免打断研究过程观感
    if (step.status === 'error') return
    const isNew = !seenStepIds.current.has(step.id)
    displayLines.push({
      id: step.id,
      label: step.label,
      status: step.status,
      animate: isNew,
      detail: step.detail,
    })
  })

  let lastRunningIndex = -1
  for (let i = displayLines.length - 1; i >= 0; i -= 1) {
    if (displayLines[i].status === 'running') {
      lastRunningIndex = i
      break
    }
  }
  const normalizedLines = displayLines.map((line, index) =>
    line.status === 'running' && lastRunningIndex !== -1 && index !== lastRunningIndex
      ? { ...line, status: 'done' as const }
      : line,
  )

  const dataSourceCount = steps.filter(
    (step) =>
      step.status !== 'error' &&
      step.category !== undefined &&
      DATA_SOURCE_CATEGORIES.has(step.category),
  ).length
  const headerHint =
    !isGenerating && (steps.length > 0 || showTodos)
      ? dataSourceCount > 0
        ? `已快速推理 · ${dataSourceCount} 条资料`
        : '已快速推理'
      : null

  const lastDetailIndex = (() => {
    for (let i = normalizedLines.length - 1; i >= 0; i -= 1) {
      if (hasExpandableDetail(normalizedLines[i].detail)) return i
    }
    return -1
  })()

  return (
    <div className="mb-4 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-3 shadow-sm">
      {headerHint && (
        <p className="mb-2 text-[11px] font-medium tracking-wide text-slate-400 dark:text-slate-500">
          {headerHint}
        </p>
      )}
      <div className="space-y-2">
        {normalizedLines.map((line, index) => (
          <StepLine
            key={line.id}
            label={line.label}
            status={line.status}
            animate={line.animate}
            detail={line.detail}
            defaultOpen={!isGenerating && index === lastDetailIndex}
          />
        ))}
      </div>
    </div>
  )
}
