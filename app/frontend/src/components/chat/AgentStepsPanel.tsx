import { useEffect, useRef } from 'react'
import { CheckCircle2, Circle, Loader2 } from 'lucide-react'
import type { AgentStep, AgentTodo } from '@/types/agentSteps'

const ANALYZING_STEP_ID = '__analyzing'

function StepLine({
  label,
  status,
  animate,
}: {
  label: string
  status: AgentStep['status']
  animate?: boolean
}) {
  const isRunning = status === 'running'
  const isDone = status === 'done'
  const isPending = status === 'pending'

  return (
    <p
      className={`text-sm leading-relaxed flex items-center gap-2 ${
        animate ? 'animate-step-in' : ''
      } ${
        isRunning
          ? 'text-slate-800 dark:text-slate-100'
          : isDone
            ? 'text-slate-600 dark:text-slate-300'
            : isPending
              ? 'text-slate-400 dark:text-slate-500'
              : 'text-red-600 dark:text-red-400'
      }`}
    >
      {isRunning ? (
        <Loader2 size={13} className="animate-spin text-brand-gold shrink-0" />
      ) : isDone ? (
        <CheckCircle2 size={13} className="text-emerald-500 shrink-0" />
      ) : isPending ? (
        <Circle size={13} className="text-slate-300 dark:text-slate-600 shrink-0" />
      ) : (
        <span className="w-[13px] h-[13px] rounded-full bg-red-400 shrink-0" />
      )}
      <span>{label}</span>
    </p>
  )
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

  if (!isGenerating && steps.length === 0 && todos.length === 0) return null

  const displayLines: Array<{ id: string; label: string; status: AgentStep['status']; animate?: boolean }> = []

  if (isGenerating || steps.length > 0) {
    displayLines.push({
      id: ANALYZING_STEP_ID,
      label: '正在分析问题…',
      status: steps.length > 0 || !isGenerating ? 'done' : 'running',
    })
  }

  steps.forEach((step) => {
    const isNew = !seenStepIds.current.has(step.id)
    displayLines.push({
      id: step.id,
      label: step.label,
      status: step.status,
      animate: isNew,
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

  return (
    <div className="mb-4 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-3 shadow-sm">
      {todos.length > 0 && (
        <section>
          <p className="mb-2 text-[11px] font-medium tracking-wide text-slate-400 dark:text-slate-500">
            执行计划
          </p>
          <div className="space-y-2">
            {todos.map((todo) => (
              <StepLine
                key={todo.id}
                label={todo.content}
                status={
                  todo.status === 'completed'
                    ? 'done'
                    : todo.status === 'in_progress'
                      ? 'running'
                      : 'pending'
                }
              />
            ))}
          </div>
        </section>
      )}

      {normalizedLines.length > 0 && (
        <section className={todos.length > 0 ? 'mt-4 border-t border-slate-100 pt-3 dark:border-slate-800' : ''}>
          {todos.length > 0 && (
            <p className="mb-2 text-[11px] font-medium tracking-wide text-slate-400 dark:text-slate-500">
              执行记录
            </p>
          )}
          <div className="space-y-2">
            {normalizedLines.map((line) => (
              <StepLine
                key={line.id}
                label={line.label}
                status={line.status}
                animate={line.animate}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
