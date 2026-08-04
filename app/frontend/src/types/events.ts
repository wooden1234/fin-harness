import type { AnswerChartSpec, Citation, AgentRoute } from './api'
import type { AgentStepDetail, AgentStepStatus } from './agentSteps'
import type { AgentTodo } from './agentSteps'

export type AgentSSEEvent =
  | { type: 'token'; content: string }
  | {
      type: 'done'
      content?: string
      citations?: Citation[]
      route?: AgentRoute
      follow_ups?: string[]
      charts?: AnswerChartSpec[]
    }
  | { type: 'interrupt'; conversation_id: string; message?: string }
  | { type: 'meta'; route?: AgentRoute }
  | {
      type: 'step'
      id: string
      label: string
      status: AgentStepStatus
      category?: string
      short_label?: string
      detail?: AgentStepDetail
    }
  | { type: 'todo_snapshot'; todos: AgentTodo[] }
  | { type: 'error'; message: string }
