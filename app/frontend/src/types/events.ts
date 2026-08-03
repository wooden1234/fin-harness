import type { Citation, AgentRoute } from './api'
import type { AgentStepStatus } from './agentSteps'
import type { AgentTodo } from './agentSteps'

export type AgentSSEEvent =
  | { type: 'token'; content: string }
  | { type: 'done'; content?: string; citations?: Citation[]; route?: AgentRoute }
  | { type: 'interrupt'; conversation_id: string; message?: string }
  | { type: 'meta'; route?: AgentRoute }
  | { type: 'step'; id: string; label: string; status: AgentStepStatus; category?: string; short_label?: string }
  | { type: 'todo_snapshot'; todos: AgentTodo[] }
  | { type: 'error'; message: string }
