export type AgentStepStatus = 'pending' | 'running' | 'done' | 'error'
export type AgentTodoStatus = 'pending' | 'in_progress' | 'completed'

export interface AgentStepDetail {
  title?: string
  query?: string
  display_text?: string
  error?: string
  columns?: string[]
  rows?: string[][]
}

export interface AgentStep {
  id: string
  label: string
  status: AgentStepStatus
  category?: string
  shortLabel?: string
  detail?: AgentStepDetail
}

export interface AgentTodo {
  id: string
  content: string
  status: AgentTodoStatus
}
