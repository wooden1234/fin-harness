export type AgentStepStatus = 'pending' | 'running' | 'done' | 'error'
export type AgentTodoStatus = 'pending' | 'in_progress' | 'completed'

export interface AgentStep {
  id: string
  label: string
  status: AgentStepStatus
  category?: string
  shortLabel?: string
}

export interface AgentTodo {
  id: string
  content: string
  status: AgentTodoStatus
}
