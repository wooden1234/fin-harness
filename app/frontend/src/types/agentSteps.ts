export type AgentStepStatus = 'pending' | 'running' | 'done' | 'error'

export interface AgentStep {
  id: string
  label: string
  status: AgentStepStatus
  category?: string
  shortLabel?: string
}
