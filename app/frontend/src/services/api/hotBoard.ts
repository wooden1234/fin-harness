import { apiFetch } from './client'

export type HotBoardPanelId = 'hot_discuss' | 'finance_lookup' | 'market_view'

export interface HotBoardPanel {
  id: HotBoardPanelId
  title: string
  items: string[]
}

export interface HotBoardResponse {
  as_of: string
  source: 'web' | 'fallback' | 'stale'
  panels: HotBoardPanel[]
}

export function fetchHotBoard(refresh = false): Promise<HotBoardResponse> {
  const query = refresh ? '?refresh=true' : ''
  return apiFetch<HotBoardResponse>(`/api/hot-board${query}`)
}
