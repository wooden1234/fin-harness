import type { Citation } from '@/types/api'

export function citationHostname(citation: Citation): string {
  if (citation.url) {
    try {
      return new URL(citation.url).hostname.replace(/^www\./, '')
    } catch {
      // fall through
    }
  }
  return citation.source || citation.title || '来源'
}

export function citationFaviconUrl(citation: Citation): string | null {
  if (!citation.url) return null
  try {
    const host = new URL(citation.url).hostname
    if (!host) return null
    return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=32`
  } catch {
    return null
  }
}

export function citationTitle(citation: Citation): string {
  return citation.title || citation.source || '未命名来源'
}

export function formatCitationDate(publishedAt?: string): string | null {
  if (!publishedAt) return null
  const trimmed = publishedAt.trim()
  if (!trimmed) return null
  const date = new Date(trimmed)
  if (!Number.isNaN(date.getTime())) {
    const y = date.getFullYear()
    const m = String(date.getMonth() + 1).padStart(2, '0')
    const d = String(date.getDate()).padStart(2, '0')
    return `${y}/${m}/${d}`
  }
  // Keep short ISO-like prefixes (YYYY-MM-DD / YYYY/MM/DD)
  const match = trimmed.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})/)
  if (match) {
    return `${match[1]}/${match[2].padStart(2, '0')}/${match[3].padStart(2, '0')}`
  }
  return trimmed.slice(0, 10)
}
