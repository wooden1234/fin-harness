import { useEffect, useRef } from 'react'
import { ExternalLink, Globe, X } from 'lucide-react'
import { useChatStore } from '@/stores/useChatStore'
import {
  citationFaviconUrl,
  citationHostname,
  citationTitle,
  formatCitationDate,
} from '@/utils/citations'

function Favicon({ url, alt }: { url: string | null; alt: string }) {
  if (!url) {
    return (
      <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center shrink-0">
        <Globe size={10} className="text-slate-500" />
      </span>
    )
  }
  return (
    <img
      src={url}
      alt={alt}
      className="w-4 h-4 rounded-sm shrink-0 bg-white"
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={(event) => {
        event.currentTarget.style.display = 'none'
      }}
    />
  )
}

export function SourcesSidebar() {
  const {
    sourcesOpen,
    activeMessageId,
    activeCitationIndex,
    messages,
    closeSources,
  } = useChatStore()
  const listRef = useRef<HTMLDivElement>(null)

  const activeMessage = messages.find((msg) => msg.id === activeMessageId)
  const citations = activeMessage?.citations ?? []

  useEffect(() => {
    if (!sourcesOpen || activeCitationIndex == null || !listRef.current) return
    const el = listRef.current.querySelector<HTMLElement>(
      `[data-citation-index="${activeCitationIndex}"]`,
    )
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [sourcesOpen, activeCitationIndex, activeMessageId])

  if (!sourcesOpen || citations.length === 0) return null

  return (
    <>
      <button
        type="button"
        aria-label="关闭搜索结果"
        className="fixed inset-0 z-30 bg-black/20 lg:hidden"
        onClick={closeSources}
      />
      <aside
        className="fixed inset-y-0 right-0 z-40 w-[min(100vw,360px)] lg:static lg:z-0 lg:w-[360px] lg:shrink-0 border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex flex-col"
      >
        <div className="h-14 shrink-0 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-4">
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">搜索结果</h2>
          <button
            type="button"
            onClick={closeSources}
            className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            aria-label="关闭"
          >
            <X size={16} />
          </button>
        </div>

        <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
          {citations.map((citation, index) => {
            const active = index === activeCitationIndex
            const host = citationHostname(citation)
            const title = citationTitle(citation)
            const dateLabel = formatCitationDate(citation.published_at)
            const favicon = citationFaviconUrl(citation)
            const body = (
              <>
                <div className="flex items-start gap-2">
                  <span className="text-[11px] font-semibold text-slate-400 w-4 shrink-0 pt-0.5">
                    {index + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400 mb-1">
                      <Favicon url={favicon} alt={host} />
                      <span className="truncate">{host}</span>
                      {dateLabel && (
                        <>
                          <span className="text-slate-300 dark:text-slate-600">·</span>
                          <span className="shrink-0">{dateLabel}</span>
                        </>
                      )}
                      {citation.url && (
                        <ExternalLink size={11} className="ml-auto shrink-0 opacity-60" />
                      )}
                    </div>
                    <p className="text-sm font-medium text-slate-800 dark:text-slate-100 leading-snug line-clamp-2">
                      {title}
                    </p>
                    {citation.snippet && (
                      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 leading-relaxed line-clamp-3">
                        {citation.snippet}
                      </p>
                    )}
                  </div>
                </div>
              </>
            )

            const className = `block w-full text-left rounded-xl border px-3 py-3 transition-colors ${
              active
                ? 'border-brand-navy/40 bg-slate-100 dark:bg-slate-900 dark:border-brand-gold/40'
                : 'border-transparent hover:bg-slate-50 dark:hover:bg-slate-900/60'
            }`

            if (citation.url) {
              return (
                <a
                  key={`${citation.url}-${index}`}
                  data-citation-index={index}
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={className}
                >
                  {body}
                </a>
              )
            }

            return (
              <div key={`${citation.source}-${index}`} data-citation-index={index} className={className}>
                {body}
              </div>
            )
          })}
        </div>
      </aside>
    </>
  )
}
