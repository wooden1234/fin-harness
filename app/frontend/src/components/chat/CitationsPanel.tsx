import type { Citation } from '@/types/api'
import { BookOpen } from 'lucide-react'

export function CitationsPanel({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null

  return (
    <div className="mt-3 rounded-xl border border-brand-gold/30 bg-amber-50/80 dark:bg-amber-950/20 p-3">
      <div className="flex items-center gap-2 text-xs font-semibold text-brand-gold mb-2">
        <BookOpen size={14} />
        引用来源 ({citations.length})
      </div>
      <ul className="space-y-2">
        {citations.map((item, index) => (
          <li
            key={`${item.source}-${index}`}
            className="text-xs text-slate-600 dark:text-slate-300 border-l-2 border-brand-gold/50 pl-2"
          >
            <span className="font-medium text-slate-800 dark:text-slate-100">{item.source}</span>
            {item.page != null && <span className="text-slate-400 ml-1">p.{item.page}</span>}
            <p className="mt-1 text-slate-500 dark:text-slate-400 leading-relaxed">{item.snippet}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}
