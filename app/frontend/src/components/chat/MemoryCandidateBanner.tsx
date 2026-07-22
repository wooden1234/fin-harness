import type { MemoryCandidate } from '@/services/api/memories'

interface Props {
  candidates: MemoryCandidate[]
  onConfirm: (id: string) => void
  onReject: (id: string) => void
}

export function MemoryCandidateBanner({ candidates, onConfirm, onReject }: Props) {
  if (candidates.length === 0) return null

  return (
    <div className="border-t border-slate-200 dark:border-slate-700 bg-amber-50 dark:bg-amber-950/30 px-4 py-3">
      <div className="max-w-4xl mx-auto space-y-2">
        {candidates.map((candidate) => (
          <div key={candidate.id} className="flex items-center justify-between gap-3 text-sm">
            <span className="text-slate-700 dark:text-slate-200">
              是否记住这个偏好：{candidate.display_text}？
            </span>
            <span className="flex gap-2 shrink-0">
              <button
                type="button"
                className="rounded bg-brand-navy px-3 py-1 text-white"
                onClick={() => onConfirm(candidate.id)}
              >
                记住
              </button>
              <button
                type="button"
                className="rounded border border-slate-300 px-3 py-1 text-slate-600 dark:text-slate-200"
                onClick={() => onReject(candidate.id)}
              >
                不记
              </button>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
