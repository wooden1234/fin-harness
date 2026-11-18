import { Check, X } from 'lucide-react'

export function HitlBanner({
  message,
  onResume,
  disabled,
}: {
  message: string
  onResume: (input: string) => void
  disabled?: boolean
}) {
  const prompt = message.replace(/输入同意或拒绝。?$/, '').trim()

  return (
    <div className="w-full max-w-5xl mx-auto px-2 sm:px-3 pb-2 shrink-0">
      <div className="rounded-3xl border border-brand-gold/50 dark:border-brand-gold/40 bg-white/95 dark:bg-slate-900 shadow-[0_8px_30px_rgba(30,58,95,0.06)] px-5 py-4">
        <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">需要批准工具调用</h4>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{prompt}</p>
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => onResume('同意')}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 rounded-full bg-brand-navy px-4 py-2 text-sm font-medium text-white shadow-md transition-colors hover:bg-brand-light disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Check size={15} />
            同意
          </button>
          <button
            type="button"
            onClick={() => onResume('拒绝')}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            <X size={15} />
            拒绝
          </button>
        </div>
      </div>
    </div>
  )
}
