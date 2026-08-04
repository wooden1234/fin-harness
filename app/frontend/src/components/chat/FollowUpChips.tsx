export function FollowUpChips({
  items,
  onSelect,
  disabled,
}: {
  items: string[]
  onSelect: (text: string) => void
  disabled?: boolean
}) {
  if (!items.length) return null
  return (
    <div className="mt-3 space-y-2">
      <p className="text-[11px] text-slate-400 dark:text-slate-500">还可以继续问</p>
      <div className="flex flex-wrap gap-2">
        {items.map((item) => (
          <button
            key={item}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(item)}
            className="text-left text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:border-brand-navy/40 hover:text-brand-navy disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {item}
          </button>
        ))}
      </div>
    </div>
  )
}
