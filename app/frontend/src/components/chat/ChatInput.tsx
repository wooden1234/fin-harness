import { Send, Square } from 'lucide-react'

export function ChatInput({
  value,
  onChange,
  onSend,
  onCancel,
  disabled,
  isGenerating,
  placeholder,
}: {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onCancel?: () => void
  disabled?: boolean
  isGenerating?: boolean
  placeholder?: string
}) {
  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-6 pt-2">
      <div className="relative flex items-end rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm focus-within:ring-2 focus-within:ring-brand-gold/40 focus-within:border-brand-gold transition-all">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSend()
            }
          }}
          disabled={disabled}
          placeholder={placeholder ?? '输入金融相关问题，例如：信用卡年费怎么收？'}
          rows={1}
          className="w-full max-h-40 min-h-[56px] py-4 pl-4 pr-24 bg-transparent border-none outline-none text-[15px] resize-none text-slate-800 dark:text-slate-200 placeholder-slate-400 disabled:opacity-60"
        />
        <div className="absolute right-3 bottom-3 flex gap-2">
          {isGenerating && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="p-2 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-500 hover:text-red-500 transition-colors"
              title="停止生成"
            >
              <Square size={16} />
            </button>
          )}
          <button
            type="button"
            onClick={onSend}
            disabled={disabled || !value.trim() || isGenerating}
            className={`p-2 rounded-xl transition-all ${
              value.trim() && !disabled && !isGenerating
                ? 'bg-brand-navy hover:bg-brand-light text-white shadow-md'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
            }`}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
      <p className="text-center mt-2 text-[11px] text-slate-400">
        回复由 AI 生成，涉及账户与资金操作请以官方渠道为准。L4 风险问题将升级人工。
      </p>
    </div>
  )
}
