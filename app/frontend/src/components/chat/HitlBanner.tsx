import { useState } from 'react'
import { Headphones, Send } from 'lucide-react'

export function HitlBanner({
  message,
  onResume,
  disabled,
}: {
  message: string
  onResume: (input: string) => void
  disabled?: boolean
}) {
  const [input, setInput] = useState('')

  const handleSubmit = () => {
    if (!input.trim() || disabled) return
    onResume(input.trim())
    setInput('')
  }

  return (
    <div className="mx-4 mb-3 rounded-xl border border-orange-300 dark:border-orange-700 bg-orange-50 dark:bg-orange-950/30 p-4">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-full bg-orange-500 text-white flex items-center justify-center shrink-0">
          <Headphones size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-orange-800 dark:text-orange-200">已升级人工处理 (L4 HITL)</h4>
          <p className="text-xs text-orange-700/80 dark:text-orange-300/80 mt-1">{message}</p>
          <div className="mt-3 flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="输入人工处理意见或补充说明…"
              disabled={disabled}
              className="flex-1 rounded-lg border border-orange-200 dark:border-orange-800 bg-white dark:bg-slate-900 px-3 py-2 text-sm outline-none focus:border-orange-400 disabled:opacity-60"
            />
            <button
              type="button"
              onClick={handleSubmit}
              disabled={disabled || !input.trim()}
              className="px-4 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium disabled:opacity-50 flex items-center gap-1"
            >
              <Send size={14} />
              恢复
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
