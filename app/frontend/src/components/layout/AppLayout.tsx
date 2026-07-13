import { useState } from 'react'
import { Sidebar, ThemeToggle } from './Sidebar'
import { ChatView } from '@/components/chat/ChatView'
import { Toast } from '@/components/common/Toast'

export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [toast, setToast] = useState<{
    message: string
    type: 'success' | 'error' | 'info'
  } | null>(null)

  return (
    <div className="h-screen flex bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      {toast && (
        <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}

      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((value) => !value)}
        onToast={(message, type = 'info') => setToast({ message, type })}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur flex items-center justify-between px-4">
          <div>
            <h1 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
              金融智能客服
            </h1>
            <p className="text-[11px] text-slate-400">SSE 流式 · JWT · PostgresSaver · HITL</p>
          </div>
          <ThemeToggle />
        </header>

        <ChatView />
      </div>
    </div>
  )
}
