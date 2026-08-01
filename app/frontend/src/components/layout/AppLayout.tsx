import { useState } from 'react'
import { Sidebar, ThemeToggle } from './Sidebar'
import { ChatView } from '@/components/chat/ChatView'
import { SourcesSidebar } from '@/components/chat/SourcesSidebar'
import { Toast } from '@/components/common/Toast'
import { useChatStore } from '@/stores/useChatStore'

export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [toast, setToast] = useState<{
    message: string
    type: 'success' | 'error' | 'info'
  } | null>(null)
  const sourcesOpen = useChatStore((state) => state.sourcesOpen)
  const activeMessageId = useChatStore((state) => state.activeMessageId)
  const messages = useChatStore((state) => state.messages)
  const conversations = useChatStore((state) => state.conversations)
  const activeConversationId = useChatStore((state) => state.activeConversationId)
  const activeCitations =
    messages.find((msg) => msg.id === activeMessageId)?.citations ?? []
  const showSourcesPanel = sourcesOpen && activeCitations.length > 0

  const activeConversation = conversations.find((item) => {
    const key = item.conversation_id ?? String(item.id)
    return key === activeConversationId
  })
  const firstUserMessage = messages.find((msg) => msg.role === 'user')?.content?.trim()
  const conversationTitle =
    activeConversation?.title?.trim() ||
    firstUserMessage ||
    (activeConversationId ? '新会话' : '')

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

      <div className="flex-1 flex min-w-0">
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-14 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur flex items-center justify-between gap-3 px-4">
            <h1 className="min-w-0 flex-1 text-[15px] font-semibold text-slate-800 dark:text-slate-100 truncate">
              {conversationTitle}
            </h1>
            <ThemeToggle />
          </header>

          <ChatView />
        </div>

        {showSourcesPanel && <SourcesSidebar />}
      </div>
    </div>
  )
}
