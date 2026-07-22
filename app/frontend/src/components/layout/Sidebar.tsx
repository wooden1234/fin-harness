import { useEffect, useRef, useState } from 'react'
import {
  MessageSquarePlus,
  Trash2,
  LogOut,
  Loader2,
  Moon,
  Sun,
  MessageSquare,
} from 'lucide-react'
import {
  createConversation,
  deleteConversation,
  fetchConversationMessages,
  listConversations,
} from '@/services/api/conversations'
import { useAuthStore } from '@/stores/useAuthStore'
import { useChatStore, type Message } from '@/stores/useChatStore'
import type { Conversation } from '@/types/api'
import { MemoryProfilePanel } from './MemoryProfilePanel'

function formatTitle(conversation: Conversation): string {
  return conversation.title || `会话 ${conversation.conversation_id?.slice(0, 8) ?? conversation.id}`
}

export function Sidebar({
  collapsed,
  onToggle,
  onToast,
}: {
  collapsed: boolean
  onToggle: () => void
  onToast: (message: string, type?: 'success' | 'error' | 'info') => void
}) {
  const { user, logout } = useAuthStore()
  const {
    conversations,
    activeConversationId,
    isGenerating,
    setConversations,
    setActiveConversationId,
    setMessages,
    resetChat,
  } = useChatStore()
  const [loadingList, setLoadingList] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const prevGenerating = useRef(isGenerating)

  const refreshConversations = async () => {
    setLoadingList(true)
    try {
      const items = await listConversations()
      setConversations(items)
    } catch (error) {
      onToast(error instanceof Error ? error.message : '加载会话失败', 'error')
    } finally {
      setLoadingList(false)
    }
  }

  useEffect(() => {
    void refreshConversations()
  }, [])

  // 消息生成完成后自动刷新会话列表（新会话标题此时已更新）
  useEffect(() => {
    if (prevGenerating.current && !isGenerating) {
      void refreshConversations()
    }
    prevGenerating.current = isGenerating
  }, [isGenerating])

  const handleNewConversation = async () => {
    try {
      const created = await createConversation()
      const conversationId = String(created.conversation_id ?? created.id)
      resetChat()
      setActiveConversationId(conversationId)
      await refreshConversations()
      onToast('已创建新会话', 'success')
    } catch (error) {
      onToast(error instanceof Error ? error.message : '创建会话失败', 'error')
    }
  }

  const handleSelectConversation = async (conversation: Conversation) => {
    const conversationKey = conversation.conversation_id ?? String(conversation.id)
    setActiveConversationId(conversationKey)
    setLoadingMessages(true)

    try {
      const records = await fetchConversationMessages(conversationKey)
      const mapped: Message[] = records.map((record) => ({
        id: `history-${record.id}`,
        role: record.sender === 'user' ? 'user' : 'assistant',
        content: record.content,
        timestamp: new Date(record.created_at).getTime(),
      }))
      setMessages(mapped)
    } catch (error) {
      onToast(error instanceof Error ? error.message : '加载消息失败', 'error')
      setMessages([])
    } finally {
      setLoadingMessages(false)
    }
  }

  const handleDeleteConversation = async (
    event: React.MouseEvent,
    conversation: Conversation,
  ) => {
    event.stopPropagation()
    const conversationKey = conversation.conversation_id ?? String(conversation.id)

    try {
      await deleteConversation(conversationKey)
      if (activeConversationId === conversationKey) {
        resetChat()
      }
      await refreshConversations()
      onToast('会话已删除', 'success')
    } catch (error) {
      onToast(error instanceof Error ? error.message : '删除失败', 'error')
    }
  }

  if (collapsed) {
    return (
      <aside className="w-14 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex flex-col items-center py-4 gap-3">
        <button
          type="button"
          onClick={onToggle}
          className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
          title="展开侧边栏"
        >
          <MessageSquare size={18} />
        </button>
        <button
          type="button"
          onClick={() => void handleNewConversation()}
          className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-brand-gold"
          title="新会话"
        >
          <MessageSquarePlus size={18} />
        </button>
      </aside>
    )
  }

  return (
    <aside className="w-72 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex flex-col">
      <div className="h-16 px-4 flex items-center justify-between border-b border-slate-200 dark:border-slate-800">
        <div>
          <div className="font-bold text-brand-navy dark:text-brand-gold">FinAgent</div>
          <div className="text-[11px] text-slate-400 truncate max-w-[160px]">{user?.email}</div>
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
        >
          收起
        </button>
      </div>

      <div className="p-3">
        <button
          type="button"
          onClick={() => void handleNewConversation()}
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-brand-navy hover:bg-brand-light text-white py-2.5 text-sm font-medium transition-colors"
        >
          <MessageSquarePlus size={16} />
          新会话
        </button>
      </div>

      <MemoryProfilePanel />

      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {loadingList ? (
          <div className="flex justify-center py-8">
            <Loader2 size={20} className="animate-spin text-brand-gold" />
          </div>
        ) : conversations.length === 0 ? (
          <p className="text-xs text-slate-400 text-center py-8 px-4">暂无历史会话</p>
        ) : (
          <div className="space-y-1">
            {conversations.map((conversation) => {
              const key = conversation.conversation_id ?? String(conversation.id)
              const active = activeConversationId === key

              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => void handleSelectConversation(conversation)}
                  className={`w-full group flex items-center gap-2 rounded-xl px-3 py-2.5 text-left transition-colors ${
                    active
                      ? 'bg-brand-navy/10 dark:bg-brand-gold/10 text-brand-navy dark:text-brand-gold'
                      : 'hover:bg-slate-100 dark:hover:bg-slate-900 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <MessageSquare size={14} className="shrink-0 opacity-60" />
                  <span className="flex-1 truncate text-sm">{formatTitle(conversation)}</span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(event) => void handleDeleteConversation(event, conversation)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        void handleDeleteConversation(event as unknown as React.MouseEvent, conversation)
                      }
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-red-500"
                  >
                    <Trash2 size={14} />
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {loadingMessages && (
          <div className="flex justify-center py-4">
            <Loader2 size={16} className="animate-spin text-slate-400" />
          </div>
        )}
      </div>

      <div className="p-3 border-t border-slate-200 dark:border-slate-800">
        <button
          type="button"
          onClick={logout}
          className="w-full flex items-center justify-center gap-2 rounded-xl border border-slate-200 dark:border-slate-700 py-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-900"
        >
          <LogOut size={16} />
          退出登录
        </button>
      </div>
    </aside>
  )
}

export function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <button
      type="button"
      onClick={() => setDark((value) => !value)}
      className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
      title="切换主题"
    >
      {dark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  )
}
