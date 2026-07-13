import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { loginUser, registerUser } from '@/services/api/auth'
import { ApiError } from '@/services/api/client'

type AuthMode = 'login' | 'register'

export function AuthPage({ onAuthenticated }: { onAuthenticated: () => Promise<void> }) {
  const [mode, setMode] = useState<AuthMode>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setLoading(true)

    try {
      if (mode === 'register') {
        await registerUser({ username, email, password })
      }
      await loginUser({ email, password })
      await onAuthenticated()
    } catch (err) {
      const message = err instanceof ApiError ? err.message : '登录失败，请稍后重试'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-brand-navy via-slate-900 to-slate-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-2xl bg-white/95 dark:bg-slate-900/95 shadow-2xl border border-white/10 p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-brand-navy text-brand-gold font-bold text-xl mb-4">
            FA
          </div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">FinAgent 金融智能客服</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
            Multi-Agent · SSE 流式 · JWT 鉴权
          </p>
        </div>

        <div className="flex mb-6 rounded-xl bg-slate-100 dark:bg-slate-800 p-1">
          {(['login', 'register'] as AuthMode[]).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                mode === item
                  ? 'bg-white dark:bg-slate-700 text-brand-navy dark:text-brand-gold shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              {item === 'login' ? '登录' : '注册'}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'register' && (
            <label className="block">
              <span className="text-xs font-semibold text-slate-600 dark:text-slate-300">用户名</span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="mt-1 w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5 text-sm outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold"
                placeholder="your_name"
              />
            </label>
          )}

          <label className="block">
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-300">邮箱</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5 text-sm outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold"
              placeholder="you@example.com"
            />
          </label>

          <label className="block">
            <span className="text-xs font-semibold text-slate-600 dark:text-slate-300">密码</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="mt-1 w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5 text-sm outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold"
              placeholder="••••••••"
            />
          </label>

          {error && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-3 py-2 text-sm text-red-700 dark:text-red-300">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-brand-navy hover:bg-brand-light text-white py-2.5 text-sm font-semibold transition-colors disabled:opacity-60"
          >
            {loading && <Loader2 size={16} className="animate-spin" />}
            {mode === 'login' ? '登录' : '注册并登录'}
          </button>
        </form>
      </div>
    </div>
  )
}
