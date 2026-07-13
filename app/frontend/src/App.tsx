import { useEffect } from 'react'
import { Loader2 } from 'lucide-react'
import { AuthPage } from '@/components/auth/AuthPage'
import { AppLayout } from '@/components/layout/AppLayout'
import { useAuthStore } from '@/stores/useAuthStore'
import { fetchCurrentUser } from '@/services/api/auth'

export default function App() {
  const { user, initialized, loading, bootstrap, setUser } = useAuthStore()

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  if (!initialized || loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-slate-950 text-brand-gold">
        <Loader2 size={32} className="animate-spin" />
      </div>
    )
  }

  if (!user) {
    return <AuthPage onAuthenticated={async () => {
      const currentUser = await fetchCurrentUser()
      setUser(currentUser)
    }} />
  }

  return <AppLayout />
}
