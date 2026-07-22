import { useEffect, useState } from 'react'
import { fetchMemoryProfile, type MemoryProfile } from '@/services/api/memories'

export function MemoryProfilePanel() {
  const [profile, setProfile] = useState<MemoryProfile | null>(null)

  useEffect(() => {
    void fetchMemoryProfile().then(setProfile).catch(() => setProfile(null))
  }, [])

  if (!profile || profile.preferences.length === 0) return null

  return (
    <div className="mx-3 mb-3 rounded-xl border border-slate-200 dark:border-slate-700 p-3">
      <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">
        我的偏好
      </div>
      <div className="space-y-1">
        {profile.preferences.map((item) => (
          <div key={item.id} className="text-xs text-slate-500 dark:text-slate-400 truncate">
            {item.display_text}
          </div>
        ))}
      </div>
    </div>
  )
}
