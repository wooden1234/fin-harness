import { useEffect } from 'react'
import { CheckCircle2, AlertTriangle } from 'lucide-react'

export function Toast({
  message,
  type,
  onClose,
}: {
  message: string
  type: 'success' | 'error' | 'info'
  onClose: () => void
}) {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000)
    return () => clearTimeout(timer)
  }, [onClose])

  const styles = {
    success: 'bg-emerald-600',
    error: 'bg-red-600',
    info: 'bg-brand-navy',
  }[type]

  return (
    <div
      className={`fixed top-4 right-4 ${styles} text-white px-5 py-3 rounded-xl shadow-lg z-50 flex items-center gap-2`}
    >
      {type === 'success' ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
      <span className="text-sm font-medium">{message}</span>
    </div>
  )
}
