import { CornerDownLeft, Loader2, Send, Square, X } from 'lucide-react'
import { useEffect, useState } from 'react'

export type PendingImage = {
  file: File
  previewUrl: string
  uploadStatus: 'uploading' | 'ready' | 'error'
  attachmentId?: string
  errorMessage?: string
}

const ACCEPTED_IMAGE_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
])

export const DEFAULT_IMAGE_QUERY = '请根据图片分析'

/** 输入区快捷提示（仅 UI；路由不依赖这些文案） */
const IMAGE_PROMPT_CHIPS = [
  '详细解读要点',
  '提取关键数据',
  '挖掘投资机会',
] as const

export { IMAGE_PROMPT_CHIPS }

function normalizeImageType(type: string): string {
  const raw = (type || '').split(';')[0].trim().toLowerCase()
  if (raw === 'image/jpg') return 'image/jpeg'
  return raw
}

function fileFromClipboardItem(item: DataTransferItem): File | null {
  if (!item.type.startsWith('image/')) return null
  const type = normalizeImageType(item.type)
  if (!ACCEPTED_IMAGE_TYPES.has(type)) return null
  const blob = item.getAsFile()
  if (!blob) return null
  const ext = type === 'image/jpeg' ? 'jpg' : type.split('/')[1] || 'png'
  if (blob instanceof File && blob.name) {
    return blob.type ? blob : new File([blob], blob.name, { type })
  }
  return new File([blob], `paste-${Date.now()}.${ext}`, { type })
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onCancel,
  disabled,
  isGenerating,
  placeholder,
  pendingImage,
  onSelectImage,
  onClearImage,
}: {
  value: string
  onChange: (value: string) => void
  onSend: (text?: string) => void
  onCancel?: () => void
  disabled?: boolean
  isGenerating?: boolean
  placeholder?: string
  pendingImage?: PendingImage | null
  onSelectImage?: (file: File) => void
  onClearImage?: () => void
}) {
  const imageUploading = pendingImage?.uploadStatus === 'uploading'
  const imageReady = pendingImage?.uploadStatus === 'ready'
  const imageError = pendingImage?.uploadStatus === 'error'
  const imageOkForSend = !pendingImage || imageReady
  const canSend =
    Boolean(value.trim() || imageReady) &&
    imageOkForSend &&
    !disabled &&
    !isGenerating
  const hasImage = Boolean(pendingImage)
  const showChips =
    imageReady && !value.trim() && !disabled && !isGenerating
  const canSendChip = imageReady && !disabled && !isGenerating
  const [lightboxOpen, setLightboxOpen] = useState(false)

  useEffect(() => {
    if (!pendingImage) setLightboxOpen(false)
  }, [pendingImage])

  useEffect(() => {
    if (!lightboxOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setLightboxOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightboxOpen])

  const takeImageFile = (file: File | null | undefined) => {
    if (!file || !onSelectImage || disabled || isGenerating) return false
    const type = normalizeImageType(file.type)
    if (!ACCEPTED_IMAGE_TYPES.has(type)) return false
    onSelectImage(
      file.type === type
        ? file
        : new File([file], file.name || `image.${type.split('/')[1]}`, { type }),
    )
    return true
  }

  return (
    <div className="w-full max-w-5xl mx-auto px-2 sm:px-3 pb-3 pt-1 shrink-0">
      <div className="rounded-3xl border border-brand-gold/50 dark:border-brand-gold/40 bg-white/95 dark:bg-slate-900 shadow-[0_8px_30px_rgba(30,58,95,0.06)] focus-within:ring-2 focus-within:ring-brand-gold/35 focus-within:border-brand-gold/70 transition-all">
        {pendingImage ? (
          <div className="px-3 pt-2.5 pb-2 border-b border-slate-100 dark:border-slate-800">
            <div className="relative inline-block">
              <button
                type="button"
                onClick={() => setLightboxOpen(true)}
                className="relative block rounded-lg overflow-hidden hover:opacity-95 transition-opacity"
                title="点击查看大图"
              >
                <img
                  src={pendingImage.previewUrl}
                  alt="待发送图片"
                  className={`h-12 w-12 object-cover border border-slate-200 dark:border-slate-700 ${
                    imageUploading ? 'opacity-60' : ''
                  }`}
                />
                {imageUploading ? (
                  <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-black/35 pointer-events-none">
                    <Loader2
                      size={20}
                      className="animate-spin text-white"
                      aria-label="图片上传中"
                    />
                  </div>
                ) : null}
              </button>
              {onClearImage ? (
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation()
                    onClearImage()
                  }}
                  className="absolute -right-1.5 -top-1.5 rounded-full bg-slate-800 text-white p-0.5 shadow z-10"
                  title="移除图片"
                >
                  <X size={12} />
                </button>
              ) : null}
            </div>
            {imageError ? (
              <p className="mt-1.5 text-xs text-red-500">
                {pendingImage.errorMessage || '图片上传失败，请移除后重试'}
              </p>
            ) : null}
            {imageUploading ? (
              <p className="mt-1.5 text-xs text-slate-400">图片上传中…</p>
            ) : null}
            {showChips ? (
              <div className="mt-2.5 flex flex-wrap gap-2">
                {IMAGE_PROMPT_CHIPS.map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    disabled={!canSendChip}
                    onClick={() => onSend(chip)}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-100 bg-sky-50/90 px-3 py-1 text-xs text-sky-700 hover:bg-sky-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed dark:border-sky-900/50 dark:bg-sky-950/40 dark:text-sky-300"
                  >
                    <CornerDownLeft size={12} className="opacity-70" />
                    {chip}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <div className="relative flex items-end">
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onPaste={(event) => {
              if (!onSelectImage || disabled || isGenerating) return
              const items = event.clipboardData?.items
              if (!items?.length) return
              for (const item of Array.from(items)) {
                const file = fileFromClipboardItem(item)
                if (file && takeImageFile(file)) {
                  event.preventDefault()
                  return
                }
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                if (canSend) onSend()
              }
            }}
            disabled={disabled}
            placeholder={
              hasImage
                ? DEFAULT_IMAGE_QUERY
                : (placeholder ?? '有问题，尽管问…')
            }
            rows={1}
            className={`w-full max-h-32 resize-none border-none bg-transparent outline-none text-[15px] text-slate-800 dark:text-slate-200 placeholder-slate-400 disabled:opacity-60 pl-5 pr-24 ${
              hasImage ? 'min-h-[44px] py-2.5' : 'min-h-[56px] py-4'
            }`}
          />
          <div
            className={`absolute right-3 flex gap-2 ${
              hasImage ? 'bottom-2' : 'bottom-3'
            }`}
          >
            {isGenerating && onCancel && (
              <button
                type="button"
                onClick={onCancel}
                className="p-2.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 hover:text-red-500 transition-colors"
                title="停止生成"
              >
                <Square size={16} />
              </button>
            )}
            <button
              type="button"
              onClick={() => onSend()}
              disabled={!canSend}
              title={
                imageUploading
                  ? '图片上传中，请稍候'
                  : imageError
                    ? '图片上传失败'
                    : undefined
              }
              className={`p-2.5 rounded-full transition-all ${
                canSend
                  ? 'bg-brand-navy hover:bg-brand-light text-white shadow-md'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
              }`}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
      <p className="text-center mt-2 text-[11px] text-slate-400">
        内容由 AI 生成，仅供参考；涉及账户与资金请以官方渠道为准。
      </p>

      {lightboxOpen && pendingImage ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setLightboxOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="查看图片"
        >
          <button
            type="button"
            className="absolute right-4 top-4 rounded-full bg-black/50 p-2 text-white hover:bg-black/70"
            onClick={() => setLightboxOpen(false)}
            title="关闭"
          >
            <X size={20} />
          </button>
          <img
            src={pendingImage.previewUrl}
            alt="待发送图片大图"
            className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      ) : null}
    </div>
  )
}
