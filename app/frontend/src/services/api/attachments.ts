import { apiFetch } from '@/services/api/client'

export type AttachmentUploadResult = {
  attachment_id: string
  object_key: string
  content_type: string
  size: number
}

export async function uploadChatImage(file: File): Promise<AttachmentUploadResult> {
  const form = new FormData()
  form.append('file', file)
  return apiFetch<AttachmentUploadResult>('/api/attachments', {
    method: 'POST',
    body: form,
  })
}
