// components/error-toast.tsx
'use client'
import { useEffect } from 'react'
import { FiAlertCircle, FiX } from 'react-icons/fi'

export function ErrorToast({ 
  message,
  onDismiss
}: {
  message: string
  onDismiss: () => void
}) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 5000)
    return () => clearTimeout(timer)
  }, [onDismiss])

  return (
    <div className="fixed bottom-4 right-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-md shadow-lg flex items-start gap-2 max-w-xs">
      <FiAlertCircle className="flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-sm">{message}</p>
      </div>
      <button onClick={onDismiss} className="text-red-500 hover:text-red-700">
        <FiX />
      </button>
    </div>
  )
}