"use client";
import { useState } from 'react';

type UploadButtonProps = {
  onUploadSuccess: (filename: string) => void;
};

export function UploadButton({ onUploadSuccess }: UploadButtonProps) {
  const [status, setStatus] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setStatus('Uploading...');

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      onUploadSuccess(data.filename);
      setStatus('Upload successful!');
    } catch (e) {
      console.error("Upload error:", e);
      setStatus('Upload failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        disabled={isLoading}
        className="relative px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 "
      >
        {isLoading ? (
          'Uploading...'
        ) : (
          <>
            Upload & Generate
            <input 
              type="file" 
              accept="audio/*" 
              onChange={handleFile} 
              className="absolute inset-0 opacity-0 cursor-pointer"
              disabled={isLoading}
            />
          </>
        )}
      </button>
      {status && <p className="text-sm text-gray-300">{status}</p>}
    </div>
  );
}