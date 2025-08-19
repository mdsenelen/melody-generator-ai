'use client';
import { useState } from 'react';

type GenerateButtonProps = {
  filename: string;
  onGenerated: (generatedFile: string) => void;
};

export function GenerateButton({ filename, onGenerated }: GenerateButtonProps) {
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      const data = await res.json();
      onGenerated(data.generatedFilename);
    } catch (e) {
      console.error(e);
      alert("Error during generation");
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleGenerate}
      disabled={loading}
      className="btn font-grotesk px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
    >
      {loading ? 'Generating...' : 'Generate'}
    </button>
  );
}