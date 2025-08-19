'use client';

import { useState } from 'react';
import { FiMusic, FiDownload } from 'react-icons/fi';

type Sample = {
  id: string;
  name: string;
  url: string; // dinleme ve indirme için
};

export function GeneratedSamples() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateSample = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: 1 }),
      });

      const data = await response.json();

      // backend'den dönen dosya ismini al
      const generatedFile = data.generated_file; // örn: "sample_123.wav"
      const fileUrl = `http://localhost:8000/download/${generatedFile}`; // veya API'nin verdiği tam url

      setSamples((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          name: `Melodi ${prev.length + 1}`,
          url: fileUrl,
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <button
        onClick={generateSample}
        disabled={isGenerating}
        className={`w-full py-2 px-4 rounded-md flex items-center justify-center gap-2 ${
          isGenerating ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'
        } text-white transition-colors`}
      >
        <FiMusic />
        {isGenerating ? 'Üretiliyor...' : 'Yeni Melodi Üret'}
      </button>

      {samples.length > 0 && (
        <div className="space-y-2">
          <h3 className="font-medium">Generated Samples</h3>
          <ul className="divide-y divide-gray-200">
            {samples.map((sample) => (
              <li
                key={sample.id}
                className="py-3 flex flex-col sm:flex-row sm:justify-between sm:items-center space-y-2 sm:space-y-0"
              >
                <span>{sample.name}</span>
                <div className="flex items-center gap-3">
                  <audio controls className="w-48">
                    <source src={sample.url} type="audio/wav" />
                    Your browser does not support audio play.
                  </audio>
                  <a
                    href={sample.url}
                    download
                    className="text-indigo-600 hover:text-indigo-800"
                    title="İndir"
                  >
                    <FiDownload />
                  </a>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
