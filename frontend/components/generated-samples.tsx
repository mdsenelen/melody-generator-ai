'use client';

import { useEffect, useRef, useState } from 'react';

import { requestJson } from '../app/lib/request';

type Sample = {
  id: string;
  name: string;
  url: string;
  downloadPath: string;
};

function createAudioObjectUrl(base64Audio: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}

export function GeneratedSamples() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const sampleUrlsRef = useRef<string[]>([]);

  useEffect(() => {
    return () => {
      sampleUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const generateSample = async () => {
    setIsGenerating(true);
    try {
      const data = await requestJson<{
        audio_b64?: string;
        download_path?: string;
        filename?: string;
      }>('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chord: 'Cmaj7', duration: 4 }),
        expectedContentType: 'application/json',
      });
      if (!data.audio_b64 || !data.download_path || !data.filename) {
        return;
      }

      const filename = data.filename;
      const downloadPath = data.download_path;
      const sampleUrl = createAudioObjectUrl(data.audio_b64);
      sampleUrlsRef.current.push(sampleUrl);

      setSamples((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          name: filename,
          url: sampleUrl,
          downloadPath,
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
        className={`flex w-full items-center justify-center gap-2 rounded-md px-4 py-2 text-white transition-colors ${
          isGenerating ? 'bg-gray-400' : 'bg-indigo-600 hover:bg-indigo-700'
        }`}
      >
        <span aria-hidden="true">♪</span>
        {isGenerating ? 'Generating...' : 'Generate sample'}
      </button>

      {samples.length > 0 ? (
        <div className="space-y-2">
          <h3 className="font-medium">Generated samples</h3>
          <ul className="divide-y divide-gray-200">
            {samples.map((sample) => (
              <li
                key={sample.id}
                className="flex flex-col space-y-2 py-3 sm:flex-row sm:items-center sm:justify-between sm:space-y-0"
              >
                <span>{sample.name}</span>
                <div className="flex items-center gap-3">
                  <audio controls className="w-48">
                    <source src={sample.url} type="audio/wav" />
                    Your browser does not support audio play.
                  </audio>
                  <a
                    href={sample.downloadPath}
                    download
                    className="text-indigo-600 hover:text-indigo-800"
                    title="Download"
                  >
                    <span aria-hidden="true">Download</span>
                  </a>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
