'use client';
import { useState } from "react";
import { UploadButton } from "../components/upload-button";
import { GenerateButton } from "../components/generate-button";
import ErrorBoundary from "../components/error-boundary";

export default function Home() {
  const [filename, setFilename] = useState<string | null>(null);
  const [generated, setGenerated] = useState<string | null>(null);
  const [detectedChords, setDetectedChords] = useState<string[]>([]);

  const handleUploadSuccess = (uploaded: string, chords?: string[]) => {
    setFilename(uploaded);
    if (Array.isArray(chords)) {
      setDetectedChords(chords);
    } else {
      setDetectedChords([]);
    }
  };

  return (
    <main className="font-grotesk prose space-y-4">
      <ErrorBoundary>
        <UploadButton onUploadSuccess={handleUploadSuccess} />
        {filename && (
          <GenerateButton filename={filename} detectedChords={detectedChords} onGenerated={setGenerated} />
        )}
        {generated && (
          <>
            <a href={`http://localhost:8000/download/${generated}`} target="_blank">
              Download
            </a>
            <audio controls className="mt-2">
              <source src={`http://localhost:8000/download/${generated}`} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </>
        )}
      </ErrorBoundary>
    </main>
  );
}
