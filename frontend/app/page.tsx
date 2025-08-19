'use client';
import { useState } from "react";
import { UploadButton } from "../components/upload-button";
import { GenerateButton } from "../components/generate-button";
import ErrorBoundary from  "../components/error-boundary";
// import Video from "next-video"
// import video1 from "../public/video1.mp4";
// import video2 from "../public/video2.mp4";
// import video3 from "../public/video3.mp4";
export default function Home() {
  const [filename, setFilename] = useState<string | null>(null);
  const [generated, setGenerated] = useState<string | null>(null);

  return (
    
    <main className="font-grotesk prose space-y-4">
      <ErrorBoundary>
        <UploadButton onUploadSuccess={setFilename} />
      
      {filename && (
        <GenerateButton filename={filename} onGenerated={setGenerated} />
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
