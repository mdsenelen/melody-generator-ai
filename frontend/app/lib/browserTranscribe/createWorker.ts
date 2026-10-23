// Isolated so jest (which can't parse import.meta) never loads it -- tests
// inject a fake WorkerLike instead. The `new URL(..., import.meta.url)`
// form is what tells the bundler to emit the worker -- and TF.js with it --
// as its own chunk, keeping both out of the page bundle.
export function createTranscribeWorker(): Worker {
  return new Worker(new URL("./transcribe.worker.ts", import.meta.url), { type: "module" });
}
