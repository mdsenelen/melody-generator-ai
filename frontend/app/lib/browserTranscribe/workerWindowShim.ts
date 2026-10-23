// Side-effect module: transcribe.worker.ts imports this FIRST, so the shim
// is in place before TF.js's own modules evaluate -- a call in the worker's
// body would only run after every import had. See installWindowShim.
import { installWindowShim } from "./workerScope";

installWindowShim(self);
