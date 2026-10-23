// Copies the Basic Pitch model and the TF.js wasm binaries out of
// node_modules into public/ml/ (gitignored) before dev/build. Copying from
// the installed packages -- rather than committing the files -- keeps them
// in lockstep with package-lock.json: the model can't drift from
// @spotify/basic-pitch, and the .wasm can't drift from
// @tensorflow/tfjs-backend-wasm's JS, which would fail at runtime.
import { cpSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "ml");

rmSync(out, { recursive: true, force: true });

const modelDir = join(root, "node_modules", "@spotify", "basic-pitch", "model");
cpSync(modelDir, join(out, "basic-pitch"), { recursive: true });

const wasmDir = join(root, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
mkdirSync(join(out, "tfjs-wasm"), { recursive: true });
for (const name of readdirSync(wasmDir).filter((file) => file.endsWith(".wasm"))) {
  cpSync(join(wasmDir, name), join(out, "tfjs-wasm", name));
}

console.log(`[copy-ml-assets] wrote ${out}`);
