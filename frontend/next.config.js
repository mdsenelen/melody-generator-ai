/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Next 16's `next dev` auto-generates AGENTS.md/CLAUDE.md in this
  // directory unless disabled; the project already has its own CLAUDE.md
  // convention at the repo root.
  agentRules: false,
  // Produces a self-contained .next/standalone build (minimal node_modules
  // subset + a server.js entrypoint) so the production Docker image doesn't
  // need to ship the full node_modules tree or run `next dev`/`next build`
  // at container start. See frontend/Dockerfile.
  output: "standalone",
  // pitchy ships as pure ESM; without this, next/jest's default
  // node_modules transform-ignore blocks it during component tests that
  // import hooks/use-audio-analyzer.ts (e.g. via AudioRecorder).
  transpilePackages: ["pitchy"],
};

module.exports = nextConfig;
