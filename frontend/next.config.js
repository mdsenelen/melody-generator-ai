/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produces a self-contained .next/standalone build (minimal node_modules
  // subset + a server.js entrypoint) so the production Docker image doesn't
  // need to ship the full node_modules tree or run `next dev`/`next build`
  // at container start. See frontend/Dockerfile.
  output: "standalone",
};

module.exports = nextConfig;
