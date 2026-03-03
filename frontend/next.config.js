/** @type {import('next').NextConfig} */
const nextConfig = {
  // "standalone" is for Docker; Vercel uses its own build system
  output: process.env.DOCKER_BUILD ? "standalone" : undefined,
  experimental: {
    serverActions: {
      bodySizeLimit: "4.5mb",
    },
  },
};

module.exports = nextConfig;
