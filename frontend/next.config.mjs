/** @type {import('next').NextConfig} */
const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
const publicApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    // When the browser talks to /api, proxy to the local/backend service in dev.
    if (/^https?:\/\//.test(publicApiBaseUrl)) {
      return [];
    }

    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
}

export default nextConfig
