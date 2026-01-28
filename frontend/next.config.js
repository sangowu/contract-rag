/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  
  // 环境变量
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  
  // 允许跨域图片
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
      },
    ],
  },
  
  // 重写规则 (可选: 代理 API 请求)
  async rewrites() {
    return [
      // 如果需要代理后端 API
      // {
      //   source: '/api/:path*',
      //   destination: 'http://localhost:8000/api/:path*',
      // },
    ];
  },
};

module.exports = nextConfig;
