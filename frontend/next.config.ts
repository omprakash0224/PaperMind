/** @type {import('next').NextConfig} */
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produce a self-contained output in .next/standalone so the Docker image
  // does not need node_modules at runtime. This shrinks the final image by ~80%.
  output: "standalone",
};

export default nextConfig;
