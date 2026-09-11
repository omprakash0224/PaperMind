# ── Stage 1: deps ─────────────────────────────────────────────────────────────
# Install production node_modules only.
FROM node:20-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev


# ── Stage 2: builder ──────────────────────────────────────────────────────────
# Build the Next.js app with all devDependencies available.
FROM node:20-alpine AS builder

WORKDIR /app

# Copy deps from previous stage (includes devDeps needed by next build)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source
COPY . .

# Build args are baked into the image at build time for public env vars.
# Secret keys (CLERK_SECRET_KEY etc.) are NOT baked in — pass at runtime.
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_API_URL=http://fastapi:8000

ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build


# ── Stage 3: runtime ──────────────────────────────────────────────────────────
# Minimal image — only the standalone server output.
FROM node:20-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Non-root user
RUN addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 nextjs

# Copy the standalone bundle produced by `output: "standalone"` in next.config
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static    ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public          ./public

USER nextjs

EXPOSE 3000

# next.js standalone entry point
CMD ["node", "server.js"]
