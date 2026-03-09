/**
 * Clerk middleware — runs at the Next.js Edge before every request.
 *
 * clerkMiddleware() in "default" mode makes all routes PUBLIC unless you
 * explicitly protect them with auth().protect() inside the callback.
 *
 * We protect everything EXCEPT:
 *   • /sign-in and /sign-up  — Clerk's hosted auth pages
 *   • /health                — backend health probe (no auth needed)
 *   • Static assets          — _next/*, favicon, etc.
 *
 * Unauthenticated users hitting a protected route are redirected to /sign-in
 * automatically by Clerk — no manual redirect logic needed in page.tsx.
 */

import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isPublicRoute = createRouteMatcher([
  "/sign-in(.*)",
  "/sign-up(.*)",
]);

export default clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect();
  }
});

export const config = {
  matcher: [
    // Run on all routes except Next.js internals and static files
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
};