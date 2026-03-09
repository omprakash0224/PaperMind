import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RAG-ChatBot",
  description: "Document Q&A powered by RAG",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    /*
      ClerkProvider reads NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY from .env.local
      automatically — no props needed here.

      The `appearance` prop is optional. Remove or customise it to match
      your brand. Full theming docs: https://clerk.com/docs/customization/overview
    */
    <ClerkProvider
    >
      <html lang="en">
        <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
