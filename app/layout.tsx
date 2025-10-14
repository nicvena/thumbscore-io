import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import AnalyticsProvider from "@/components/AnalyticsProvider";
import AnalyticsErrorBoundary from "@/components/AnalyticsErrorBoundary";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Thumbscore.io — AI Thumbnail Scoring",
  description: "Get your YouTube thumbnails AI-scored in seconds. Data-backed predictions, real-world accuracy.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AnalyticsErrorBoundary>
          <AnalyticsProvider>
            {children}
          </AnalyticsProvider>
        </AnalyticsErrorBoundary>
      </body>
    </html>
  );
}
