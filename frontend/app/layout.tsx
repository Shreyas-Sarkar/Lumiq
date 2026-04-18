import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lumiq — Deterministic AI Data Analysis",
  description:
    "Lumiq executes real code on your dataset to generate answers, charts, and insights. Every result is grounded in execution, not probability.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen" style={{ backgroundColor: "var(--bg)", color: "var(--text)" }}>
        {children}
      </body>
    </html>
  );
}
