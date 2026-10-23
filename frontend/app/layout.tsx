import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const manrope = localFont({
  src: "./fonts/Manrope-Variable.ttf",
  variable: "--font-sans",
  display: "swap",
  weight: "200 800",
});

export const metadata: Metadata = {
  title: "Lumiq | Follow the evidence",
  description:
    "Investigate CSV data with inspectable calculation plans, computed evidence, and clearly separated model interpretation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={manrope.variable}>
      <body
        className="min-h-screen"
        style={{ backgroundColor: "var(--bg)", color: "var(--text)" }}
      >
        {children}
      </body>
    </html>
  );
}
