import type { Metadata } from "next";
import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";
import "../styles/globals.css";
import Header from "../components/header";
import Nav from "../components/nav";
import ErrorBoundary from "../components/error-boundary";
import Footer from "../components/footer";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  style: ["normal", "italic"],
  variable: "--font-fraunces",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MelodyAI",
  description: "Analyse melodies, render progressions, and generate music variants.",
  icons: {
    icon: "/logo.svg",
    apple: "/logo.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-background text-foreground min-h-screen overflow-x-hidden font-sans antialiased">
        <div className="flex min-h-screen w-full flex-col">
          <header className="border-border flex flex-col gap-3 border-b px-6 py-4 md:flex-row md:items-center md:justify-between md:px-10">
            <Header />
            <Nav />
          </header>
          <ErrorBoundary>
            <main className="flex-1 px-4 py-8 sm:px-6 lg:px-10">{children}</main>
          </ErrorBoundary>
          <Footer />
        </div>
      </body>
    </html>
  );
}
