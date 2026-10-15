import type { Metadata } from "next";
import { Fraunces, Inter, JetBrains_Mono, Lora } from "next/font/google";
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

const lora = Lora({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-lora",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Melodia",
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
    <html
      lang="en"
      className={`${inter.variable} ${fraunces.variable} ${lora.variable} ${jetbrainsMono.variable}`}
    >
      <body className="bg-background text-foreground min-h-screen overflow-x-hidden font-sans antialiased">
        <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 sm:px-6 lg:px-10">
          <header className="sticky top-0 z-30 flex flex-col gap-3 py-4 backdrop-blur-sm md:flex-row md:items-center md:justify-between">
            <Header />
            <Nav />
          </header>
          <ErrorBoundary>
            <main className="flex-1 py-10">{children}</main>
          </ErrorBoundary>
          <Footer />
        </div>
      </body>
    </html>
  );
}
