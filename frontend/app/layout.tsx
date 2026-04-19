import type { Metadata } from "next";
import { Share_Tech_Mono, Space_Grotesk } from "next/font/google";
import "../styles/globals.css";
import Header from "../components/header";
import Nav from "../components/nav";
import ErrorBoundary from "../components/error-boundary";
import Footer from "../components/footer";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-space-grotesk",
});

const shareTechMono = Share_Tech_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-share-tech-mono",
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
    <html lang="en" className={`${spaceGrotesk.variable} ${shareTechMono.variable}`}>
      <body className="relative min-h-screen overflow-x-hidden text-white">
        <video
          autoPlay
          loop
          muted
          playsInline
          className="fixed inset-0 -z-10 h-full w-full object-cover"
          style={{ pointerEvents: "none" }}
        >
          <source src="/background.mp4" type="video/mp4" />
        </video>
        <div className="fixed inset-0 -z-10 bg-black/55" />

        <div className="relative z-10">
          <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 sm:px-6 lg:px-8">
            <div className="flex flex-1 flex-col gap-6 py-6">
              <div className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl shadow-black/30 backdrop-blur-md md:flex-row md:items-center md:justify-between">
                <Header />
                <Nav />
              </div>
              <ErrorBoundary>
                <main className="flex-1">{children}</main>
              </ErrorBoundary>
            </div>
            <Footer />
          </div>
        </div>
      </body>
    </html>
  );
}
