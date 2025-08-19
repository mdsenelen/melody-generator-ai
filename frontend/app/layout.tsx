import { Share_Tech_Mono, Space_Grotesk } from "next/font/google";
import '../styles/globals.css';
import Header from '../components/header';
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${shareTechMono.variable}`}>
      <body className="min-h-screen bg-gray-50">
        <div className="flex flex-col min-h-screen">
          <div>
            <div className="flex flex-row items-center gap-4 p-4">
              <Header />
              <Nav />
            </div>
            <ErrorBoundary>
              <main className="container mx-auto px-4 py-8">
                {children}
              </main>
            </ErrorBoundary>
          </div>
          <Footer />
        </div>
      </body>
    </html>
  );
}