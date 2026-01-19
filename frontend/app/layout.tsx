import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Lyric-Audio Alignment | Antigravity',
  description: 'A Comparative Study of Lyric-Audio Alignment Methods: DTW, HMM, and Deep Learning',
  keywords: ['lyrics', 'audio', 'alignment', 'music', 'karaoke', 'machine learning'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
