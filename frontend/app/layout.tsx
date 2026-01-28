import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CUAD Contract Assistant',
  description: 'AI-powered contract analysis and Q&A',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
