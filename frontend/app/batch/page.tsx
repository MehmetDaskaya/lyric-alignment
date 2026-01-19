'use client';

import BatchProcessingPanel from '@/components/BatchProcessingPanel';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function BatchPage() {
  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      {/* Header */}
      <header className="mb-8">
        <Link 
          href="/" 
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Single Song Mode
        </Link>
        
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
          Batch Processing
        </h1>
        <p className="text-gray-400 mt-2">
          Run alignment models on multiple songs and compare results
        </p>
      </header>

      {/* Batch Panel */}
      <BatchProcessingPanel />
    </main>
  );
}
