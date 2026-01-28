'use client';

import { useRef, useEffect } from 'react';
import type { Message, Citation } from '@/lib/types';
import { MessageItem } from './MessageItem';

interface Props {
  messages: Message[];
  onCitationClick?: (citation: Citation) => void;
}

export function MessageList({ messages, onCitationClick }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && (
        <div className="text-center text-gray-500 dark:text-gray-400 mt-20">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 text-blue-600 dark:text-blue-400">
              <path fillRule="evenodd" d="M4.848 2.771A49.144 49.144 0 0 1 12 2.25c2.43 0 4.817.178 7.152.52 1.978.292 3.348 2.024 3.348 3.97v6.02c0 1.946-1.37 3.678-3.348 3.97a48.901 48.901 0 0 1-3.476.383.39.39 0 0 0-.297.17l-2.755 4.133a.75.75 0 0 1-1.248 0l-2.755-4.133a.39.39 0 0 0-.297-.17 48.9 48.9 0 0 1-3.476-.384c-1.978-.29-3.348-2.024-3.348-3.97V6.741c0-1.946 1.37-3.68 3.348-3.97Z" clipRule="evenodd" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold mb-2 text-gray-900 dark:text-gray-100">
            CUAD Contract Assistant
          </h2>
          <p className="text-sm max-w-md mx-auto">
            Ask questions about your contract documents. I can help you find specific clauses, 
            understand terms, and analyze contract details.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {[
              'What is the termination clause?',
              'When does the contract expire?',
              'What are the payment terms?',
            ].map((q) => (
              <button
                key={q}
                className="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {messages.map((message) => (
        <MessageItem 
          key={message.id} 
          message={message}
          onCitationClick={onCitationClick}
        />
      ))}
      
      <div ref={bottomRef} />
    </div>
  );
}
