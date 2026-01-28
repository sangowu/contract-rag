'use client';

import type { Message, Citation } from '@/lib/types';
import { StreamingText } from './StreamingText';
import { CitationBadge } from '../citation/CitationBadge';

interface Props {
  message: Message;
  onCitationClick?: (citation: Citation) => void;
}

export function MessageItem({ message, onCitationClick }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
        }`}
      >
        {/* 角色标识 */}
        {!isUser && (
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Assistant
          </div>
        )}

        {/* 消息内容 */}
        <div className="whitespace-pre-wrap leading-relaxed">
          {message.isStreaming ? (
            <StreamingText text={message.content} isStreaming={true} />
          ) : (
            message.content
          )}
        </div>

        {/* 引用列表 */}
        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              Sources:
            </div>
            <div className="flex flex-wrap gap-2">
              {message.citations.map((citation) => (
                <CitationBadge
                  key={citation.index}
                  citation={citation}
                  onClick={() => onCitationClick?.(citation)}
                />
              ))}
            </div>
          </div>
        )}

        {/* 性能指标 */}
        {message.metrics && !message.isStreaming && (
          <div className="mt-2 text-xs opacity-60">
            {message.metrics.totalTimeMs && (
              <span>Response time: {message.metrics.totalTimeMs}ms</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
