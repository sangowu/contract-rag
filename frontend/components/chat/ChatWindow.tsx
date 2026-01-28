'use client';

import { useCallback } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { streamRAG } from '@/lib/sse';
import { MessageList } from './MessageList';
import { InputArea } from './InputArea';
import type { Message, Citation } from '@/lib/types';

export function ChatWindow() {
  const { 
    messages, 
    isLoading,
    currentDocument,
    addMessage, 
    updateMessage, 
    appendToMessage,
    setLoading,
    setSelectedCitation,
  } = useChatStore();

  const handleSend = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return;

    // 添加用户消息
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    // 添加助手消息 (流式)
    const assistantId = `assistant-${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };
    addMessage(assistantMessage);
    setLoading(true);

    // 开始流式生成
    const startTime = Date.now();

    streamRAG(query, {
      onStatus: (status) => {
        console.log('Status:', status);
      },
      onCitations: (cits) => {
        const citations: Citation[] = cits.map((c: any, i: number) => ({
          index: i + 1,
          chunkId: c.chunk_id || c.chunkId,
          fileName: c.file_name || c.fileName,
          pageNum: c.page_num || c.pageNum,
        }));
        updateMessage(assistantId, { citations });
      },
      onText: (text) => {
        appendToMessage(assistantId, text);
      },
      onDone: () => {
        const totalTime = Date.now() - startTime;
        updateMessage(assistantId, {
          isStreaming: false,
          metrics: { totalTimeMs: totalTime },
        });
        setLoading(false);
      },
      onError: (error) => {
        updateMessage(assistantId, {
          content: assistantMessage.content || `Error: ${error}`,
          isStreaming: false,
        });
        setLoading(false);
      },
    }, {
      fileName: currentDocument?.fileName,
    });
  }, [isLoading, currentDocument, addMessage, updateMessage, appendToMessage, setLoading]);

  const handleCitationClick = useCallback((citation: Citation) => {
    setSelectedCitation(citation);
    // TODO: 滚动到 PDF 对应位置
    console.log('Citation clicked:', citation);
  }, [setSelectedCitation]);

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900">
      <MessageList 
        messages={messages} 
        onCitationClick={handleCitationClick}
      />
      <InputArea 
        onSend={handleSend} 
        disabled={isLoading}
        placeholder={currentDocument 
          ? `Ask about ${currentDocument.fileName}...` 
          : "Ask a question about your contract..."
        }
      />
    </div>
  );
}
