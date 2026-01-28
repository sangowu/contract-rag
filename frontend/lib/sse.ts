/**
 * SSE 流式客户端
 */
import type { SSEEvent, Citation } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface StreamCallbacks {
  onStatus?: (message: string) => void;
  onCitations?: (citations: Citation[]) => void;
  onText?: (text: string) => void;
  onDone?: () => void;
  onError?: (error: string) => void;
}

export function streamRAG(
  query: string,
  callbacks: StreamCallbacks,
  options?: {
    fileName?: string;
    topK?: number;
    maxTokens?: number;
  }
): () => void {
  const controller = new AbortController();
  
  const fetchStream = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/generation/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          file_name: options?.fileName,
          top_k: options?.topK || 10,
          max_tokens: options?.maxTokens || 512,
          use_rerank: true,
        }),
        signal: controller.signal,
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: SSEEvent = JSON.parse(line.slice(6));
              
              switch (event.type) {
                case 'status':
                  callbacks.onStatus?.(event.content);
                  break;
                case 'citations':
                  callbacks.onCitations?.(event.content);
                  break;
                case 'text':
                  callbacks.onText?.(event.content);
                  break;
                case 'done':
                  callbacks.onDone?.();
                  break;
                case 'error':
                  callbacks.onError?.(event.content);
                  break;
              }
            } catch (e) {
              // 忽略解析错误
            }
          }
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        callbacks.onError?.((error as Error).message);
      }
    }
  };

  fetchStream();

  // 返回取消函数
  return () => controller.abort();
}
