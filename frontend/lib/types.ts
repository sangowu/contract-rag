/**
 * CUAD Assistant 类型定义
 */

// =============================================================================
// API 响应类型
// =============================================================================

export interface BaseResponse<T = any> {
  ok: boolean;
  data?: T;
  error?: string;
}

// =============================================================================
// 消息类型
// =============================================================================

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  
  // 流式状态
  isStreaming?: boolean;
  
  // 引用信息
  citations?: Citation[];
  
  // 性能指标
  metrics?: {
    retrievalTimeMs?: number;
    generationTimeMs?: number;
    totalTimeMs?: number;
  };
}

// =============================================================================
// 引用类型
// =============================================================================

export interface Citation {
  index: number;
  chunkId: string;
  fileName: string;
  text?: string;
  pageNum?: number;
  bbox?: BoundingBox;
}

export interface BoundingBox {
  pageNum: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// =============================================================================
// PDF 类型
// =============================================================================

export interface PDFDocument {
  id: string;
  fileName: string;
  fileSize: number;
  pageCount: number;
  status: 'uploading' | 'processing' | 'ready' | 'error';
  uploadTime: Date;
  error?: string;
}

export interface PDFHighlight {
  pageNum: number;
  rects: BoundingBox[];
  color?: string;
  citationIndex?: number;
}

// =============================================================================
// 检索类型
// =============================================================================

export interface RetrievalResult {
  chunkId: string;
  text: string;
  fileName: string;
  pageNum?: number;
  score?: number;
  rerankScore?: number;
  bbox?: BoundingBox;
}

// =============================================================================
// RAG 请求/响应
// =============================================================================

export interface RAGRequest {
  query: string;
  fileName?: string;
  topK?: number;
  useRerank?: boolean;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

export interface RAGResponse {
  ok: boolean;
  answer: string;
  error?: string;
  contexts?: RetrievalResult[];
  citations?: Citation[];
  retrievalTimeMs?: number;
  generationTimeMs?: number;
  totalTimeMs?: number;
}

// =============================================================================
// SSE 事件类型
// =============================================================================

export type SSEEventType = 'status' | 'citations' | 'text' | 'done' | 'error';

export interface SSEEvent {
  type: SSEEventType;
  content: any;
}

// =============================================================================
// 聊天状态
// =============================================================================

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  currentDocument?: PDFDocument;
  selectedCitation?: Citation;
  
  // Actions
  addMessage: (message: Message) => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  appendToMessage: (id: string, text: string) => void;
  setLoading: (loading: boolean) => void;
  setCurrentDocument: (doc: PDFDocument | undefined) => void;
  setSelectedCitation: (citation: Citation | undefined) => void;
  clearMessages: () => void;
}
