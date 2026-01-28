/**
 * API 客户端
 */
import type { RAGRequest, RAGResponse, BaseResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = API_BASE_URL + endpoint;
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || 'Request failed');
  }
  return response.json();
}

export async function ragQuery(req: RAGRequest): Promise<RAGResponse> {
  return request('/api/generation/rag', {
    method: 'POST',
    body: JSON.stringify({
      query: req.query,
      file_name: req.fileName,
      top_k: req.topK || 10,
      use_rerank: req.useRerank !== false,
      max_tokens: req.maxTokens || 512,
      temperature: req.temperature || 0.1,
      return_contexts: true,
    }),
  });
}

export async function retrieve(query: string, fileName?: string, topK = 10): Promise<BaseResponse> {
  return request('/api/retrieval/search', {
    method: 'POST',
    body: JSON.stringify({
      query,
      file_name: fileName,
      top_k: topK,
      use_hybrid: true,
      use_rerank: true,
    }),
  });
}

export async function uploadPDF(file: File): Promise<BaseResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(API_BASE_URL + '/api/pdf/upload', {
    method: 'POST',
    body: formData,
  });
  return response.json();
}

export async function parsePDF(fileId: string): Promise<BaseResponse> {
  return request('/api/pdf/parse', {
    method: 'POST',
    body: JSON.stringify({ file_id: fileId, extract_tables: true, extract_bbox: true }),
  });
}

export async function healthCheck(): Promise<BaseResponse> {
  return request('/health');
}
