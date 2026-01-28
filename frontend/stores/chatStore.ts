/**
 * 聊天状态管理
 */
import { create } from 'zustand';
import type { ChatState, Message, Citation, PDFDocument } from '@/lib/types';

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  currentDocument: undefined,
  selectedCitation: undefined,

  addMessage: (message: Message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  updateMessage: (id: string, updates: Partial<Message>) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, ...updates } : m
      ),
    })),

  appendToMessage: (id: string, text: string) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + text } : m
      ),
    })),

  setLoading: (loading: boolean) => set({ isLoading: loading }),
  
  setCurrentDocument: (doc: PDFDocument | undefined) => set({ currentDocument: doc }),
  
  setSelectedCitation: (citation: Citation | undefined) => set({ selectedCitation: citation }),
  
  clearMessages: () => set({ messages: [] }),
}));
