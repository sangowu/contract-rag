'use client';

import { useState, useCallback, useRef } from 'react';
import { uploadPDF, parsePDF } from '@/lib/api';

interface Props {
  onUploadComplete?: (fileId: string, fileName: string) => void;
  onError?: (error: string) => void;
}

export function PDFUploader({ onUploadComplete, onError }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      onError?.('Please upload a PDF file');
      return;
    }

    setIsUploading(true);
    setProgress(10);

    try {
      // 上传文件
      const uploadResult = await uploadPDF(file);
      setProgress(50);

      if (!uploadResult.ok) {
        throw new Error(uploadResult.error || 'Upload failed');
      }

      const fileId = (uploadResult as any).file_id;

      // 解析文件
      setProgress(70);
      const parseResult = await parsePDF(fileId);
      setProgress(100);

      if (!parseResult.ok) {
        throw new Error(parseResult.error || 'Parse failed');
      }

      onUploadComplete?.(fileId, file.name);
    } catch (error) {
      onError?.((error as Error).message);
    } finally {
      setIsUploading(false);
      setProgress(0);
    }
  }, [onUploadComplete, onError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
        transition-colors duration-200
        ${isDragging 
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
          : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
        }
        ${isUploading ? 'pointer-events-none opacity-70' : ''}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
      />

      {isUploading ? (
        <div className="space-y-3">
          <div className="w-12 h-12 mx-auto border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Processing... {progress}%
          </p>
          <div className="w-48 mx-auto h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="w-12 h-12 mx-auto text-gray-400">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
          </div>
          <p className="text-gray-600 dark:text-gray-400">
            <span className="text-blue-600 dark:text-blue-400 font-medium">Click to upload</span>
            {' '}or drag and drop
          </p>
          <p className="text-xs text-gray-500">PDF files only</p>
        </div>
      )}
    </div>
  );
}
