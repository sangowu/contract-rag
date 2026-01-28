'use client';

import type { Citation } from '@/lib/types';

interface Props {
  citation: Citation;
  onClick?: () => void;
}

export function CitationBadge({ citation, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-200 dark:hover:bg-blue-800 transition-colors cursor-pointer"
    >
      <span className="font-semibold">[{citation.index}]</span>
      <span className="truncate max-w-[100px]">{citation.fileName}</span>
      {citation.pageNum && <span>p.{citation.pageNum}</span>}
    </button>
  );
}
