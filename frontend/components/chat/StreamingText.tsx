'use client';

import { useEffect, useState } from 'react';

interface Props {
  text: string;
  isStreaming: boolean;
}

export function StreamingText({ text, isStreaming }: Props) {
  const [displayText, setDisplayText] = useState('');
  const [cursorVisible, setCursorVisible] = useState(true);

  useEffect(() => {
    setDisplayText(text);
  }, [text]);

  useEffect(() => {
    if (!isStreaming) return;
    
    const interval = setInterval(() => {
      setCursorVisible((v) => !v);
    }, 500);
    
    return () => clearInterval(interval);
  }, [isStreaming]);

  return (
    <span>
      {displayText}
      {isStreaming && (
        <span className={`transition-opacity ${cursorVisible ? 'opacity-100' : 'opacity-0'}`}>
          ▊
        </span>
      )}
    </span>
  );
}
