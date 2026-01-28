# CUAD Contract Assistant - Frontend

AI-powered contract analysis and Q&A frontend built with Next.js 14.

## Features

- Real-time streaming responses (SSE)
- PDF upload and processing
- Citation highlighting
- Dark mode support
- Responsive design

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State**: Zustand
- **Streaming**: Server-Sent Events (SSE)

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Configuration

Create a `.env.local` file:

```bash
cp .env.example .env.local
```

Edit the API URL:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                  # Next.js App Router
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Home page
│   └── globals.css      # Global styles
├── components/
│   ├── chat/            # Chat components
│   │   ├── ChatWindow   # Main chat container
│   │   ├── MessageList  # Message list
│   │   ├── MessageItem  # Single message
│   │   ├── InputArea    # Input field
│   │   └── StreamingText# Streaming text display
│   ├── pdf/             # PDF components
│   │   └── PDFUploader  # PDF upload
│   └── citation/        # Citation components
│       └── CitationBadge# Citation badge
├── lib/
│   ├── api.ts           # API client
│   ├── sse.ts           # SSE streaming client
│   └── types.ts         # TypeScript types
├── stores/
│   └── chatStore.ts     # Zustand state
└── styles/
    └── globals.css      # Global CSS
```

## API Integration

The frontend connects to the backend API at `NEXT_PUBLIC_API_URL`.

### Endpoints Used

- `POST /api/generation/stream` - Streaming RAG query
- `POST /api/generation/rag` - Non-streaming RAG query
- `POST /api/retrieval/search` - Document retrieval
- `POST /api/pdf/upload` - PDF upload
- `POST /api/pdf/parse` - PDF parsing
- `GET /health` - Health check

## Deployment

### Vercel (Recommended)

```bash
npm i -g vercel
vercel
```

### Static Export

```bash
npm run build
# Output in .next/standalone
```

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

## License

MIT
