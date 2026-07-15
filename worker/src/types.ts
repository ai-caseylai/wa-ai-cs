/** Environment bindings */
export interface Env {
  WHATSAPP_DO: DurableObjectNamespace<WhatsAppDO>;
  CONVERSATION_DO: DurableObjectNamespace;
  VECTORIZE: VectorizeIndex;
  DB: D1Database;
  MEDIA_BUCKET: R2Bucket;
  ADMIN_PASSWORD: string;
  DEEPSEEK_API_KEY: string;
  DASHSCOPE_API_KEY: string;
}

/** Standardized incoming message */
export interface ChatMessage {
  sender: string;
  senderName: string;
  text: string;
  isGroup: boolean;
  groupId?: string;
  msgType: 'text' | 'image' | 'audio' | 'video' | 'document';
  mediaUrl?: string;
  mediaMime?: string;
  timestamp: number;
}

/** RAG document chunk */
export interface DocChunk {
  id: string;
  text: string;
  source: string;
  title: string;
  namespace: string;
  embedding?: number[];
}

/** Conversation state in Durable Object */
export interface ConversationState {
  messages: { role: 'user' | 'assistant' | 'system'; content: string }[];
  kbNamespace: string;
  lastActive: number;
}

/** Admin API: ingest request */
export interface IngestRequest {
  namespace: string;
  title: string;
  source?: string;
  content: string;
  chunks?: string[];
}

/** Chat API response */
export interface ChatResponse {
  reply: string;
  sources?: { text: string; score: number }[];
  confidence?: number;
}

/** WhatsApp DO interface (RPC) */
export interface WhatsAppDO {
  sendText(phone: string, text: string): Promise<{ success: boolean; id?: string }>;
  getStatus(): Promise<{ connected: boolean; jid?: string }>;
  getQR(): Promise<string | null>;
}
