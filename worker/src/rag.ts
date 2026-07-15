/**
 * RAG Engine — Workers AI embedding + DeepSeek LLM
 * Embedding: @cf/baai/bge-base-en-v1.5 (768d, free, Workers AI built-in)
 * Chat: DeepSeek via secret DEEPSEEK_KEY
 * Storage: Cloudflare Vectorize
 */
import type { Env, ChatResponse } from './types';

async function getEmbedding(text: string, env: Env): Promise<number[]> {
  const key = env.DASHSCOPE_KEY || '';
  const resp = await fetch('https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings', {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'text-embedding-v4', input: text.slice(0, 8000) }),
  });
  const data = await resp.json() as any;
  return data.data[0].embedding;
}

async function chatCompletion(
  messages: { role: string; content: string }[],
  env: Env,
  opts?: { maxTokens?: number; temperature?: number }
): Promise<string> {
  const key = env.DEEPSEEK_KEY || '';
  const resp = await fetch('https://api.deepseek.com/v1/chat/completions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages,
      max_tokens: opts?.maxTokens ?? 800,
      temperature: opts?.temperature ?? 0.3,
    }),
  });
  const data = await resp.json() as any;
  return data.choices[0].message.content;
}

export function splitText(text: string, chunkSize = 500): string[] {
  const paragraphs = text.split(/\n\n+/);
  const chunks: string[] = [];
  let current = '';
  for (const p of paragraphs) {
    const trimmed = p.trim();
    if (!trimmed) continue;
    if (current.length + trimmed.length < chunkSize) {
      current = current ? `${current}\n\n${trimmed}` : trimmed;
    } else {
      if (current) chunks.push(current.slice(0, chunkSize));
      current = trimmed;
    }
  }
  if (current) chunks.push(current.slice(0, chunkSize));
  return chunks.length ? chunks : [text.slice(0, chunkSize)];
}

export async function ingestDocument(
  env: Env, namespace: string, title: string, source: string, content: string, chunkSize = 500,
): Promise<number> {
  const chunks = splitText(content, chunkSize);
  const idPrefix = `${namespace}_${btoa(title + source).slice(0, 16)}`;
  const vectors: VectorizeVector[] = [];

  for (let i = 0; i < chunks.length; i++) {
    try {
      const embedding = await getEmbedding(chunks[i], env);
      vectors.push({
        id: `${idPrefix}_${i}`,
        values: embedding,
        metadata: { text: chunks[i], source, title, namespace, chunkIndex: i },
      });
    } catch (e) { console.error(`[rag] embed chunk ${i}:`, e); }
  }

  for (let i = 0; i < vectors.length; i += 100) {
    if ((env as any).VECTORIZE) await env.VECTORIZE.upsert(vectors.slice(i, i + 100));
  }
  return vectors.length;
}


export async function searchAndAnswer(
  env: Env, query: string, _namespace: string,
  history: { role: 'user' | 'assistant'; content: string }[],
  opts?: { maxTokens?: number },
): Promise<ChatResponse> {
  // Direct DeepSeek reply (Vectorize not yet configured)
  const systemPrompt = '你係專業AI客服，用繁體中文簡短回答。如果唔知答案，誠實話俾用戶知。';
  try {
    const reply = await chatCompletion(
      [{ role: 'system', content: systemPrompt }, ...history.slice(-6), { role: 'user', content: query }],
      env, { maxTokens: opts?.maxTokens ?? 800, temperature: 0.3 },
    );
    return { reply: reply.slice(0, 4000), sources: [], confidence: 0.5 };
  } catch (e) {
    return { reply: '⚠️ AI 暫時未能回應，請稍後再試。', sources: [], confidence: 0 };
  }
}


export async function getKBStats(_env: Env, namespace: string): Promise<{ namespace: string; vectorCount: number }> {
  return { namespace, vectorCount: -1 };
}

// ═══════════════════════════════════════════════════════
// Image Processing: qwen-vl-max → text → embed → store
// ═══════════════════════════════════════════════════════

/**
 * Use qwen-vl-max to describe/OCR an image.
 * Returns a detailed text description in Traditional Chinese.
 */
export async function visionDescribe(env: Env, imageBase64: string, mimeType = 'image/png'): Promise<string> {
  const key = env.DASHSCOPE_KEY || '';
  const resp = await fetch('https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen-vl-max',
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mimeType};base64,${imageBase64}` } },
          { type: 'text', text: '請詳細描述呢張圖片嘅所有文字內容、數據、表格。如係文件/菜單/收據，列出所有項目同價錢。用繁體中文。' },
        ],
      }],
      max_tokens: 2000,
    }),
  });
  const data = await resp.json() as any;
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Full pipeline: image → vision → embed → Vectorize store.
 * Returns the description text and number of chunks stored.
 */
export async function ingestImage(
  env: Env,
  namespace: string,
  title: string,
  imageBase64: string,
  mimeType?: string,
): Promise<{ description: string; chunks: number }> {
  const description = await visionDescribe(env, imageBase64, mimeType);
  if (!description) return { description: '', chunks: 0 };

  const chunks = await ingestDocument(env, namespace, title, title, description);
  return { description, chunks };
}
