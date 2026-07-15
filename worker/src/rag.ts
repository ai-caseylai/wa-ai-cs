/**
 * RAG Engine — LlamaIndex-inspired document ingestion + retrieval
 * Uses Cloudflare Vectorize for storage, external API for embeddings/LLM.
 */
import type { Env, DocChunk, ChatResponse } from './types';

// ═══════════════════════════════════════════════════════
// Embedding: DashScope text-embedding-v4
// ═══════════════════════════════════════════════════════
async function getEmbedding(text: string, env: Env): Promise<number[]> {
  const key = env.DASHSCOPE_API_KEY || '';
  const resp = await fetch(
    'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings',
    {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'text-embedding-v4', input: text.slice(0, 8000) }),
    }
  );
  const data = await resp.json() as any;
  return data.data[0].embedding;
}

// ═══════════════════════════════════════════════════════
// LLM: DeepSeek Chat
// ═══════════════════════════════════════════════════════
async function chatCompletion(
  messages: { role: string; content: string }[],
  env: Env,
  opts?: { maxTokens?: number; temperature?: number }
): Promise<string> {
  const key = env.DEEPSEEK_API_KEY || '';
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

// ═══════════════════════════════════════════════════════
// Document Ingestion (LlamaIndex-style)
// ═══════════════════════════════════════════════════════

/**
 * Split text into chunks.
 * LlamaIndex-inspired: paragraph-aware chunking with overlap.
 */
export function splitText(text: string, chunkSize = 500, overlap = 50): string[] {
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

/**
 * Ingest document into Vectorize index.
 * Returns number of chunks ingested.
 */
export async function ingestDocument(
  env: Env,
  namespace: string,
  title: string,
  source: string,
  content: string,
  chunkSize = 500,
): Promise<number> {
  const chunks = splitText(content, chunkSize);

  // Generate unique ID prefix from title+source
  const idPrefix = `${namespace}_${btoa(title + source).slice(0, 20)}`;

  const vectors: VectorizeVector[] = [];
  for (let i = 0; i < chunks.length; i++) {
    try {
      const embedding = await getEmbedding(chunks[i], env);
      vectors.push({
        id: `${idPrefix}_${i}`,
        values: embedding,
        metadata: {
          text: chunks[i],
          source,
          title,
          namespace,
          chunkIndex: i,
        },
      });
    } catch (e) {
      console.error(`[rag] embed chunk ${i} failed:`, e);
    }
  }

  if (vectors.length > 0) {
    // Upsert in batches of 100
    for (let i = 0; i < vectors.length; i += 100) {
      const batch = vectors.slice(i, i + 100);
      await env.VECTORIZE.upsert(batch);
    }
  }

  return vectors.length;
}

// ═══════════════════════════════════════════════════════
// RAG Query
// ═══════════════════════════════════════════════════════

/**
 * Search knowledge base and generate answer.
 */
export async function searchAndAnswer(
  env: Env,
  query: string,
  namespace: string,
  history: { role: 'user' | 'assistant'; content: string }[],
  opts?: { topK?: number; scoreThreshold?: number; maxTokens?: number },
): Promise<ChatResponse> {
  const topK = opts?.topK ?? 5;
  const threshold = opts?.scoreThreshold ?? 0.3;
  const maxTokens = opts?.maxTokens ?? 800;

  // Step 1: Rewrite query for better search
  let searchQuery = query;
  try {
    searchQuery = await chatCompletion(
      [
        { role: 'system', content: '將口語改寫成 5-8 個搜尋關鍵詞，只輸出關鍵詞空格分隔。' },
        { role: 'user', content: query },
      ],
      env,
      { maxTokens: 80, temperature: 0.1 },
    );
    searchQuery = searchQuery.trim() || query;
  } catch {
    // Fallback to original query
  }

  // Step 2: Get embedding and search Vectorize
  let results: VectorizeMatch[] = [];
  try {
    const queryEmbedding = await getEmbedding(searchQuery, env);
    const searchResults = await env.VECTORIZE.query(queryEmbedding, {
      topK,
      returnMetadata: true,
      filter: { namespace },
    });
    results = searchResults.matches.filter((m: VectorizeMatch) => m.score >= threshold);
  } catch (e) {
    console.error('[rag] search error:', e);
  }

  // Step 3: No results
  if (results.length === 0) {
    return {
      reply: '唔好意思，我喺知識庫入面搵唔到相關嘅資訊。你可以換個方式問，或者聯絡真人客服幫你。 🙏',
      sources: [],
      confidence: 0,
    };
  }

  // Step 4: Build context and generate answer
  const chunksText = results
    .map((r, i) => `[${i + 1}] ${(r.metadata as any)?.text?.slice(0, 800) || ''}`)
    .join('\n\n---\n\n');

  const systemPrompt = `你係專業嘅 AI 客服助手，用繁體中文簡短回答。根據以下知識庫內容回答：

${chunksText}

規則：
- 只根據知識庫內容回答，唔好自己創作
- 如果資料不足，誠實告知
- 回答簡潔，適合 WhatsApp 閱讀`;

  const messages: { role: string; content: string }[] = [
    { role: 'system', content: systemPrompt },
    ...history.slice(-6),
    { role: 'user', content: query },
  ];

  let reply: string;
  try {
    reply = await chatCompletion(messages, env, { maxTokens, temperature: 0.3 });
  } catch {
    // Fallback: return top chunk
    reply = `根據知識庫：\n\n${(results[0].metadata as any)?.text?.slice(0, 500) || ''}`;
  }

  const sources = results.slice(0, 3).map((r) => ({
    text: ((r.metadata as any)?.text || '').slice(0, 200),
    score: r.score,
  }));

  // Append source notes
  const sourceNames = [...new Set(results.map((r) => (r.metadata as any)?.title).filter(Boolean))];
  if (sourceNames.length > 0) {
    reply += `\n\n📚 來源：${sourceNames.slice(0, 3).join('、')}`;
  }

  return {
    reply: reply.slice(0, 4000),
    sources,
    confidence: results[0]?.score || 0,
  };
}

/**
 * Get knowledge base stats.
 */
export async function getKBStats(env: Env, namespace: string): Promise<{
  namespace: string;
  vectorCount: number;
}> {
  try {
    const info = await env.VECTORIZE.describe();
    return {
      namespace,
      vectorCount: info.dimensions ? 0 : 0, // Describe doesn't give count; use DB
    };
  } catch {
    return { namespace, vectorCount: -1 };
  }
}
