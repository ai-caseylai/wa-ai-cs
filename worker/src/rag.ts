/**
 * RAG Engine v2 — LlamaIndex-style 進階檢索
 * 
 * 功能：
 *   - Hybrid Search (BM25 關鍵詞 + 向量語義)
 *   - Reciprocal Rank Fusion 混合排序
 *   - LLM Re-ranking (DeepSeek 重新打分)
 *   - 三語 Query 優化 (廣東話/繁中/英文)
 *   - 自動語言檢測 + 同語言回覆
 */
import type { Env, ChatResponse } from './types';

// ═══════════════════════════════════════════════════════
// 1. Embedding: DashScope text-embedding-v4 (多語言)
// ═══════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════
// 2. Multi-Model LLM (DeepSeek / Qwen / GLM)
// ═══════════════════════════════════════════════════════

type ModelProvider = 'deepseek' | 'qwen' | 'glm';

interface ModelConfig {
  base: string;
  key: string;
  model: string;
}

function getModelConfig(provider: ModelProvider, env: Env): ModelConfig {
  switch (provider) {
    case 'qwen':
      return { base: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', key: env.DASHSCOPE_KEY || '', model: 'qwen3-max' };
    case 'glm':
      return { base: 'https://api.z.ai/api/paas/v4', key: env.GLM_KEY || '', model: 'glm-5.2' };
    default:
      return { base: 'https://api.deepseek.com/v1', key: env.DEEPSEEK_KEY || '', model: 'deepseek-chat' };
  }
}

/**
 * Chat completion with model selection.
 * 
 * 策略：
 *   - Re-rank → DeepSeek (最快、最準)
 *   - 粵語生成 → Qwen (最地道)
 *   - 預設 → DeepSeek (最穩定)
 */
async function chatCompletion(
  messages: { role: string; content: string }[],
  env: Env,
  opts?: { maxTokens?: number; temperature?: number; provider?: ModelProvider },
): Promise<string> {
  const p = opts?.provider || 'deepseek';
  const cfg = getModelConfig(p, env);
  if (!cfg.key) return '';

  // GLM needs higher max_tokens (reasoning uses tokens too)
  const actualMaxTokens = p === 'glm' 
    ? Math.max(opts?.maxTokens ?? 800, 2000)  // GLM: at least 2000
    : (opts?.maxTokens ?? 800);

  try {
    const resp = await fetch(`${cfg.base}/chat/completions`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${cfg.key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: cfg.model,
        messages,
        max_tokens: actualMaxTokens,
        temperature: opts?.temperature ?? 0.3,
      }),
    });
    const data = await resp.json() as any;
    const err = data.error;
    if (err) {
      console.error(`[llm:${p}] ${err.code}: ${err.message}`);
      // Fallback chain: GLM → Qwen → DeepSeek
      if (p === 'glm') return chatCompletion(messages, env, { ...opts, provider: 'qwen' });
      if (p === 'qwen') return chatCompletion(messages, env, { ...opts, provider: 'deepseek' });
      throw new Error(err.message || 'LLM error');
    }
    
    // GLM sometimes returns reasoning_content only, extract actual content
    let content = data.choices[0].message.content;
    if (!content && data.choices[0].message.reasoning_content) {
      // Use reasoning as fallback content
      content = data.choices[0].message.reasoning_content.slice(-500);
    }
    return content || '';
  } catch (e: any) {
    console.error(`[llm:${p}] fetch error:`, e.message);
    if (p === 'glm') return chatCompletion(messages, env, { ...opts, provider: 'qwen' });
    if (p === 'qwen') return chatCompletion(messages, env, { ...opts, provider: 'deepseek' });
    throw e;
  }
}

// ═══════════════════════════════════════════════════════
// 3. 三語檢測
// ═══════════════════════════════════════════════════════
type Lang = 'yue' | 'zh-Hant' | 'zh-Hans' | 'en';

function detectLang(text: string): Lang {
  // Cantonese-specific particles
  if (/[嘅咩喺啲哋嚟嗰咦唷啱冇佢哋]/u.test(text)) return 'yue';
  // Simplified Chinese (GB characters)
  if (/[什么是个们这那进过关体验]/u.test(text)) return 'zh-Hans';
  // Traditional Chinese (Big5 range)
  if (/[\u4e00-\u9fff]/.test(text)) return 'zh-Hant';
  return 'en';
}

const LANG_LABELS: Record<Lang, string> = {
  'yue': '廣東話',
  'zh-Hant': '繁體中文',
  'zh-Hans': '簡體中文',
  'en': 'English',
};

// ═══════════════════════════════════════════════════════
// 4. BM25 關鍵詞搜尋 (LlamaIndex-style Hybrid)
// ═══════════════════════════════════════════════════════

/** Simple BM25-like TF-IDF scoring */
function bm25Score(query: string, doc: string, avgLen: number, k1 = 1.5, b = 0.75): number {
  const queryTerms = tokenize(query);
  const docTerms = tokenize(doc);
  const docLen = docTerms.length;
  if (docLen === 0) return 0;

  let score = 0;
  const docFreq = new Map<string, number>();
  for (const t of docTerms) docFreq.set(t, (docFreq.get(t) || 0) + 1);

  for (const qt of queryTerms) {
    const tf = docFreq.get(qt) || 0;
    if (tf === 0) continue;
    // Simplified IDF (assuming term appears in 1 doc for local scoring)
    const idf = Math.log(1 + 1);
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + b * (docLen / avgLen));
    score += idf * (numerator / denominator);
  }
  return score;
}

/** Tokenize for CJK + English */
function tokenize(text: string): string[] {
  // Split CJK characters individually, keep English words together
  const tokens: string[] = [];
  let buf = '';
  for (const ch of text) {
    if (/[\u4e00-\u9fff\u3400-\u4dbf]/.test(ch)) {
      if (buf) { tokens.push(buf.toLowerCase()); buf = ''; }
      tokens.push(ch);
    } else if (/[a-zA-Z0-9]/.test(ch)) {
      buf += ch;
    } else {
      if (buf) { tokens.push(buf.toLowerCase()); buf = ''; }
    }
  }
  if (buf) tokens.push(buf.toLowerCase());
  return tokens.filter(t => t.length > 0);
}

// ═══════════════════════════════════════════════════════
// 5. Hybrid Search + Re-rank
// ═══════════════════════════════════════════════════════

interface ScoredDoc {
  text: string;
  source: string;
  title: string;
  score: number;
}

/**
 * Reciprocal Rank Fusion: combine BM25 + Vector rankings.
 */
function rrf(results: ScoredDoc[][], k = 60): ScoredDoc[] {
  const scoreMap = new Map<string, { doc: ScoredDoc; score: number }>();
  for (const rankList of results) {
    for (let i = 0; i < rankList.length; i++) {
      const key = rankList[i].text.slice(0, 100);
      const existing = scoreMap.get(key);
      const rrfScore = 1 / (k + i + 1);
      if (existing) {
        existing.score += rrfScore;
      } else {
        scoreMap.set(key, { doc: rankList[i], score: rrfScore });
      }
    }
  }
  return [...scoreMap.values()].sort((a, b) => b.score - a.score).map(v => ({ ...v.doc, score: v.score }));
}

/**
 * LLM Re-ranking: use DeepSeek to re-rank top candidates.
 */
async function llmRerank(
  query: string,
  candidates: ScoredDoc[],
  env: Env,
  topN = 5,
): Promise<ScoredDoc[]> {
  if (candidates.length <= topN) return candidates;

  const candidateText = candidates.map((c, i) => `[${i}] ${c.text.slice(0, 300)}`).join('\n\n');
  const prompt = `你係一個搜尋排序專家。根據用戶問題，對以下文檔片段進行相關性排序。
只輸出最相關的 ${topN} 個編號（用逗號分隔，如：3,1,5,2,4），不要解釋。

用戶問題：${query}

文檔片段：
${candidateText}

最相關的 ${topN} 個編號：`;

  try {
    const result = await chatCompletion(
      [{ role: 'user', content: prompt }],
      env,
      { maxTokens: 50, temperature: 0, provider: 'deepseek' },
    );
    const indices = result.match(/\d+/g)?.map(Number) || [];
    const reranked = indices
      .filter(i => i >= 0 && i < candidates.length)
      .map(i => candidates[i]);
    // Fill remaining with next best
    for (const c of candidates) {
      if (reranked.length >= topN) break;
      if (!reranked.find(r => r.text === c.text)) reranked.push(c);
    }
    return reranked.slice(0, topN);
  } catch {
    return candidates.slice(0, topN);
  }
}

// ═══════════════════════════════════════════════════════
// 6. 三語 Query 優化
// ═══════════════════════════════════════════════════════

/**
 * 粵語優化 Query Expansion (LlamaIndex-style)
 * 
 * 策略：
 *   粵語輸入 → ①原文 + ②書面語正規化 + ③英文翻譯 → 3個 queries
 *   英文輸入 → ①原文 + ②繁體翻譯 + ③簡體翻譯 → 3個 queries
 *   繁體輸入 → ①原文 + ②廣東話轉換 + ③英文 → 3個 queries
 * 
 * 多個 query 分別做 embedding search，結果用 RRF 融合。
 */
async function expandQuery(query: string, env: Env): Promise<string[]> {
  const lang = detectLang(query);
  const queries = [query]; // 原文 always included

  if (lang === 'yue') {
    // 粵語 → 正規化為書面語（最重要！提升 embedding 準確度）
    const normalized = await normalizeCantonese(query, env);
    if (normalized !== query) queries.push(normalized);
    // 也加入英文翻譯
    try {
      const en = await chatCompletion(
        [{ role: 'user', content: `Translate to English (just the translation): ${query}` }],
        env, { maxTokens: 100, temperature: 0.1 },
      );
      if (en.trim() && !queries.includes(en.trim())) queries.push(en.trim());
    } catch {}
  } else if (lang === 'en') {
    try {
      const zh = await chatCompletion(
        [{ role: 'user', content: `Translate to Traditional Chinese (just the translation): ${query}` }],
        env, { maxTokens: 100, temperature: 0.1 },
      );
      if (zh.trim()) queries.push(zh.trim());
      const simp = await chatCompletion(
        [{ role: 'user', content: `Translate to Simplified Chinese (just the translation): ${query}` }],
        env, { maxTokens: 100, temperature: 0.1 },
      );
      if (simp.trim() && simp !== zh.trim()) queries.push(simp.trim());
    } catch {}
  } else {
    // 繁中/簡中 → 加上英文
    try {
      const en = await chatCompletion(
        [{ role: 'user', content: `Translate to English (just the translation): ${query}` }],
        env, { maxTokens: 100, temperature: 0.1 },
      );
      if (en.trim()) queries.push(en.trim());
    } catch {}
  }

  return queries.slice(0, 5);
}

// ═══════════════════════════════════════════════════════
// 7. Main RAG Pipeline
// ═══════════════════════════════════════════════════════

export async function searchAndAnswer(
  env: Env,
  query: string,
  namespace: string,
  history: { role: 'user' | 'assistant'; content: string }[],
  opts?: { topK?: number; scoreThreshold?: number; maxTokens?: number },
): Promise<ChatResponse> {
  const topK = opts?.topK ?? 5;
  const maxTokens = opts?.maxTokens ?? 800;
  const userLang = detectLang(query);

  // Step 1: Expand query for trilingual search
  const queries = await expandQuery(query, env);

  // Step 2: Multi-query Vector Search (每個 query variant 獨立搜尋)
  const allVectorResults: ScoredDoc[][] = [];

  for (const q of queries.slice(0, 3)) { // 最多 3 個 variant
    try {
      const queryEmbedding = await getEmbedding(q, env);
      if ((env as any).VECTORIZE) {
        const sr = await (env as any).VECTORIZE.query(queryEmbedding, {
          topK: topK * 2,
          returnMetadata: true,
          filter: { namespace },
        });
        const results: ScoredDoc[] = [];
        for (const m of sr.matches) {
          if (m.metadata?.text) {
            results.push({
              text: m.metadata.text,
              source: m.metadata.source || '',
              title: m.metadata.title || '',
              score: m.score,
            });
          }
        }
        if (results.length > 0) allVectorResults.push(results);
      }
    } catch (e) {
      console.error(`[rag] vector search for "${q.slice(0, 30)}":`, e);
    }
  }

  // BM25 search on all collected docs (if available)
  const bm25Results: ScoredDoc[] = [];
  const allDocs = [...new Set(allVectorResults.flat().map(r => r.text))];
  if (allDocs.length > 0) {
    const avgLen = allDocs.reduce((s, d) => s + d.length, 0) / allDocs.length;
    for (const doc of allDocs) {
      let maxScore = 0;
      for (const q of queries.slice(0, 2)) {
        const s = bm25Score(q, doc, avgLen);
        if (s > maxScore) maxScore = s;
      }
      if (maxScore > 0) {
        bm25Results.push({ text: doc, source: '', title: '', score: maxScore });
      }
    }
    bm25Results.sort((a, b) => b.score - a.score);
  }

  // RRF Fusion: BM25 + all vector query results
  const allRankLists = [bm25Results.slice(0, topK * 3)];
  for (const vr of allVectorResults) {
    allRankLists.push(vr.slice(0, topK * 3));
  }
  const vectorResults = rrf(allRankLists);

  // Filter empty results
  const validResults = vectorResults.filter(r => r.text);

  // Step 3: LLM Re-ranking
  let finalResults = validResults;
  if (validResults.length > topK) {
    finalResults = await llmRerank(query, validResults, env, topK);
  }

  // Step 4: No results → direct LLM answer
  if (finalResults.length === 0) {
    const langHint = userLang === 'yue' ? '廣東話' : userLang === 'zh-Hant' ? '繁體中文' : userLang === 'zh-Hans' ? '簡體中文' : 'English';
    try {
      const reply = await chatCompletion(
        [
          { role: 'system', content: `你係專業AI客服，用${langHint}回答。如果唔知答案，誠實話俾用戶知。` },
          ...history.slice(-6),
          { role: 'user', content: query },
        ],
        env,
        { maxTokens, temperature: 0.3 },
      );
      return { reply: reply.slice(0, 4000), sources: [], confidence: 0.3 };
    } catch {
      return { reply: '唔好意思，我暫時未能回應。請稍後再試。 🙏', sources: [], confidence: 0 };
    }
  }

  // Step 5: Build context with trilingual system prompt
  const chunksText = finalResults.map((r, i) => `[${i + 1}] ${r.text.slice(0, 800)}`).join('\n\n---\n\n');
  const langLabel = LANG_LABELS[userLang];

  const systemPrompt = `你係專業嘅 AI 客服助手。根據以下知識庫內容回答用戶問題。
⚠️ 重要：必須用${langLabel}回答。如果用戶用廣東話問，就用廣東話答；用英文問就用英文答；用簡體就用簡體。

知識庫內容：
${chunksText}

規則：
- 只根據知識庫內容回答，唔好自己創作
- 如果資料不足，誠實講「知識庫未有相關資訊」
- 回答要簡潔清晰，適合 WhatsApp 閱讀
- 嚴格使用${langLabel}，不要混雜其他語言
- 如果有餐牌/價錢資料，原文照錄，不要擅自改動`;

  // 選模型：粵語用 Qwen（更地道），其他用 DeepSeek
  const chatProvider: ModelProvider = userLang === 'yue' ? 'qwen' : 'deepseek';
  
  let reply: string;
  try {
    reply = await chatCompletion(
      [
        { role: 'system', content: systemPrompt },
        ...history.slice(-6),
        { role: 'user', content: query },
      ],
      env,
      { maxTokens, temperature: 0.3, provider: chatProvider },
    );
  } catch (e) {
    console.error('[rag] chat error:', e);
    reply = finalResults[0]?.text?.slice(0, 500) || '暫時未能回應';
  }

  // Add sources
  const sourceNames = [...new Set(finalResults.map(r => r.title).filter(Boolean))];
  if (sourceNames.length) {
    const srcLabel = userLang === 'en' ? 'Sources' : '來源';
    reply += `\n\n📚 ${srcLabel}：${sourceNames.slice(0, 3).join('、')}`;
  }

  return {
    reply: reply.slice(0, 4000),
    sources: finalResults.map(r => ({ text: r.text.slice(0, 200), score: r.score })),
    confidence: finalResults[0]?.score || 0,
  };
}

// ═══════════════════════════════════════════════════════
// 8. Document ingestion + image processing (保持)
// ═══════════════════════════════════════════════════════

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

  if ((env as any).VECTORIZE) {
    for (let i = 0; i < vectors.length; i += 100) {
      await (env as any).VECTORIZE.upsert(vectors.slice(i, i + 100));
    }
  }
  return vectors.length;
}

export async function getKBStats(_env: Env, namespace: string): Promise<{ namespace: string; vectorCount: number }> {
  return { namespace, vectorCount: -1 };
}

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

export async function ingestImage(
  env: Env, namespace: string, title: string, imageBase64: string, mimeType?: string,
): Promise<{ description: string; chunks: number }> {
  const description = await visionDescribe(env, imageBase64, mimeType);
  if (!description) return { description: '', chunks: 0 };
  const chunks = await ingestDocument(env, namespace, title, title, description);
  return { description, chunks };
}
