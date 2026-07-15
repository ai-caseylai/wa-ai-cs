/**
 * WhatsApp AI 客服 — Cloudflare Workers
 * 
 * Architecture:
 *   WhatsApp → Baileys DO → Webhook → ConversationDO → RAG (Vectorize + DeepSeek) → Reply
 * 
 * Routes:
 *   GET  /                    → Onboarding web UI
 *   POST /api/chat            → Chat with AI
 *   POST /api/ingest          → Upload document to knowledge base
 *   POST /api/setup           → Setup new customer
 *   GET  /api/stats/:ns       → Knowledge base stats
 *   POST /api/wa/send         → Send WhatsApp message
 *   GET  /api/wa/status       → WhatsApp connection status
 *   GET  /api/wa/qr           → Get QR code for pairing
 */

import { Hono } from 'hono';
import type { Env } from './types';
import { ConversationDO } from './conversation-do';
import { ingestDocument, getKBStats } from './rag';

export { ConversationDO };

const app = new Hono<{ Bindings: Env }>();

// ═══════════════════════════════════════════════════════
// Onboarding Web UI
// ═══════════════════════════════════════════════════════
app.get('/', (c) => c.html(ONBOARD_HTML));

// ═══════════════════════════════════════════════════════
// Health
// ═══════════════════════════════════════════════════════
app.get('/api/health', (c) => c.json({
  status: 'ok',
  service: 'wa-ai-cs',
  timestamp: new Date().toISOString(),
}));

// ═══════════════════════════════════════════════════════
// Setup: create session + send welcome WhatsApp
// ═══════════════════════════════════════════════════════
app.post('/api/setup', async (c) => {
  const { phone } = await c.req.json<{ phone: string }>();
  if (!phone || !/^\d{7,15}$/.test(phone)) {
    return c.json({ error: '請輸入正確電話號碼' }, 400);
  }

  // Send welcome message via WhatsApp DO
  try {
    const waId = c.env.WHATSAPP_DO.idFromName('main');
    const wa = c.env.WHATSAPP_DO.get(waId);
    await wa.sendText(phone, '👋 你好！你嘅 AI 客服已經準備好。上傳產品資料後，AI 就會自動回答客人問題！');
  } catch (e) {
    return c.json({ error: `WhatsApp 未連接：${(e as Error).message}` }, 503);
  }

  // Create conversation with KB namespace
  const convId = c.env.CONVERSATION_DO.idFromName(phone);
  const conv = c.env.CONVERSATION_DO.get(convId);
  await conv.setNamespace(`kb_${phone}`);

  return c.json({ phone, kb: `kb_${phone}`, ok: true });
});

// ═══════════════════════════════════════════════════════
// Chat: query AI via RAG
// ═══════════════════════════════════════════════════════
app.post('/api/chat', async (c) => {
  const { phone, message } = await c.req.json<{ phone: string; message: string }>();
  if (!message?.trim()) return c.json({ error: '請輸入訊息' }, 400);

  const userKey = phone || 'anonymous';
  const convId = c.env.CONVERSATION_DO.idFromName(userKey);
  const conv = c.env.CONVERSATION_DO.get(convId);
  const result = await conv.chat(message, c.env);

  return c.json(result);
});

// ═══════════════════════════════════════════════════════
// Ingest: upload document to Vectorize
// ═══════════════════════════════════════════════════════
app.post('/api/ingest', async (c) => {
  const auth = c.req.header('Authorization');
  if (auth !== `Bearer ${c.env.ADMIN_PASSWORD}`) {
    return c.json({ error: 'unauthorized' }, 401);
  }

  let body: { namespace: string; title: string; source?: string; content: string };
  try {
    body = await c.req.json();
  } catch {
    // Try multipart form
    const form = await c.req.formData();
    const file = form.get('file') as File | null;
    const namespace = form.get('namespace')?.toString() || 'default';
    const title = form.get('title')?.toString() || 'untitled';

    if (!file) return c.json({ error: 'no file or JSON body' }, 400);

    let content: string;
    if (file.type === 'application/pdf') {
      // For PDF, store as raw text (basic)
      content = await file.text();
    } else {
      content = await file.text();
    }

    const count = await ingestDocument(c.env, namespace, title, title, content);
    return c.json({ ok: true, namespace, title, chunks: count });
  }

  const { namespace = 'default', title, source = title || 'manual', content } = body;
  if (!content) return c.json({ error: 'content required' }, 400);

  const count = await ingestDocument(c.env, namespace, title, source, content);
  return c.json({ ok: true, namespace, title, chunks: count });
});

// ═══════════════════════════════════════════════════════
// Stats: get KB info
// ═══════════════════════════════════════════════════════
app.get('/api/stats/:namespace', async (c) => {
  const namespace = c.req.param('namespace');
  const stats = await getKBStats(c.env, namespace);
  return c.json(stats);
});

// ═══════════════════════════════════════════════════════
// WhatsApp: send message
// ═══════════════════════════════════════════════════════
app.post('/api/wa/send', async (c) => {
  const { phone, text } = await c.req.json<{ phone: string; text: string }>();
  if (!phone || !text) return c.json({ error: 'phone and text required' }, 400);

  const waId = c.env.WHATSAPP_DO.idFromName('main');
  const wa = c.env.WHATSAPP_DO.get(waId);
  const result = await wa.sendText(phone, text);

  return c.json(result);
});

// ═══════════════════════════════════════════════════════
// WhatsApp: status
// ═══════════════════════════════════════════════════════
app.get('/api/wa/status', async (c) => {
  const waId = c.env.WHATSAPP_DO.idFromName('main');
  const wa = c.env.WHATSAPP_DO.get(waId);
  const status = await wa.getStatus();
  return c.json(status);
});

// ═══════════════════════════════════════════════════════
// WhatsApp: QR code for pairing
// ═══════════════════════════════════════════════════════
app.get('/api/wa/qr', async (c) => {
  const waId = c.env.WHATSAPP_DO.idFromName('main');
  const wa = c.env.WHATSAPP_DO.get(waId);
  const qr = await wa.getQR();
  if (!qr) return c.json({ error: 'QR not available' }, 404);
  return c.html(`<html><body style="display:flex;justify-content:center;align-items:center;min-height:100vh;background:#111"><img src="${qr}" style="max-width:300px"></body></html>`);
});

export default app;

// ═══════════════════════════════════════════════════════
// Onboarding HTML (Carrd-style, single page)
// ═══════════════════════════════════════════════════════
const ONBOARD_HTML = `<!DOCTYPE html>
<html lang="zh-HK">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>WhatsApp AI 客服 — Cloudflare</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Helvetica Neue',sans-serif;background:#0f0f0f;color:#eee;min-height:100vh;display:flex;align-items:center;justify-content:center}
.cont{max-width:460px;width:90%;margin:20px auto;display:flex;flex-direction:column;align-items:center;gap:12px}
h1{font-size:1.6rem;font-weight:700;background:linear-gradient(135deg,#f6821f,#f4a460);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:#888;font-size:.85rem;margin-bottom:4px}
.bar{width:100%;background:#1a1a1a;border-radius:14px;padding:10px 14px;border:1px solid #2a2a2a;display:flex;gap:8px}
.bar input{flex:1;padding:10px 12px;border-radius:8px;border:1px solid #333;background:#0f0f0f;color:#eee;font-size:.85rem;outline:none}
.bar input:focus{border-color:#f6821f}
.btn{padding:10px 16px;border-radius:8px;border:none;font-weight:600;cursor:pointer;font-size:.85rem}
.btn-go{background:#f6821f;color:#fff}
.btn-go:hover{opacity:.9}
.btn-go:disabled{opacity:.4;cursor:not-allowed}
.btn-out{background:transparent;border:1px solid #333;color:#eee}
.status{font-size:.8rem;padding:8px 12px;border-radius:8px;width:100%}
.status.ok{background:#0a2a0a;color:#4f4}
.status.err{background:#2a0a0a;color:#f44}
.status.wait{background:#2a2a0a;color:#f0c040}
.upload{width:100%;display:flex;gap:8px;align-items:center}
.upload-zone{flex:1;border:2px dashed #333;border-radius:10px;padding:10px;text-align:center;cursor:pointer;font-size:.8rem;color:#888}
.upload-zone:hover{border-color:#f6821f}
.phone-frame{background:#111;border-radius:20px;padding:10px;border:1px solid #333;width:100%}
.phone-hd{background:#075e54;color:#fff;padding:10px 14px;border-radius:10px 10px 0 0;display:flex;align-items:center;gap:8px}
.phone-hd .av{width:34px;height:34px;border-radius:50%;background:#25D366;display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.phone-bd{background:#e5ddd5;min-height:360px;display:flex;flex-direction:column}
.chat-msgs{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:6px;max-height:50vh;min-height:320px}
.msg{max-width:85%;padding:8px 12px;border-radius:8px;font-size:.85rem;line-height:1.4;word-break:break-word}
.msg.ai{background:#fff;align-self:flex-start;box-shadow:0 1px 2px rgba(0,0,0,.1)}
.msg.user{background:#dcf8c6;align-self:flex-end;margin-left:auto}
.msg .tm{font-size:.6rem;color:#999;margin-top:2px;text-align:right}
.chat-in{display:flex;padding:6px;gap:6px;background:#f0f0f0;border-radius:0 0 10px 10px}
.chat-in input{flex:1;padding:8px 12px;border-radius:16px;border:1px solid #ddd;outline:none;font-size:.85rem}
.chat-in button{width:34px;height:34px;border-radius:50%;background:#25D366;border:none;color:#fff;font-size:1rem;cursor:pointer;flex-shrink:0}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#999;animation:blink 1.4s infinite;margin:0 2px}
.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,100%{opacity:.2}50%{opacity:1}}
.hidden{display:none!important}
.prog{width:100%;height:4px;background:#2a2a2a;border-radius:2px;overflow:hidden}
.prog-fill{height:100%;background:#f6821f;width:0%;transition:width .3s}
</style>
</head>
<body>
<div class="cont">
  <h1>🤖 WhatsApp AI 客服</h1>
  <p class="sub">Cloudflare Workers · Vectorize · DeepSeek</p>

  <div class="bar">
    <input type="text" id="phone" placeholder="WhatsApp 號碼" value="">
    <button class="btn btn-go" id="btnSetup" onclick="doSetup()">📱 設定</button>
  </div>
  <div class="status hidden" id="setupStatus"></div>

  <div id="uploadSection" class="hidden upload">
    <div class="upload-zone" onclick="document.getElementById('fileInput').click()">📄 上載 PDF</div>
    <input type="file" id="fileInput" accept=".pdf" class="hidden" onchange="handleFile(this.files)">
    <button class="btn btn-go" id="btnIngest" onclick="doIngest()" disabled>📚 加入</button>
  </div>
  <div class="prog hidden" id="prog"><div class="prog-fill" id="progFill"></div></div>

  <div class="phone-frame">
    <div class="phone-hd">
      <div class="av">🤖</div>
      <div><div style="font-weight:600;font-size:.9rem">AI 客服</div><div style="font-size:.65rem;opacity:.8" id="simStatus">未設定</div></div>
    </div>
    <div class="phone-bd">
      <div class="chat-msgs" id="chatMsgs">
        <div class="msg ai">你好！我係 AI 客服助手 👋<br>輸入號碼開始設定，然後喺度測試</div>
      </div>
      <div class="chat-in">
        <input type="text" id="chatInput" placeholder="輸入訊息..." onkeydown="if(event.key==='Enter')sendMsg()">
        <button onclick="sendMsg()">➤</button>
      </div>
    </div>
  </div>
</div>

<script>
let phone = localStorage.getItem('wa_phone') || '';
let setupDone = localStorage.getItem('wa_setup') === 'true';
if (phone) { document.getElementById('phone').value = phone; }
if (setupDone) { showUpload(); document.getElementById('btnSetup').textContent='✅ 已設定'; document.getElementById('btnSetup').disabled=true; document.getElementById('simStatus').textContent='在線 — '+phone; }

function showUpload() {
  document.getElementById('uploadSection').classList.remove('hidden');
}
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function doSetup() {
  const p = document.getElementById('phone').value.trim();
  if (!p || !/^\\d{7,15}$/.test(p)) return alert('請輸入正確號碼');
  const s = document.getElementById('setupStatus');
  s.className='status wait'; s.classList.remove('hidden'); s.textContent='⏳ 設定中...';
  try {
    const r = await (await fetch('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({phone:p})})).json();
    if (r.error) { s.className='status err'; s.textContent='❌ '+r.error; return; }
    phone = p; localStorage.setItem('wa_phone',p); localStorage.setItem('wa_setup','true');
    s.className='status ok'; s.textContent='✅ 已連接！';
    document.getElementById('btnSetup').textContent='✅ 已設定'; document.getElementById('btnSetup').disabled=true;
    document.getElementById('simStatus').textContent='在線 — '+p;
    showUpload();
  } catch(e) { s.className='status err'; s.textContent='❌ '+e.message; }
}

let fileToUpload = null;
function handleFile(fl) {
  if (fl.length > 0) { fileToUpload = fl[0]; document.getElementById('btnIngest').disabled = false; }
}
async function doIngest() {
  if (!fileToUpload) return;
  const bar = document.getElementById('prog'); const fill = document.getElementById('progFill');
  bar.classList.remove('hidden'); fill.style.width='50%';
  const fd = new FormData(); fd.append('file',fileToUpload); fd.append('namespace','kb_'+phone); fd.append('title',fileToUpload.name);
  await fetch('/api/ingest',{method:'POST',headers:{Authorization:'Bearer changeme'},body:fd});
  fill.style.width='100%';
  document.getElementById('btnIngest').textContent='✅ 已加入'; document.getElementById('btnIngest').disabled=true;
  setTimeout(()=>bar.classList.add('hidden'),1500);
}

async function sendMsg() {
  const inp = document.getElementById('chatInput'); const msg = inp.value.trim();
  if (!msg) return;
  const area = document.getElementById('chatMsgs');
  const now = new Date().toLocaleTimeString('zh-HK',{hour:'2-digit',minute:'2-digit'});
  area.innerHTML += '<div class="msg user">'+esc(msg)+'<div class="tm">'+now+'</div></div>';
  inp.value=''; area.scrollTop=area.scrollHeight;
  const tid = 't'+Date.now();
  area.innerHTML += '<div class="msg ai" id="'+tid+'"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  area.scrollTop=area.scrollHeight;
  try {
    const r = await (await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({phone,message:msg})})).json();
    document.getElementById(tid)?.remove();
    area.innerHTML += '<div class="msg ai">'+esc(r.reply||r.error||'Error')+'<div class="tm">'+now+'</div></div>';
    area.scrollTop=area.scrollHeight;
  } catch(e) {
    document.getElementById(tid)?.remove();
    area.innerHTML += '<div class="msg ai" style="color:#f44">❌ '+esc(e.message)+'</div>';
  }
}
</script>
</body>
</html>`;
