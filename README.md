# 🤖 WhatsApp AI 客服 — Cloudflare Edition

**WhatsApp AI 客服系統，全架構運行在 Cloudflare 上。**

## 🏗️ 架構

```
WhatsApp → Baileys DO → ConversationDO → Vectorize RAG → DeepSeek → Reply
                                │
                          Onboarding Web UI
                          (Hono + Static HTML)
```

## 📁 專案結構

```
wa-ai-cs/
├── worker/                 # Cloudflare Worker (TypeScript)
│   ├── src/
│   │   ├── index.ts        # Hono 路由 + Onboarding UI
│   │   ├── whatsapp-do.ts  # Baileys WhatsApp DO
│   │   ├── conversation-do.ts  # 對話狀態 DO
│   │   ├── rag.ts          # RAG 引擎 (Vectorize + DeepSeek)
│   │   └── types.ts        # 型別定義
│   ├── wrangler.jsonc      # Cloudflare 配置
│   ├── package.json
│   └── tsconfig.json
├── setup/                  # Python 資源設定
│   ├── setup.py            # cloudflare-python SDK
│   └── requirements.txt
└── README.md
```

## 🚀 部署步驟

### 1. 設定 Cloudflare 資源

```bash
cd setup
pip install -r requirements.txt
export CLOUDFLARE_API_TOKEN="your-token"
python setup.py
```

### 2. 更新 wrangler.jsonc

將 `setup.py` 輸出的 `database_id` 填入 `wrangler.jsonc`。

### 3. 設定 Secrets

```bash
cd worker
npx wrangler secret put ADMIN_PASSWORD
npx wrangler secret put DEEPSEEK_API_KEY
npx wrangler secret put DASHSCOPE_API_KEY
```

### 4. 部署

```bash
cd worker
npm install
npx wrangler deploy
```

## 🔧 API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | Onboarding Web UI |
| POST | `/api/setup` | 設定新客戶 `{phone}` |
| POST | `/api/chat` | AI 對話 `{phone, message}` |
| POST | `/api/ingest` | 上傳文檔到知識庫 |
| GET | `/api/stats/:ns` | 知識庫統計 |
| POST | `/api/wa/send` | 發送 WhatsApp 訊息 |
| GET | `/api/wa/status` | WhatsApp 連線狀態 |
| GET | `/api/wa/qr` | QR Code 配對 |

## 🛠 技術棧

- **Runtime**: Cloudflare Workers
- **WhatsApp**: Baileys (Durable Object)
- **RAG**: Cloudflare Vectorize + DeepSeek Chat
- **Embedding**: DashScope text-embedding-v4
- **框架**: Hono
- **資源管理**: cloudflare-python SDK
