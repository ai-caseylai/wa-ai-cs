#!/usr/bin/env python3
"""
Cloudflare 資源設定腳本
使用 cloudflare-python SDK 建立 Vectorize / D1 / R2

環境變數：
  CLOUDFLARE_API_TOKEN  — API token
  CLOUDFLARE_ACCOUNT_ID — Account ID（可選，自動檢測）
"""

import os
import sys
import time
from cloudflare import Cloudflare

API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
if not API_TOKEN:
    print("❌ 請設定 CLOUDFLARE_API_TOKEN")
    sys.exit(1)

client = Cloudflare(api_token=API_TOKEN)

# Get account ID
accounts = client.accounts.list()
if not accounts:
    print("❌ 找不到帳戶")
    sys.exit(1)

account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", accounts[0].id)
print(f"📋 Account: {account_id}")

# ═══════════════════════════════════════════════════════
# 1. Vectorize Index
# ═══════════════════════════════════════════════════════
INDEX_NAME = "wa-ai-cs-kb"
print(f"\n🔍 建立 Vectorize Index: {INDEX_NAME}")

try:
    # Check if exists
    indexes = client.vectorize.indexes.list(account_id=account_id)
    existing = [i for i in indexes if i.name == INDEX_NAME]
    if existing:
        print(f"   ✅ 已存在: {existing[0].name}")
    else:
        index = client.vectorize.indexes.create(
            account_id=account_id,
            name=INDEX_NAME,
            description="WhatsApp AI 客服知識庫",
            preset="dasbhoomni-text-embedding-v4",  # or custom dimensions
            metric="cosine",
        )
        print(f"   ✅ 已建立: {index.name}")
except Exception as e:
    print(f"   ⚠️  Vectorize: {e}")

# ═══════════════════════════════════════════════════════
# 2. D1 Database
# ═══════════════════════════════════════════════════════
DB_NAME = "wa-ai-cs-db"
print(f"\n🗄️  建立 D1 Database: {DB_NAME}")

db_id = None
try:
    dbs = client.d1.databases.list(account_id=account_id)
    existing = [d for d in dbs if d.name == DB_NAME]
    if existing:
        db_id = existing[0].uuid
        print(f"   ✅ 已存在: {existing[0].name} ({db_id})")
    else:
        db = client.d1.databases.create(account_id=account_id, name=DB_NAME)
        db_id = db.uuid
        print(f"   ✅ 已建立: {db.name} ({db_id})")
except Exception as e:
    print(f"   ⚠️  D1: {e}")

# ═══════════════════════════════════════════════════════
# 3. R2 Bucket
# ═══════════════════════════════════════════════════════
BUCKET_NAME = "wa-ai-cs-media"
print(f"\n📦 建立 R2 Bucket: {BUCKET_NAME}")

try:
    buckets = client.r2.buckets.list(account_id=account_id)
    existing = [b for b in buckets if b.name == BUCKET_NAME]
    if existing:
        print(f"   ✅ 已存在: {existing[0].name}")
    else:
        client.r2.buckets.create(account_id=account_id, name=BUCKET_NAME)
        print(f"   ✅ 已建立: {BUCKET_NAME}")
except Exception as e:
    print(f"   ⚠️  R2: {e}")

# ═══════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════
print(f"""
╔══════════════════════════════════════════════╗
║  Cloudflare 資源設定完成                      ║
╠══════════════════════════════════════════════╣
║  Vectorize:  {INDEX_NAME:<30} ║
║  D1:         {DB_NAME:<30} ║
║  R2:         {BUCKET_NAME:<30} ║
╠══════════════════════════════════════════════╣
║  下一步：                                     ║
║  1. 更新 wrangler.jsonc 中的 database_id     ║
║  2. npx wrangler deploy                      ║
║  3. 設定 secrets:                            ║
║     npx wrangler secret put ADMIN_PASSWORD   ║
║     npx wrangler secret put DEEPSEEK_API_KEY ║
║     npx wrangler secret put DASHSCOPE_API_KEY║
╚══════════════════════════════════════════════╝
""")

if db_id:
    print(f"📝 D1 database_id: {db_id}")
    print(f"   請更新 wrangler.jsonc: \"database_id\": \"{db_id}\"")
