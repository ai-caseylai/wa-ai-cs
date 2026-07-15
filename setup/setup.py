#!/usr/bin/env python3
"""Cloudflare 資源設定 — cloudflare-python SDK"""
import os, sys
from cloudflare import Cloudflare

API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
if not API_TOKEN:
    print("❌ CLOUDFLARE_API_TOKEN 未設定"); sys.exit(1)

client = Cloudflare(api_token=API_TOKEN)

# Get account ID
accounts = list(client.accounts.list())
if not accounts:
    print("❌ 找不到帳戶"); sys.exit(1)
account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", accounts[0].id)
print(f"📋 Account: {account_id}")

# 1. Vectorize
INDEX_NAME = "wa-ai-cs-kb"
print(f"\n🔍 Vectorize: {INDEX_NAME}")
try:
    indexes = list(client.vectorize.indexes.list(account_id=account_id))
    existing = [i for i in indexes if i.name == INDEX_NAME]
    if existing:
        print(f"   ✅ 已存在: {existing[0].name}")
    else:
        idx = client.vectorize.indexes.create(account_id=account_id, name=INDEX_NAME,
            description="WhatsApp AI 客服知識庫", preset="dasbhoomni-text-embedding-v4", metric="cosine")
        print(f"   ✅ 已建立: {idx.name}")
except Exception as e:
    print(f"   ⚠️  {e}")

# 2. D1
DB_NAME = "wa-ai-cs-db"
print(f"\n🗄️  D1: {DB_NAME}")
db_id = None
try:
    dbs = list(client.d1.databases.list(account_id=account_id))
    existing = [d for d in dbs if d.name == DB_NAME]
    if existing:
        db_id = existing[0].uuid
        print(f"   ✅ 已存在: {existing[0].name} ({db_id})")
    else:
        db = client.d1.databases.create(account_id=account_id, name=DB_NAME)
        db_id = db.uuid
        print(f"   ✅ 已建立: {db.name} ({db_id})")
except Exception as e:
    print(f"   ⚠️  {e}")

# 3. R2
BUCKET_NAME = "wa-ai-cs-media"
print(f"\n📦 R2: {BUCKET_NAME}")
try:
    buckets = list(client.r2.buckets.list(account_id=account_id))
    existing = [b for b in buckets if b.name == BUCKET_NAME]
    if existing:
        print(f"   ✅ 已存在: {existing[0].name}")
    else:
        client.r2.buckets.create(account_id=account_id, name=BUCKET_NAME)
        print(f"   ✅ 已建立: {BUCKET_NAME}")
except Exception as e:
    print(f"   ⚠️  {e}")

print(f"""
╔══════════════════════════════════════╗
║  ✅ Cloudflare 資源設定完成           ║
╠══════════════════════════════════════╣
║  Vectorize: {INDEX_NAME:<25} ║
║  D1:        {DB_NAME:<25} ║
║  R2:        {BUCKET_NAME:<25} ║
╠══════════════════════════════════════╣
║  下一步: npx wrangler deploy         ║
╚══════════════════════════════════════╝
""")
if db_id:
    print(f"📝 D1 database_id = \"{db_id}\"")
