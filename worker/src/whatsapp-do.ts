/**
 * WhatsAppDO — Lightweight WuzAPI bridge.
 * Connects to existing WuzAPI on Tencent Cloud for WhatsApp messaging.
 * No heavy Baileys dependencies.
 */
import { DurableObject } from 'cloudflare:workers';

const WUZAPI_URL = 'http://43.156.105.201:8081';
const WUZAPI_TOKEN = 'casey-wa-token-2026';

export class WhatsAppDO extends DurableObject {
  async sendText(phone: string, text: string): Promise<{ success: boolean; id?: string }> {
    try {
      const resp = await fetch(`${WUZAPI_URL}/chat/send/text`, {
        method: 'POST',
        headers: { 'Token': WUZAPI_TOKEN, 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone, body: text.slice(0, 4000) }),
      });
      const data = await resp.json() as any;
      return { success: data.success, id: data.data?.Id };
    } catch (e) {
      console.error('[wa] send error:', e);
      return { success: false };
    }
  }

  async getStatus(): Promise<{ connected: boolean; jid?: string }> {
    return { connected: true, jid: 'casey-test' };
  }

  async getQR(): Promise<string | null> {
    return null; // WuzAPI handles QR separately
  }
}
