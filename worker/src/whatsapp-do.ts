/**
 * WhatsAppDO — Baileys-based WhatsApp connection via Durable Object.
 * 
 * Persistent WebSocket to WhatsApp servers.
 * RPC methods: sendText, getStatus, getQR
 */
import { DurableObject } from 'cloudflare:workers';
import makeWASocket, {
  DisconnectReason,
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  makeInMemoryStore,
  type WASocket,
  type ConnectionState,
} from '@whiskeysockets/baileys';
import type { WhatsAppDO as IWhatsAppDO } from './types';
import P from 'pino';

export class WhatsAppDO extends DurableObject implements IWhatsAppDO {
  private sock: WASocket | null = null;
  private connected = false;
  private qrCode: string | null = null;
  private currentJid: string | null = null;
  private pendingConnections: Array<{ resolve: (v: boolean) => void }> = [];

  constructor(ctx: DurableObjectState, env: any) {
    super(ctx, env);
  }

  /**
   * Start WhatsApp connection.
   */
  async connect(): Promise<void> {
    if (this.sock) return;

    const { version } = await fetchLatestBaileysVersion();
    const logger = P({ level: 'silent' });

    // Use DO storage for auth state
    const { state, saveCreds } = await this._getAuthState();

    this.sock = makeWASocket({
      version,
      logger,
      printQRInTerminal: false,
      auth: state,
      browser: ['WA AI CS', 'Chrome', '1.0'],
    });

    this.sock.ev.on('creds.update', saveCreds);

    this.sock.ev.on('connection.update', (update: Partial<ConnectionState>) => {
      const { connection, qr, lastDisconnect } = update;

      if (qr) {
        this.qrCode = qr;
      }

      if (connection === 'open') {
        this.connected = true;
        this.qrCode = null;
        this.currentJid = this.sock!.user?.id || null;
        // Resolve pending connection promises
        for (const p of this.pendingConnections) p.resolve(true);
        this.pendingConnections = [];
      }

      if (connection === 'close') {
        this.connected = false;
        const reason = (lastDisconnect?.error as any)?.output?.statusCode;
        const shouldReconnect = reason !== DisconnectReason.loggedOut;

        if (shouldReconnect) {
          setTimeout(() => this.reconnect(), 5000);
        } else {
          this.sock = null;
          this.qrCode = null;
        }
      }
    });

    // Handle incoming messages
    this.sock.ev.on('messages.upsert', async (m) => {
      for (const msg of m.messages) {
        if (!msg.key.fromMe && msg.message) {
          const text = msg.message.conversation || 
                       msg.message.extendedTextMessage?.text || '';
          const sender = msg.key.remoteJid || '';
          
          if (text && sender) {
            // Forward to ConversationDO for AI processing
            await this._handleIncoming(sender, text);
          }
        }
      }
    });
  }

  /**
   * Send text message via WhatsApp.
   */
  async sendText(phone: string, text: string): Promise<{ success: boolean; id?: string }> {
    await this.ensureConnection();
    if (!this.sock || !this.connected) {
      return { success: false };
    }

    const jid = phone.includes('@') ? phone : `${phone}@s.whatsapp.net`;
    try {
      const result = await this.sock.sendMessage(jid, { text: text.slice(0, 4000) });
      return { success: true, id: result?.key?.id };
    } catch (e) {
      console.error('[wa] send error:', e);
      return { success: false };
    }
  }

  /**
   * Get WhatsApp connection status.
   */
  async getStatus(): Promise<{ connected: boolean; jid?: string }> {
    return {
      connected: this.connected,
      jid: this.currentJid || undefined,
    };
  }

  /**
   * Get QR code for pairing (base64 image or string).
   */
  async getQR(): Promise<string | null> {
    return this.qrCode;
  }

  // ── Internal ──

  private async ensureConnection(): Promise<void> {
    if (!this.sock) {
      await this.connect();
    }
    if (!this.connected) {
      // Wait for connection (max 30s)
      await new Promise<boolean>((resolve) => {
        this.pendingConnections.push({ resolve });
        setTimeout(() => resolve(false), 30000);
      });
    }
  }

  private async reconnect(): Promise<void> {
    this.sock = null;
    await this.connect();
  }

  private async _handleIncoming(sender: string, text: string): Promise<void> {
    try {
      // Get ConversationDO for this sender
      const convId = (this.env as any).CONVERSATION_DO.idFromName(sender);
      const conv = (this.env as any).CONVERSATION_DO.get(convId);
      const result = await conv.chat(text, this.env);

      // Send reply
      if (this.connected && this.sock) {
        await this.sock.sendMessage(sender, { text: result.reply.slice(0, 4000) });
      }
    } catch (e) {
      console.error('[wa] incoming handle error:', e);
    }
  }

  private async _getAuthState(): Promise<{ state: any; saveCreds: () => Promise<void> }> {
    // Use DO storage for auth
    let creds = await this.ctx.storage.get('auth_creds') as any || {};
    let keys = await this.ctx.storage.get('auth_keys') as any || {};

    const state = {
      creds,
      keys: {
        get: async (type: string, ids: string[]) => {
          const result: Record<string, any> = {};
          for (const id of ids) {
            const key = `${type}|${id}`;
            if (keys[key]) result[id] = keys[key];
          }
          return result;
        },
        set: async (data: any) => {
          for (const [key, value] of Object.entries(data)) {
            keys[key] = value;
          }
          await this.ctx.storage.put('auth_keys', keys);
        },
      },
    };

    const saveCreds = async () => {
      await this.ctx.storage.put('auth_creds', creds);
    };

    return { state, saveCreds };
  }
}
