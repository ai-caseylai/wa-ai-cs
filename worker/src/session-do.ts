/**
 * SessionDO — 驗證碼 + 對話存儲 + 文件管理
 * 
 * 每個 session 一個 DO instance，用 idFromName(code) 存取。
 * 存儲: code, phone, verified, chat history, uploaded files
 */
import { DurableObject } from 'cloudflare:workers';

export interface SessionState {
  code: string;
  phone: string;
  verified: boolean;
  createdAt: number;
  chatHistory: { role: 'user' | 'assistant'; content: string; time: string }[];
  files: { name: string; chunks: number; time: string }[];
}

export class SessionDO extends DurableObject {
  private state: SessionState;

  constructor(ctx: DurableObjectState, env: any) {
    super(ctx, env);
    this.state = {
      code: '',
      phone: '',
      verified: false,
      createdAt: Date.now(),
      chatHistory: [],
      files: [],
    };
  }

  async init(code: string, phone: string): Promise<SessionState> {
    this.state = {
      code,
      phone,
      verified: false,
      createdAt: Date.now(),
      chatHistory: [],
      files: [],
    };
    await this.save();
    return this.state;
  }

  async getState(): Promise<SessionState> {
    const saved = await this.ctx.storage.get<SessionState>('state');
    if (saved) this.state = saved;
    return this.state;
  }

  async verify(): Promise<SessionState> {
    await this.getState();
    this.state.verified = true;
    await this.save();
    return this.state;
  }

  async addChat(role: 'user' | 'assistant', content: string): Promise<void> {
    await this.getState();
    this.state.chatHistory.push({
      role,
      content: content.slice(0, 4000),
      time: new Date().toISOString(),
    });
    if (this.state.chatHistory.length > 200) {
      this.state.chatHistory = this.state.chatHistory.slice(-200);
    }
    await this.save();
  }

  async addFile(name: string, chunks: number): Promise<void> {
    await this.getState();
    this.state.files.push({
      name,
      chunks,
      time: new Date().toISOString(),
    });
    await this.save();
  }

  async getChatHistory(): Promise<SessionState['chatHistory']> {
    await this.getState();
    return this.state.chatHistory;
  }

  private async save(): Promise<void> {
    await this.ctx.storage.put('state', this.state);
  }
}
