/**
 * ConversationDO — Durable Object for per-user conversation state.
 * Stores chat history, active KB namespace, session management.
 */
import { DurableObject } from 'cloudflare:workers';
import type { ConversationState, ChatResponse } from './types';
import { searchAndAnswer } from './rag';

export class ConversationDO extends DurableObject {
  private state: ConversationState;

  constructor(ctx: DurableObjectState, env: any) {
    super(ctx, env);
    this.state = {
      messages: [],
      kbNamespace: 'default',
      lastActive: Date.now(),
    };
  }

  async initialize(): Promise<void> {
    const saved = await this.ctx.storage.get<ConversationState>('state');
    if (saved) this.state = saved;

    // Reset if inactive for 2 hours
    if (Date.now() - this.state.lastActive > 7200000) {
      this.state = { messages: [], kbNamespace: 'default', lastActive: Date.now() };
    }
  }

  /**
   * Process a chat message and return AI response.
   */
  async chat(text: string, env: any): Promise<ChatResponse> {
    await this.initialize();

    // Reset command
    if (text.trim().toLowerCase() === 'reset' || text.trim() === '從頭開始') {
      this.state.messages = [];
      this.state.lastActive = Date.now();
      await this.save();
      return { reply: '已重置。有咩幫到你？', sources: [], confidence: 1 };
    }

    // Detect /kb command to switch namespace
    const kbMatch = text.match(/^\/kb\s+(\S+)/);
    if (kbMatch) {
      this.state.kbNamespace = kbMatch[1];
      this.state.lastActive = Date.now();
      await this.save();
      return { reply: `✅ 已切換到知識庫：${kbMatch[1]}`, sources: [], confidence: 1 };
    }

    // RAG query
    const result = await searchAndAnswer(
      env,
      text,
      this.state.kbNamespace,
      this.state.messages,
    );

    // Update history
    this.state.messages.push({ role: 'user', content: text });
    this.state.messages.push({ role: 'assistant', content: result.reply });
    if (this.state.messages.length > 20) {
      this.state.messages = this.state.messages.slice(-20);
    }
    this.state.lastActive = Date.now();
    await this.save();

    return result;
  }

  /**
   * Set the knowledge base namespace for this conversation.
   */
  async setNamespace(namespace: string): Promise<void> {
    this.state.kbNamespace = namespace;
    this.state.lastActive = Date.now();
    await this.save();
  }

  private async save(): Promise<void> {
    await this.ctx.storage.put('state', this.state);
  }
}
