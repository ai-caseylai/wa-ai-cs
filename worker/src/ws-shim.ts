/**
 * WebSocket shim: Bridges Workers' native WebSocket to the `ws` API that Baileys expects.
 * Aliased via wrangler.jsonc: "ws" → "./src/ws-shim.ts"
 * Uses addEventListener because Workers WebSocket types don't expose on* setters.
 */
import { EventEmitter } from "node:events";

export default class WebSocketShim extends EventEmitter {
  static OPEN = 1;
  static CLOSED = 3;
  static CLOSING = 2;
  static CONNECTING = 0;

  readyState: number;
  private _ws: WebSocket | null = null;

  constructor(url: string, _opts?: Record<string, unknown>) {
    super();
    this.readyState = WebSocketShim.CONNECTING;
    this.setMaxListeners(0);

    try {
      this._ws = new WebSocket(url);
    } catch (err) {
      this.readyState = WebSocketShim.CLOSED;
      queueMicrotask(() => this.emit("error", err));
      return;
    }

    this._ws.addEventListener("open", () => {
      this.readyState = WebSocketShim.OPEN;
      this.emit("open");
    });

    this._ws.addEventListener("message", (event: Event) => {
      const me = event as MessageEvent;
      let data: Buffer | string = me.data as string;
      if ((data as any) instanceof ArrayBuffer) {
        data = Buffer.from(data);
      }
      this.emit("message", data);
    });

    this._ws.addEventListener("close", (event: Event) => {
      const ce = event as CloseEvent;
      this.readyState = WebSocketShim.CLOSED;
      this.emit("close", ce.code, ce.reason);
    });

    this._ws.addEventListener("error", (event: Event) => {
      this.emit("error", event);
    });
  }

  send(data: string | Buffer | ArrayBuffer, cb?: (err?: Error) => void): void {
    try {
      this._ws?.send(data as string);
      cb?.();
    } catch (err) {
      cb?.(err as Error);
    }
  }

  close(code?: number, reason?: string): void {
    if (this.readyState !== WebSocketShim.CLOSED) {
      this.readyState = WebSocketShim.CLOSING;
      this._ws?.close(code, reason);
    }
  }

  terminate(): void {
    this._ws?.close(1006, "Terminated");
    this.readyState = WebSocketShim.CLOSED;
  }

  setMaxListeners(_n: number): this {
    return this;
  }
}
