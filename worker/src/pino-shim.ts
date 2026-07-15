/**
 * Minimal pino logger shim for Cloudflare Workers.
 * Aliased via wrangler.jsonc: "pino" → "./src/pino-shim.ts"
 */

type LogFn = (msg: string | Record<string, unknown>, ...args: unknown[]) => void;

class PinoShim {
  info: LogFn;
  error: LogFn;
  warn: LogFn;
  debug: LogFn;
  trace: LogFn;

  constructor() {
    const log = (level: string): LogFn => {
      return (msg: string | Record<string, unknown>, ...args: unknown[]) => {
        const entry =
          typeof msg === "string"
            ? { level, msg, ...(args.length ? { args } : {}) }
            : { level, ...msg };
        console.log(JSON.stringify(entry));
      };
    };

    this.info = log("info");
    this.error = log("error");
    this.warn = log("warn");
    this.debug = log("debug");
    this.trace = log("trace");
  }

  child(_bindings: Record<string, unknown>): PinoShim {
    // Return the same logger instance (simplified)
    return this;
  }
}

const logger = new PinoShim();

export default logger;
export { PinoShim };
