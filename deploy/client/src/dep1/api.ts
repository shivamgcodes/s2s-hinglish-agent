// Same-origin HTTP calls. DEP1 INTERFACE section 5 routes (records, samples) plus the serverless Space session API
// (DESIGN.md 5.1): POST/GET/DELETE /api/session, /api/config, /metrics?sid=. Everything is on the Space origin.
import { FullRecord, Metrics, RecordSummary, SampleItem, ScriptResponse, SpaceConfig, SpaceSession } from "./types";

const getJSON = async <T,>(path: string): Promise<T> => {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) {
    throw new Error(`${path}: HTTP ${r.status}`);
  }
  return (await r.json()) as T;
};

export const fetchRecords = () => getJSON<RecordSummary[]>("/api/records");
export const fetchRecord = (id: string) => getJSON<FullRecord>(`/api/records/${encodeURIComponent(id)}`);
export const fetchSamples = () => getJSON<SampleItem[]>("/api/samples");
// D-SCRIPT-PANEL: the expected conversation for a record + pairing (available=false when that call does not exist)
export const fetchScript = (id: string, pairing: string) =>
  getJSON<ScriptResponse>(`/api/script/${encodeURIComponent(id)}?pairing=${encodeURIComponent(pairing)}`);
// Last 0x07 metrics event the Space relay saw for this sid (the live panel reads the socket's own 0x07 events).
export const fetchMetrics = (sid: string) => getJSON<Metrics>(`/metrics?sid=${encodeURIComponent(sid)}`);
export const fetchConfig = () => getJSON<SpaceConfig>("/api/config");

export class SessionError extends Error {
  status: number;
  body: any;
  constructor(status: number, body: any) {
    super(sessionErrorText(status, body));
    this.status = status;
    this.body = body;
  }
}

const sessionErrorText = (status: number, body: any): string => {
  const e = body?.error;
  if (status === 403 && e === "passcode") return "Wrong or missing passcode.";
  if (status === 429 && e === "capacity") return `All demo slots are in use. Try again in about ${body?.retry_after_s ?? 30} s.`;
  if (status === 429 && e === "rate_limited") return `Too many calls from this network. Try again in ${Math.ceil((body?.retry_after_s ?? 60) / 60)} min.`;
  if (status === 400) return `Bad session config: ${body?.detail ?? e ?? "?"}`;
  if (status === 503) return `The demo is not configured: ${body?.detail ?? e ?? "?"}`;
  return `POST /api/session: HTTP ${status} ${e ?? ""}`;
};

export const createSession = async (body: { record_id: string | null; pairing: string; seed: number | null; passcode?: string }): Promise<SpaceSession> => {
  const r = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const js = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new SessionError(r.status, js);
  }
  return js as SpaceSession;
};

export const getSession = (sid: string) => getJSON<SpaceSession>(`/api/session/${encodeURIComponent(sid)}`);

// Releases the claimed GPU worker (LB) or cancels the queued job. keepalive: also works from pagehide.
export const deleteSession = (sid: string | null | undefined) => {
  if (!sid) return;
  fetch(`/api/session/${encodeURIComponent(sid)}`, { method: "DELETE", keepalive: true }).catch(() => undefined);
};
