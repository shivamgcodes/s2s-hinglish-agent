// Serverless warm-up panel (DESIGN.md 5.4 item 2): polls GET /api/session/{sid} every 1.5 s while the Space wakes the
// RunPod GPU endpoint and claims a worker. Only on "ready" does the caller mount <Conversation> (which opens the
// websocket). Cancel sends DELETE /api/session/{sid} (releases the claim / cancels the queued job).
import { FC, useEffect, useRef, useState } from "react";
import { deleteSession, getSession } from "./api";
import { Card, Disclaimer } from "./Panels";
import { Dep1Session, END_REASON_TEXT, SpaceSession } from "./types";

const POLL_MS = 1500;

type Props = {
  sid: string;
  dep1: Dep1Session | null;
  onReady: () => void;
  onBack: () => void;
};

export const Warmup: FC<Props> = ({ sid, dep1, onReady, onBack }) => {
  const [s, setS] = useState<SpaceSession | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [t0] = useState(() => Date.now());
  const [now, setNow] = useState(Date.now());
  const done = useRef(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const x = await getSession(sid);
        if (!alive) return;
        setS(x);
        setErr(null);
        if (x.state === "ready" && !done.current) {
          done.current = true;
          onReady();
        }
      } catch (e: any) {
        if (alive) setErr(String(e?.message ?? e));
      }
    };
    tick();
    const id = setInterval(() => { if (!done.current) tick(); }, POLL_MS);
    const clock = setInterval(() => setNow(Date.now()), 500);
    return () => { alive = false; clearInterval(id); clearInterval(clock); };
  }, [sid, onReady]);

  const cancel = () => {
    done.current = true;
    deleteSession(sid);
    onBack();
  };

  const elapsed = Math.round(s?.elapsed_s ?? (now - t0) / 1000);
  const state = s?.state ?? "waking";
  const final = state === "failed" || state === "ended";
  const head = state === "busy" ? "All GPU workers are in a call: waiting for one to free up…"
    : state === "failed" ? "Could not start the GPU worker"
    : state === "ended" ? "This session ended before the call started"
    : "Warming up the GPU…";

  return (
    <div className="min-h-screen w-full bg-neutral-50 p-4">
      <div className="max-w-2xl mx-auto flex flex-col gap-4">
        <Card title="Starting your call" testId="warmup-panel"
          right={<span className="text-xs text-gray-500 font-mono">{dep1 ? `${dep1.recordId} · ${dep1.pairing}` : "stock session"}</span>}>
          <div className="flex items-center gap-3">
            {!final && <span className="inline-block h-4 w-4 rounded-full bg-orange-300 animate-pulse" />}
            <p data-testid="warmup-state" data-state={state} className={`text-base ${final ? "text-red-700 font-semibold" : "text-gray-800"}`}>
              {head} {!final && <span className="tabular-nums">{elapsed} s</span>}
            </p>
          </div>
          <p className="text-xs text-gray-500 mt-1" data-testid="warmup-detail">
            {state === "ended" && s?.end_reason ? (END_REASON_TEXT[s.end_reason] ?? s.end_reason) + (s?.detail ? ` (${s.detail})` : "") : (s?.detail ?? "starting")}
          </p>
          {!final && (
            <p className="text-xs text-gray-500 mt-2">
              The GPU scales to zero when nobody is talking. The first call after a quiet period loads the 7B model:
              about 1-2 min, up to about 8 min on a fresh GPU host. Keep this tab open; the call starts by itself.
            </p>
          )}
          {err && <p className="text-xs text-red-600 mt-1">/api/session: {err}</p>}
          <div className="mt-3 flex gap-2">
            <button data-testid="btn-cancel" onClick={cancel}
              className={`py-1 px-3 rounded border text-sm ${final ? "bg-[#76b900] text-white border-[#5a8d00]" : "bg-white"}`}>
              {final ? "Back" : "Cancel"}
            </button>
          </div>
        </Card>
        <Disclaimer />
      </div>
    </div>
  );
};
