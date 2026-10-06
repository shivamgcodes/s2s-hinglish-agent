// DEP1 display panels. Display only: no backend logic beyond session config (spec Track 2).
import { FC, useEffect, useMemo, useRef, useState } from "react";
import { fetchSamples } from "./api";
import { ACTION_STAGES, ActionEvent, ActionStage, Metrics, SampleItem, StatBlock, TextEvent } from "./types";
import { Dep1State, SessionEvent } from "./useDep1Events";

const FRAME_HZ = 12.5;
const STEP_TARGET_MS = 80;

export const fmt = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(d);
const fmtT = (s: number) => {
  const m = Math.floor(s / 60);
  const r = s - 60 * m;
  return `${m}:${r < 10 ? "0" : ""}${r.toFixed(1)}`;
};

export const Card: FC<{ title: string; right?: React.ReactNode; className?: string; children: React.ReactNode; testId?: string }> = ({
  title, right, className, children, testId,
}) => (
  <section data-testid={testId} className={`bg-white border border-gray-200 rounded-lg p-3 flex flex-col min-h-0 ${className ?? ""}`}>
    <div className="flex items-baseline justify-between mb-2">
      <h2 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">{title}</h2>
      {right}
    </div>
    {children}
  </section>
);

// ---------------------------------------------------------------------------------------------- disclaimer
export const Disclaimer: FC<{ className?: string }> = ({ className }) => (
  <p data-testid="disclaimer" className={`text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded px-3 py-2 ${className ?? ""}`}>
    Research demo. Hinglish (Hindi-English code-switched) agent: PersonaPlex 7B with a LoRA fine-tuned on synthetic
    call data only (V4 adapter, V4_A2 step 600), evaluated on a V4 test set of 30 held-out calls. The model tends to
    misbehave or get confused on compound calls (calls with multiple questions and commands). Records, customers and
    order IDs are synthetic. Actions run against stub tools on an in-memory copy of the record. Single session; about
    3.5 min of conversation per session (context limit; the session panel shows this session&apos;s limit).
  </p>
);

// ---------------------------------------------------------------------------------------------- session info
export const SessionInfo: FC<{ session: SessionEvent | null; previewPrompt?: string; fallback?: Partial<SessionEvent> }> = ({
  session, previewPrompt, fallback,
}) => {
  const s = session ?? fallback ?? null;
  const [open, setOpen] = useState(false);
  if (!s) {
    return <Card title="Session" testId="session-panel"><p className="text-xs text-gray-500">Stock session (no record): router idle.</p></Card>;
  }
  const prompt = s.role_prompt ?? previewPrompt ?? "";
  return (
    <Card title="Session" testId="session-panel"
      right={<span className={`text-xs ${session ? "text-green-700" : "text-gray-400"}`}>{session ? "confirmed by server" : "requested"}</span>}>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-gray-500">record</dt><dd className="font-mono">{s.record_id ?? "–"}</dd>
        <dt className="text-gray-500">agent type</dt><dd>{s.agent_type ?? "–"}</dd>
        <dt className="text-gray-500">pairing / voice</dt><dd>{s.pairing ?? "–"} · {s.agent_gender === "m" ? "male" : s.agent_gender === "f" ? "female" : "–"} · {s.voice ?? "–"}</dd>
        <dt className="text-gray-500">seed</dt><dd>{s.seed ?? "–"}</dd>
        {session?.session_id && (<><dt className="text-gray-500">session id</dt><dd className="font-mono">{session.session_id}</dd></>)}
        {session?.context_frames_left !== undefined && (<><dt className="text-gray-500">context at start</dt>
          <dd>{session.context_frames_left} frames ({fmt(session.context_frames_left / FRAME_HZ, 0)} s)</dd></>)}
        <dt className="text-gray-500">router</dt><dd>{s.router_supported === false ? "not supported for this agent type" : "active"}</dd>
      </dl>
      {prompt && (
        <div className="mt-2">
          <button className="text-xs text-blue-700 underline" onClick={() => setOpen(o => !o)}>{open ? "hide" : "show"} role prompt</button>
          {open && <p data-testid="session-role-prompt" className="mt-1 text-xs text-gray-700 bg-gray-50 border rounded p-2 whitespace-pre-wrap">{prompt}</p>}
        </div>
      )}
    </Card>
  );
};

// ---------------------------------------------------------------------------------------------- text stream
type Seg = { start: number; t: number; toks: TextEvent[] };
const SENT_END = /[.?!।]\s*$/;

// tests/tcommon.segments rule (INTERFACE section 9): new segment on a gap >= 15 PAD frames or after a sentence end.
const segmentize = (texts: TextEvent[]): Seg[] => {
  const segs: Seg[] = [];
  let cur: Seg | null = null;
  let prevFrame = -1e9;
  for (const ev of texts) {
    if (!cur || ev.frame - prevFrame > 15 || SENT_END.test(cur.toks[cur.toks.length - 1]?.piece ?? "")) {
      cur = { start: ev.frame, t: ev.t ?? ev.frame / FRAME_HZ, toks: [] };
      segs.push(cur);
    }
    cur.toks.push(ev);
    prevFrame = ev.frame;
  }
  return segs;
};

export const TextStreamPanel: FC<{ state: Dep1State; className?: string }> = ({ state, className }) => {
  const box = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  const useEvents = state.texts.length > 0;
  const segs = useMemo(() => segmentize(state.texts), [state.texts]);
  const triggerFrames = useMemo(() => {
    const m = new Map<number, string>();
    state.actions.filter(a => a.stage === "trigger" && a.frame !== null && a.frame !== undefined)
      .forEach(a => m.set(a.frame as number, a.payload?.phrase ?? "trigger"));
    return m;
  }, [state.actions]);
  const nForced = useMemo(() => state.texts.filter(t => t.forced).length, [state.texts]);
  const lastFrame = state.texts.length ? state.texts[state.texts.length - 1].frame : null;

  useEffect(() => {
    if (follow && box.current) {
      box.current.scrollTop = box.current.scrollHeight;
    }
  }, [state.texts, state.stockPieces, follow]);

  return (
    <Card title="Live text stream" testId="text-panel" className={className}
      right={<span className="text-xs text-gray-500">
        {useEvents ? `${state.texts.length} tokens · frame ${lastFrame ?? "–"}${nForced ? ` · ${nForced} forced` : ""}` : `${state.stockPieces.length} pieces (stock 0x02)`}
        <label className="ml-2"><input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)} /> follow</label>
      </span>}>
      <div ref={box} className="overflow-y-auto flex-1 min-h-[12rem] max-h-[28rem] text-sm leading-relaxed font-mono bg-gray-50 rounded p-2">
        {useEvents ? segs.map((s, i) => (
          <div key={i} className="mb-1" data-testid="text-seg">
            <span className="text-gray-400 text-xs mr-2 select-none">[{fmtT(s.t)} f{s.start}]</span>
            {s.toks.map((t, j) => (
              <span key={j}>
                <span title={`frame ${t.frame} · t ${fmt(t.t, 2)} s · token ${t.token}${t.forced ? " · forced" : ""}`}
                  className={t.forced ? "bg-amber-200 text-amber-900" : ""}>{t.piece}</span>
                {triggerFrames.has(t.frame) && <span className="text-[10px] bg-red-100 text-red-800 rounded px-1 mx-0.5 align-middle">TRIGGER</span>}
              </span>
            ))}
          </div>
        )) : (
          <span>{state.stockPieces.join("")}</span>
        )}
        {!useEvents && state.stockPieces.length === 0 && <span className="text-gray-400">waiting for text…</span>}
      </div>
      <p className="text-[11px] text-gray-400 mt-1">Agent text stream (one entry per non-PAD token). Amber = forced by the inject hook. Line breaks: gap ≥ 15 frames or sentence end.</p>
    </Card>
  );
};

// ---------------------------------------------------------------------------------------------- actions
const STAGE_STYLE: Record<ActionStage, string> = {
  trigger: "bg-sky-100 text-sky-900 border-sky-300",
  asr: "bg-indigo-100 text-indigo-900 border-indigo-300",
  needle: "bg-violet-100 text-violet-900 border-violet-300",
  resolved: "bg-teal-100 text-teal-900 border-teal-300",
  executed: "bg-green-100 text-green-900 border-green-400",
  unbound: "bg-orange-100 text-orange-900 border-orange-400",
  needs_clarification: "bg-amber-100 text-amber-900 border-amber-400",
  error: "bg-red-100 text-red-900 border-red-400",
};

const j = (x: any) => {
  try { return JSON.stringify(x); } catch { return String(x); }
};

const StageDetail: FC<{ a: ActionEvent }> = ({ a }) => {
  const p = a.payload ?? {};
  switch (a.stage) {
    case "trigger":
      return <span>“{p.phrase}” · {p.rule}{p.score !== undefined && p.score !== null ? ` ${fmt(p.score, 2)}` : ""} · segment: <i>{p.segment}</i></span>;
    case "asr":
      return <span>{p.backend} · {fmt(p.audio_s, 1)} s audio · <span className="font-mono">{p.text}</span></span>;
    case "needle":
      return <span className="font-mono break-all">{j(p.function_calls)}{p.model_segments ? <span className="text-gray-500"> · model: {j(p.model_segments)}</span> : null}</span>;
    case "resolved":
      return <span className="font-mono break-all">{(p.calls ?? []).map((c: any, i: number) => (
        <span key={i} className="block">{c.name}({j(c.arguments)}) · order_ref={j(c.order_ref)} → <b>{c.resolved_id ?? "null"}</b> [{c.resolver_rule}]</span>
      ))}</span>;
    case "executed":
      return <span className="font-mono break-all"><b>{p.name}</b> on {p.resolved_id} → {j(p.result)}{p.record_diff ? <span className="block text-gray-600">diff: {j(p.record_diff)}</span> : null}</span>;
    case "unbound":
      return <span className="font-mono break-all">{p.name} · order_ref={j(p.order_ref)} · DROPPED: {p.reason}</span>;
    case "needs_clarification":
      return <span className="font-mono break-all"><b>{p.name}</b> NOT executed · ask the customer: {(p.ask_reasons ?? []).join(", ")}{p.server_args ? <span className="block text-gray-600">heard: {j(p.server_args)}</span> : null}</span>;
    case "error":
      return <span className="font-mono break-all">{p.where}: {p.message}</span>;
    default:
      return <span className="font-mono break-all">{j(p)}</span>;
  }
};

export const ActionsPanel: FC<{ actions: ActionEvent[]; className?: string }> = ({ actions, className }) => {
  const groups = useMemo(() => {
    const order: string[] = [];
    const m = new Map<string, ActionEvent[]>();
    for (const a of actions) {
      const k = a.trigger_id ?? "?";
      if (!m.has(k)) { m.set(k, []); order.push(k); }
      m.get(k)!.push(a);
    }
    return order.reverse().map(k => ({ id: k, evs: m.get(k)! })); // newest first
  }, [actions]);
  const counts = useMemo(() => {
    const c: Partial<Record<ActionStage, number>> = {};
    actions.forEach(a => { c[a.stage] = (c[a.stage] ?? 0) + 1; });
    return c;
  }, [actions]);

  return (
    <Card title="Actions (router)" testId="actions-panel" className={className}
      right={<span className="text-xs text-gray-500">{ACTION_STAGES.filter(s => counts[s]).map(s => `${s} ${counts[s]}`).join(" · ") || "no events yet"}</span>}>
      <p className="text-[11px] text-gray-400 mb-2">trigger detected → ASR transcript → Needle call → resolved call → executed (stub tool), or needs clarification (not executed: the agent must ask the customer).</p>
      {groups.length === 0 && <p className="text-xs text-gray-500">Waiting for a check-line phrase (“ek minute”, “let me check”, …).</p>}
      <div className="flex flex-col gap-2 overflow-y-auto max-h-[28rem]">
        {groups.map(g => {
          const trig = g.evs.find(e => e.stage === "trigger");
          const asr = g.evs.find(e => e.stage === "asr");
          const needle = g.evs.find(e => e.stage === "needle");
          const last = g.evs[g.evs.length - 1];
          return (
            <div key={g.id} data-testid="action-group" className="border border-gray-200 rounded p-2">
              <div className="flex flex-wrap items-baseline gap-x-3 text-xs mb-1">
                <span className="font-mono font-semibold">{g.id}</span>
                {trig?.frame !== undefined && trig?.frame !== null && <span className="text-gray-500">frame {trig.frame} ({fmtT((trig.frame as number) / FRAME_HZ)})</span>}
                <span className="text-gray-600">trigger→ASR {fmt(asr?.since_trigger_ms, 0)} ms</span>
                <span className="text-gray-600">ASR→Needle {fmt(needle?.latency_ms, 0)} ms</span>
                <span className="text-gray-800 font-medium">total {fmt(last?.since_trigger_ms, 0)} ms</span>
              </div>
              <ol className="flex flex-col gap-1">
                {g.evs.map((a, i) => (
                  <li key={i} className="flex gap-2 items-start text-xs">
                    <span className={`shrink-0 w-20 text-center border rounded px-1 break-words leading-tight ${STAGE_STYLE[a.stage] ?? "bg-gray-100"}`}>{a.stage === "needs_clarification" ? "needs clarification" : a.stage}</span>
                    <span className="shrink-0 w-28 text-gray-500 tabular-nums">{fmt(a.latency_ms, 0)} ms · +{fmt(a.since_trigger_ms, 0)}</span>
                    <span className="min-w-0"><StageDetail a={a} /></span>
                  </li>
                ))}
              </ol>
            </div>
          );
        })}
      </div>
    </Card>
  );
};

// ---------------------------------------------------------------------------------------------- metrics
const Stat: FC<{ label: string; b: StatBlock | undefined; warnAt?: number }> = ({ label, b, warnAt }) => (
  <tr>
    <td className="pr-2 text-gray-500">{label}</td>
    {(["last", "p50", "p95", "max"] as const).map(k => {
      const v = b ? b[k] : null;
      const bad = warnAt !== undefined && v !== null && v !== undefined && v > warnAt;
      return <td key={k} className={`pr-2 tabular-nums text-right ${bad ? "text-red-600 font-semibold" : ""}`}>{fmt(v, 1)}</td>;
    })}
    <td className="tabular-nums text-right text-gray-400">{b?.n ?? "–"}</td>
  </tr>
);

// Serverless fork (DESIGN.md 5.4 item 3): no 1 s HTTP /metrics poll through the relay; the socket's own 0x07
// metrics events (every 2 s, DEP1 INTERFACE 4.1) feed this panel.
export const MetricsPanel: FC<{ metricsEvent?: Metrics | null; className?: string }> = ({ metricsEvent, className }) => {
  const x = metricsEvent ?? null;
  const ctxLeft = x?.session?.context_frames_left;
  const smi = x?.vram?.nvidia_smi_used_mib;
  return (
    <Card title="Latency / server" testId="metrics-panel" className={className}
      right={<span className="text-xs text-gray-400">{x ? "0x07 metrics · every 2 s" : "waiting for metrics…"}</span>}>
      <table className="text-xs w-full">
        <thead><tr className="text-gray-400"><th className="text-left font-normal">ms</th><th className="text-right font-normal pr-2">last</th><th className="text-right font-normal pr-2">p50</th><th className="text-right font-normal pr-2">p95</th><th className="text-right font-normal pr-2">max</th><th className="text-right font-normal">n</th></tr></thead>
        <tbody>
          <Stat label="frame step" b={x?.step_ms} warnAt={STEP_TARGET_MS} />
          <Stat label="lm step" b={x?.lm_step_ms} />
        </tbody>
      </table>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs mt-2">
        <dt className="text-gray-500">RTF</dt><dd className={`tabular-nums ${x?.rtf && x.rtf > 1 ? "text-red-600 font-semibold" : ""}`}>{fmt(x?.rtf, 3)} <span className="text-gray-400">(target &lt; 1; step target ≤ {STEP_TARGET_MS} ms p95)</span></dd>
        <dt className="text-gray-500">VRAM</dt><dd className="tabular-nums">{smi ? `${smi} MiB used` : "–"} · alloc {fmt(x?.vram?.allocated_gib, 2)} GiB · max {fmt(x?.vram?.max_allocated_gib, 2)} GiB</dd>
        <dt className="text-gray-500">conversation</dt><dd className="tabular-nums">{x?.session?.active ? `${fmt(x.session.conv_s, 1)} s · frame ${x.session.frame ?? "–"}` : "no active session"}</dd>
        <dt className="text-gray-500">context left</dt><dd className="tabular-nums">{ctxLeft !== null && ctxLeft !== undefined ? `${ctxLeft} frames (${fmt(ctxLeft / FRAME_HZ, 0)} s)` : "–"}</dd>
        <dt className="text-gray-500">inject / mute</dt><dd>{x?.inject?.active ? `active (${x.inject.queued_frames ?? 0} queued)` : "off"} · {x?.mute ? "muted" : "not muted"}</dd>
        <dt className="text-gray-500">router counts</dt><dd>{x?.router?.counts ? ACTION_STAGES.map(s => `${s} ${x.router?.counts?.[s] ?? 0}`).join(" · ") : "–"}</dd>
      </dl>
    </Card>
  );
};

// ---------------------------------------------------------------------------------------------- samples
const N_PLACEHOLDERS = 5;
export const SamplesPanel: FC<{ className?: string }> = ({ className }) => {
  const [items, setItems] = useState<SampleItem[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetchSamples().then(setItems).catch(e => { setErr(String(e?.message ?? e)); setItems([]); });
  }, []);
  const n = items?.length ?? 0;
  return (
    <Card title="Handpicked samples" testId="samples-panel" className={className}
      right={<span className="text-xs text-gray-400">{items === null ? "loading…" : `${n} file${n === 1 ? "" : "s"}`}</span>}>
      <p className="text-[11px] text-gray-400 mb-2">Pre-rendered stereo recordings of the fine-tuned agent (chosen by hand, not a random sample).</p>
      <ul className="flex flex-col gap-2">
        {(items ?? []).map(s => (
          <li key={s.url} data-testid="sample-item" className="text-xs">
            <div className="flex justify-between"><span className="font-mono">{s.name}</span><span className="text-gray-500">{s.note}</span></div>
            <audio controls preload="none" src={s.url} className="w-full h-8" />
          </li>
        ))}
        {items !== null && n === 0 && Array.from({ length: N_PLACEHOLDERS }, (_, i) => (
          <li key={i} data-testid="sample-placeholder" className="text-xs text-gray-400 border border-dashed rounded px-2 py-1">
            sample {i + 1}: not supplied yet
          </li>
        ))}
      </ul>
      {items !== null && n === 0 && <p className="text-[11px] text-gray-400 mt-2">Drop stereo .wav/.mp3 files (+ notes.json) into <span className="font-mono">space/samples/</span> before pushing the Space{err ? ` (/api/samples: ${err})` : ""}.</p>}
    </Card>
  );
};
