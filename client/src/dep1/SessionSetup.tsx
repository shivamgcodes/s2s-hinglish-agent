// Session-setup panel: agent type -> demo record -> pairing (agent gender/voice) -> role-prompt preview.
// The prompt shown is the server's own session_config (GET /api/records/{id}.session_configs[gN]); the client
// sends only record_id/pairing/agent_type/seed and lets the server build the prompt (INTERFACE.md section 3).
// Serverless fork: records come from the Space (computed locally there); Connect hands the choice + passcode (when
// /api/config asks for one) to Queue.tsx, which creates the Space session. The stock free-prompt panel is shown only
// when /api/config says allow_free_prompt (the worker refuses free prompts by default).
import { FC, Fragment, useEffect, useMemo, useState } from "react";
import { fetchConfig, fetchRecord, fetchRecords } from "./api";
import { Card, Disclaimer, SamplesPanel } from "./Panels";
import { DEFAULT_SEED, Dep1Session, FullRecord, GENDER_VOICE, PAIRING_GENDER, PAIRINGS, Pairing, RecordSummary, SpaceConfig } from "./types";

const AGENT_TYPE_LABEL: Record<string, string> = {
  food_delivery_support: "Food delivery support",
  ecommerce_support: "E-commerce support",
  cab_ride_support: "Cab ride support",
  subscription_account_support: "Subscription / account support",
  airport_ticket_counter: "Airport ticket counter",
};
const label = (t: string) => AGENT_TYPE_LABEL[t] ?? t;

const SKIP_FIELDS = new Set(["role_prompts", "session_configs", "information_tokens_spm", "record_errors", "information"]);

const RecordView: FC<{ rec: FullRecord }> = ({ rec }) => {
  const [open, setOpen] = useState(false);
  const facts = rec.facts && typeof rec.facts === "object" ? rec.facts : null;
  return (
    <div className="text-xs">
      {facts && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 mb-2" data-testid="record-facts">
          {Object.entries(facts).map(([k, v]) => (
            <Fragment key={k}><dt className="text-gray-500">{k.replace(/_/g, " ")}</dt><dd>{String(v)}</dd></Fragment>
          ))}
        </dl>
      )}
      {Array.isArray(rec.distractors) && rec.distractors.length > 0 && (
        <p className="text-gray-600 mb-1"><span className="text-gray-500">also on file:</span> {rec.distractors.join("; ")}</p>
      )}
      {rec.caller_situation && <p className="text-gray-600 mb-1"><span className="text-gray-500">caller situation (not in prompt):</span> {rec.caller_situation}</p>}
      <button className="text-blue-700 underline" onClick={() => setOpen(o => !o)}>{open ? "hide" : "show"} full record JSON</button>
      {open && (
        <pre className="mt-1 max-h-64 overflow-auto bg-gray-50 border rounded p-2 text-[11px]">
          {JSON.stringify(Object.fromEntries(Object.entries(rec).filter(([k]) => !SKIP_FIELDS.has(k))), null, 1)}
        </pre>
      )}
    </div>
  );
};

type Props = {
  onStart: (s: Dep1Session, passcode: string) => Promise<void>;
  onStartStock: (passcode: string) => Promise<void>;
  startError?: string | null;
  showMicrophoneAccessMessage: boolean;
  stockTextPrompt: string;
  setStockTextPrompt: (v: string) => void;
  stockVoicePrompt: string;
  setStockVoicePrompt: (v: string) => void;
};

export const SessionSetup: FC<Props> = ({
  onStart, onStartStock, startError, showMicrophoneAccessMessage,
  stockTextPrompt, setStockTextPrompt, stockVoicePrompt, setStockVoicePrompt,
}) => {
  const [records, setRecords] = useState<RecordSummary[] | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [agentType, setAgentType] = useState<string>("");
  const [recordId, setRecordId] = useState<string>("");
  const [pairing, setPairing] = useState<Pairing>("g1");
  const [seedStr, setSeedStr] = useState<string>(String(DEFAULT_SEED)); // raw input; parsed at connect ("-" or "" -> default)
  const [full, setFull] = useState<FullRecord | null>(null);
  const [recErr, setRecErr] = useState<string | null>(null);
  const [stock, setStock] = useState(false);
  const [busy, setBusy] = useState(false);
  const [cfgSpace, setCfgSpace] = useState<SpaceConfig | null>(null);
  const [passcode, setPasscode] = useState<string>("");

  useEffect(() => {
    fetchConfig().then(setCfgSpace).catch(() => setCfgSpace(null));
  }, []);

  useEffect(() => {
    fetchRecords().then(rs => {
      setRecords(rs);
      if (rs.length) {
        setAgentType(rs[0].agent_type);
        setRecordId(rs[0].record_id);
      }
    }).catch(e => setLoadErr(String(e?.message ?? e)));
  }, []);

  const agentTypes = useMemo(() => Array.from(new Set((records ?? []).map(r => r.agent_type))), [records]);
  const inType = useMemo(() => (records ?? []).filter(r => r.agent_type === agentType), [records, agentType]);
  const summary = useMemo(() => (records ?? []).find(r => r.record_id === recordId) ?? null, [records, recordId]);

  useEffect(() => {
    if (!recordId) return;
    let alive = true;
    setFull(null);
    setRecErr(null);
    fetchRecord(recordId).then(r => { if (alive) setFull(r); }).catch(e => { if (alive) setRecErr(String(e?.message ?? e)); });
    return () => { alive = false; };
  }, [recordId]);

  const gender = PAIRING_GENDER[pairing];
  const cfg = full?.session_configs?.[pairing];
  const agentName = summary ? (gender === "f" ? summary.agent_name_f : summary.agent_name_m) : "";

  const start = async () => {
    if (!recordId) return;
    setBusy(true);
    try {
      const v = parseInt(seedStr, 10);
      await onStart({ recordId, pairing, agentType, seed: Number.isFinite(v) ? v : DEFAULT_SEED, preview: cfg }, passcode);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-neutral-50 p-4">
      <div className="max-w-5xl mx-auto flex flex-col gap-4">
        <header>
          <h1 className="text-2xl text-black font-semibold">Hinglish support agent · live demo</h1>
          <p className="text-sm text-gray-600">PersonaPlex full-duplex speech model + V3 Hinglish LoRA, with a Needle action router. Pick a synthetic customer record, then talk as the customer.</p>
        </header>
        <Disclaimer />

        <div className="grid md:grid-cols-[1fr_1.3fr] gap-4">
          <Card title="1 · Session setup" testId="setup-panel">
            {loadErr && <p className="text-sm text-red-600">Could not load /api/records: {loadErr}</p>}
            {!records && !loadErr && <p className="text-sm text-gray-500">loading records…</p>}
            {records && (
              <div className="flex flex-col gap-3 text-sm">
                <label className="flex flex-col gap-1">
                  <span className="text-gray-600">Agent type</span>
                  <select data-testid="sel-agent-type" className="p-2 border rounded bg-white" value={agentType}
                    onChange={e => { setAgentType(e.target.value); const first = records.find(r => r.agent_type === e.target.value); setRecordId(first?.record_id ?? ""); }}>
                    {agentTypes.map(t => <option key={t} value={t}>{label(t)} ({records.filter(r => r.agent_type === t).length})</option>)}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-gray-600">Demo record</span>
                  <select data-testid="sel-record" className="p-2 border rounded bg-white" value={recordId} onChange={e => setRecordId(e.target.value)}>
                    {inType.map(r => <option key={r.record_id} value={r.record_id}>{r.record_id} · {r.brand} · {r.primary_id}</option>)}
                  </select>
                </label>
                <div className="flex flex-col gap-1">
                  <span className="text-gray-600">Agent (pairing → voice)</span>
                  <div className="flex gap-2 flex-wrap" data-testid="sel-pairing">
                    {PAIRINGS.map(p => {
                      const g = PAIRING_GENDER[p];
                      const nm = summary ? (g === "f" ? summary.agent_name_f : summary.agent_name_m) : "";
                      return (
                        <button key={p} onClick={() => setPairing(p)}
                          className={`px-2 py-1 rounded border text-xs ${p === pairing ? "bg-[#76b900] text-white border-[#5a8d00]" : "bg-white text-gray-700"}`}>
                          {p} · {nm} ({g === "f" ? "F" : "M"})
                        </button>
                      );
                    })}
                  </div>
                  <span className="text-xs text-gray-500">Voice: <span className="font-mono">{cfg?.voice ?? GENDER_VOICE[gender]}</span> ({gender === "f" ? "female" : "male"} agent{agentName ? `, ${agentName}` : ""})</span>
                </div>
                <label className="flex items-center gap-2">
                  <span className="text-gray-600">Seed</span>
                  <input type="text" inputMode="numeric" data-testid="seed" className="p-1 border rounded w-28" value={seedStr} onChange={e => setSeedStr(e.target.value)} />
                  <span className="text-xs text-gray-400">default {DEFAULT_SEED}; -1 = no reseed</span>
                </label>
                {cfgSpace?.passcode_required && (
                  <label className="flex items-center gap-2">
                    <span className="text-gray-600">Passcode</span>
                    <input type="password" data-testid="passcode" className="p-1 border rounded w-40" value={passcode} onChange={e => setPasscode(e.target.value)} />
                  </label>
                )}
                {showMicrophoneAccessMessage && <p className="text-red-600">Please enable your microphone before proceeding.</p>}
                {startError && <p data-testid="start-error" className="text-sm text-red-600">{startError}</p>}
                <button data-testid="btn-connect" disabled={!recordId || busy}
                  onClick={start}
                  className="mt-1 py-2 px-4 rounded-lg bg-[#76b900] text-white font-semibold disabled:bg-gray-300">
                  {busy ? "Starting…" : "Connect (microphone)"}
                </button>
                <p className="text-xs text-gray-500">Connect wakes a GPU worker (cold start about 1-2 min after a quiet period), then the model runs the prompt phase (≈10 s) before audio starts. Use headphones; one call per worker, up to {cfgSpace?.max_call_s ?? 300} s.</p>
                {cfgSpace && cfgSpace.ok === false && <p className="text-xs text-red-600">The Space is not fully configured (see its logs); calls will fail.</p>}
              </div>
            )}
          </Card>

          <Card title="2 · Record and role prompt" testId="preview-panel">
            {recErr && <p className="text-sm text-red-600">Could not load record: {recErr}</p>}
            {!full && !recErr && <p className="text-sm text-gray-500">loading…</p>}
            {full && (
              <div className="flex flex-col gap-3">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Role prompt sent to the model ({pairing}, built by the server from records.json)</div>
                  <p data-testid="role-prompt-preview" className="text-xs bg-gray-50 border rounded p-2 whitespace-pre-wrap">{cfg?.role_prompt ?? "(server returned no session_configs for this pairing)"}</p>
                </div>
                <RecordView rec={full} />
              </div>
            )}
          </Card>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <SamplesPanel />
          {cfgSpace?.allow_free_prompt && <Card title="Stock PersonaPlex session (no record)" testId="stock-panel"
            right={<button className="text-xs text-blue-700 underline" onClick={() => setStock(s => !s)}>{stock ? "hide" : "show"}</button>}>
            <p className="text-xs text-gray-500">Free-form prompt, no record: the action router stays idle.</p>
            {stock && (
              <div className="flex flex-col gap-2 mt-2 text-sm">
                <textarea className="p-2 border rounded h-24" value={stockTextPrompt} maxLength={1000} onChange={e => setStockTextPrompt(e.target.value)} />
                <select className="p-2 border rounded bg-white" value={stockVoicePrompt} onChange={e => setStockVoicePrompt(e.target.value)}>
                  {["NATF2.pt", "NATM1.pt", "NATF0.pt", "NATF1.pt", "NATF3.pt", "NATM0.pt", "NATM2.pt", "NATM3.pt"].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
                <button className="py-1 px-3 rounded border bg-white" onClick={() => onStartStock(passcode)}>Connect (stock)</button>
              </div>
            )}
          </Card>}
        </div>
      </div>
    </div>
  );
};
