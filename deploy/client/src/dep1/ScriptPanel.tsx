// D-SCRIPT-PANEL (2026-10-06): "Script: what to say", shown to the right of the live text stream. It shows the
// expected conversation for the selected record + pairing: the V4 synthetic call the LoRA was trained / evaluated on.
// Customer lines are what the user says. Agent lines are dimmed: they show what the model is expected to say.
// The next customer line is highlighted with a rough heuristic: the agent lines whose words show up, in order, in
// the model's text stream count as spoken.
import { FC, useEffect, useMemo, useRef, useState } from "react";
import { fetchScript } from "./api";
import { Card } from "./Panels";
import { ScriptResponse, ScriptTurn, TextEvent } from "./types";

const SPLIT_TEXT: Record<string, string> = {
  train: "train split: the model was fine-tuned on this exact dialogue",
  val: "validation split: never trained on",
  test: "test split (held-out call): never trained on",
  test_scenario: "held-out test scenario: never trained on",
};
const TAG_TEXT: Record<string, [string, string]> = {
  check_line: ["check", "the line that triggers the router (Needle)"],
  confirm_write: ["write", "the agent confirms the change back"],
  read: ["read", "the agent states a fact from the record"],
};

// very common Hinglish / support-call words: they match almost any agent line, so they are not evidence
const STOP = new Set(("main mein aap aapka aapki aapke hai hain hoon kar karo karna sakti sakta sakte help please sir " +
  "madam kya bataiye batayiye order theek thik toh nahi haan yes the and for you your this that kaise kahan abhi " +
  "bhi koi kuch aur liye wala wali hello namaste thank thanks welcome second dekh deti deta leti leta raha rahi " +
  "rahe gaya gayi gaye hua hui diya diye").split(" "));
const words = (s: string): string[] =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(" ").filter(w => w.length >= 3 && !STOP.has(w));

// Index of the last agent turn that the model text appears to have spoken (-1: none yet). Agent turns are matched in
// order: one of the next 3 agent turns counts as spoken when at least half of its distinct content words (2 or more;
// common words ignored) appear in a window
// of the model's words after the previous match. Approximate on purpose.
export const lastSpokenAgentTurn = (turns: ScriptTurn[], modelText: string): number => {
  const m = words(modelText);
  const agent = turns.map((t, i) => (t.speaker === "agent" ? i : -1)).filter(i => i >= 0);
  let cursor = 0;
  let last = -1;
  let k = 0;
  while (k < agent.length && cursor < m.length) {
    let matched = false;
    for (let c = k; c < Math.min(k + 3, agent.length); c++) {
      const w = Array.from(new Set(words(turns[agent[c]].text)));
      if (w.length < 2) continue;
      const win = m.slice(cursor, cursor + 3 * w.length + 30);
      let found = 0;
      let maxPos = -1;
      for (const x of w) {
        const p = win.indexOf(x);
        if (p >= 0) {
          found++;
          maxPos = Math.max(maxPos, p);
        }
      }
      if (found / w.length >= 0.5) {
        last = agent[c];
        cursor += maxPos + 1;
        k = c + 1;
        matched = true;
        break;
      }
    }
    if (!matched) break;
  }
  return last;
};

const fmtArgs = (args: Record<string, unknown>) =>
  Object.entries(args).map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v) : String(v)}`).join(", ");

export const ScriptPanel: FC<{ recordId?: string | null; pairing?: string | null; texts: TextEvent[]; className?: string }> = ({
  recordId, pairing, texts, className,
}) => {
  const [script, setScript] = useState<ScriptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setScript(null);
    setError(null);
    if (!recordId || !pairing) return;
    let alive = true;
    fetchScript(recordId, pairing)
      .then(s => alive && setScript(s))
      .catch(e => alive && setError(String(e?.message ?? e)));
    return () => {
      alive = false;
    };
  }, [recordId, pairing]);

  const modelText = useMemo(() => texts.map(t => t.piece).join(""), [texts]);
  const turns = script?.available ? script.turns : [];
  const next = useMemo(() => {
    if (!turns.length) return -1;
    const last = lastSpokenAgentTurn(turns, modelText);
    return turns.findIndex((t, i) => i > last && t.speaker === "customer");
  }, [turns, modelText]);

  useEffect(() => {
    // keep the next line in view inside the panel only (scrollIntoView would also scroll the page)
    const el = box.current?.querySelector<HTMLElement>("[data-next='1']");
    if (follow && box.current && el) {
      box.current.scrollTop = Math.max(0, el.offsetTop - box.current.offsetTop - 48);
    }
  }, [next, follow, script]);

  const split = script?.split ?? "";
  return (
    <Card title="Script: what to say" testId="script-panel" className={className}
      right={<span className="text-xs text-gray-500 flex items-center gap-2">
        <label><input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)} /> follow</label>
      </span>}>
      {!recordId && <p className="text-xs text-gray-500">No script for a stock session (no record).</p>}
      {recordId && error && <p className="text-xs text-red-700">Script not loaded: {error}</p>}
      {recordId && !error && !script && <p className="text-xs text-gray-400">loading script…</p>}
      {script && !script.available && (
        <p data-testid="script-none" className="text-xs text-gray-600 bg-gray-50 border rounded p-2">
          No script for this pairing ({script.call_id}): that synthetic call was not generated. Try another pairing.
        </p>
      )}
      {script?.available && (
        <>
          <p className="text-[11px] text-gray-500 mb-1">
            You are the <b>customer</b>: say the highlighted lines in Hinglish, in your own words if you like. Agent
            lines show what the model is expected to say; it will not follow them word for word.
          </p>
          <div ref={box} data-testid="script-turns" className="overflow-y-auto flex-1 min-h-[12rem] max-h-[28rem] text-sm leading-snug flex flex-col gap-1 pr-1">
            {turns.map((t, i) => {
              const isNext = i === next;
              const tags = t.tags.filter(x => TAG_TEXT[x]);
              return t.speaker === "customer" ? (
                <div key={i} data-testid="script-customer" data-next={isNext ? "1" : "0"}
                  className={`rounded border-l-4 px-2 py-1 ${isNext ? "border-blue-600 bg-blue-100 ring-1 ring-blue-300" : "border-blue-400 bg-blue-50"}`}>
                  <span className="block text-[10px] uppercase tracking-wide text-blue-800 font-semibold">you say{isNext ? " · next" : ""}</span>
                  <span className="text-gray-900">{t.text}</span>
                </div>
              ) : (
                <div key={i} data-testid="script-agent" className="px-2 py-0.5 text-gray-500 text-[13px]">
                  <span className="text-[10px] uppercase tracking-wide text-gray-400 mr-1">agent (expected)</span>
                  <span className="italic">{t.text}</span>
                  {tags.map(x => (
                    <span key={x} title={TAG_TEXT[x][1]} data-testid={`script-tag-${x}`}
                      className="ml-1 align-middle text-[10px] border border-gray-300 text-gray-500 rounded px-1 not-italic">{TAG_TEXT[x][0]}</span>
                  ))}
                </div>
              );
            })}
            {script.writes.length > 0 && (
              <div className="mt-1 text-[11px] text-gray-600 bg-gray-50 border rounded px-2 py-1">
                Expected action{script.writes.length > 1 ? "s" : ""}:{" "}
                {script.writes.map((w, i) => <code key={i} className="block break-words">{w.tool}({fmtArgs(w.args)})</code>)}
              </div>
            )}
          </div>
          <p className="text-[11px] text-gray-400 mt-1" data-testid="script-note">
            Synthetic V4 call <span className="font-mono">{script.call_id}</span> · {SPLIT_TEXT[split] ?? split}.
            Tags: <i>check</i> = router trigger line, <i>write</i> = agent confirms the change, <i>read</i> = fact
            from the record. &quot;next&quot; is a rough guess from the model&apos;s text.
          </p>
        </>
      )}
    </Card>
  );
};
