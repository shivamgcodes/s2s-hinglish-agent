// DEP1 client-side types. Source of truth: /root/deploy/INTERFACE.md (frozen). Display only.

export type Pairing = "g1" | "g2" | "g3" | "g4";
export const PAIRINGS: Pairing[] = ["g1", "g2", "g3", "g4"];
// INTERFACE.md section 3: g1/g3 female agent -> NATF2.pt, g2/g4 male -> NATM1.pt
export const PAIRING_GENDER: Record<Pairing, "f" | "m"> = { g1: "f", g2: "m", g3: "f", g4: "m" };
export const GENDER_VOICE: Record<"f" | "m", string> = { f: "NATF2.pt", m: "NATM1.pt" };
export const DEFAULT_SEED = 1001;

export type RecordSummary = {
  record_id: string;
  agent_type: string;
  brand: string;
  agent_name_f: string;
  agent_name_m: string;
  information: string;
  primary_id: string;
  secondary_id: string;
};

export type SessionConfig = {
  type?: "session_config";
  agent_type: string;
  record_id: string;
  pairing: Pairing;
  agent_gender: "f" | "m";
  voice: string;
  role_prompt: string;
  seed: number;
  router_supported: boolean;
};

export type FullRecord = { [k: string]: any } & {
  session_configs?: Partial<Record<Pairing, SessionConfig>>;
};

// What the setup panel hands to the Conversation page.
export type Dep1Session = {
  recordId: string;
  pairing: Pairing;
  agentType: string;
  seed: number;
  preview?: SessionConfig; // client-side copy of the config, shown until the server's "session" event arrives
};

// needs_clarification (D-ROUTER-V2, 2026-10-06): the Needle v2 router could not fill a call safely (e.g. a 9-digit
// phone, an unknown order ID); the call is NOT executed, the agent has to ask the customer.
export type ActionStage = "trigger" | "asr" | "needle" | "resolved" | "executed" | "unbound" | "needs_clarification" | "error";
export const ACTION_STAGES: ActionStage[] = ["trigger", "asr", "needle", "resolved", "executed", "unbound", "needs_clarification", "error"];

// Turn filler ("ticker") fed to the MODEL input only (2026-10-06): one event per real transition, from the worker's
// engine thread. start = the agent paused (not after a check-line), the 1 s ticker goes to the model from the next
// frame; stop = buffer done (reason done) or switched off (reason off); cancel = the customer started speaking;
// skip_check = the agent said a check-line, so no filler.
export type FillerEvent = {
  type: "filler";
  state: "start" | "stop" | "cancel" | "skip_check";
  reason?: string;
  frame: number;
  t: number;
  mode: string;
  seconds?: number;
  session_id?: string;
  t_wall?: number;
};
export type TurnFillEvent = { type: "turn_fill"; mode: string; t_wall?: number };
export type TurnFillMetrics = { mode?: string; source?: string; seconds?: number; fills?: number;
  skipped_check_line?: number; cancelled_by_user?: number; filling?: boolean } | null;

export type ActionEvent = {
  type: "action";
  session_id?: string;
  t_wall?: number;
  trigger_id: string;
  stage: ActionStage;
  payload: any;
  latency_ms: number | null;
  since_trigger_ms: number | null;
  frame: number | null;
};

export type TextEvent = {
  type: "text";
  frame: number;
  t: number;
  token: number;
  piece: string;
  forced: boolean;
};

export type StatBlock = { last: number | null; p50: number | null; p95: number | null; max: number | null; n: number | null } | null;

export type Metrics = {
  t_wall?: number;
  session?: {
    active: boolean;
    session_id?: string | null;
    record_id?: string | null;
    agent_type?: string | null;
    frame?: number | null;
    conv_s?: number | null;
    context_frames_left?: number | null;
  } | null;
  step_ms?: StatBlock;
  lm_step_ms?: StatBlock;
  rtf?: number | null;
  vram?: {
    allocated_gib?: number | null;
    reserved_gib?: number | null;
    max_allocated_gib?: number | null;
    nvidia_smi_used_mib?: number | null;
  } | null;
  ring?: { seconds?: number | null; end_sample?: number | null } | null;
  inject?: { active?: boolean | null; queued_frames?: number | null } | null;
  mute?: boolean | null;
  router?: { last_event_t_wall?: number | null; counts?: Partial<Record<ActionStage, number>> | null } | null;
  turn_fill?: TurnFillMetrics;
};

export type SampleItem = { name: string; url: string; note?: string };

// Serverless Space (DESIGN.md 5.1)
export type SpaceConfig = {
  mode: "lb" | "queue";
  passcode_required: boolean;
  max_call_s: number;
  max_concurrent: number;
  build?: string;
  allow_free_prompt?: boolean;
  claim_ttl_s?: number;
  ok?: boolean;
};

export type SpaceSessionState = "waking" | "busy" | "ready" | "in_call" | "ended" | "failed";

export type SpaceSession = {
  sid: string;
  state: SpaceSessionState;
  detail?: string;
  elapsed_s?: number;
  state_elapsed_s?: number;
  worker_id?: string;
  end_reason?: string;
};

// session_end reasons: DEP1 (client_closed, context_full, error) + serverless (time_limit, worker_shutdown from the
// worker; worker_lost, relay_error synthesised by the Space relay)
export const END_REASON_TEXT: Record<string, string> = {
  client_closed: "you disconnected",
  context_full: "the model's context is full (about 3.5 min per call)",
  error: "server error",
  time_limit: "call time limit reached",
  worker_lost: "the GPU worker went away (scale-down or crash); start a new call",
  worker_shutdown: "the GPU worker is shutting down; start a new call",
  relay_error: "could not reach the GPU worker",
};

// D-SCRIPT-PANEL (2026-10-06): GET /api/script/{record_id}?pairing=gN, the expected V4 synthetic conversation.
export type ScriptTurn = { speaker: "agent" | "customer"; text: string; tags: string[] };
export type ScriptWrite = { tool: string; args: Record<string, unknown> };
export type ScriptSplit = "train" | "val" | "test" | "test_scenario";
export type ScriptResponse = {
  record_id: string;
  pairing: string;
  call_id: string;
  available: boolean;
  split?: ScriptSplit;
  split_note?: string;
  agent_name?: string;
  turns: ScriptTurn[];
  writes: ScriptWrite[];
  detail?: string;
};
