// Collects DEP1 kind-0x07 events (and stock 0x02 text as a fallback) from the live /api/chat socket.
import { useCallback, useEffect, useRef, useState } from "react";
import { decodeMessage } from "../protocol/encoder";
import { ActionEvent, FillerEvent, Metrics, SessionConfig, TextEvent, TurnFillEvent } from "./types";

const MAX_TEXT = 4000; // context limit ~2770 frames (~221 s, D8); a text event is at most one per frame

export type SessionEvent = SessionConfig & { session_id?: string; context_frames_left?: number; t_wall?: number };
export type SessionEnd = { reason: string; frames: number | null; t_wall?: number; detail?: string; source?: string };
export type InjectEvent = { state: string; first_frame: number; last_frame: number; n_frames: number; t_wall?: number };

export type Dep1State = {
  session: SessionEvent | null;
  texts: TextEvent[];          // from 0x07 text events
  stockPieces: string[];       // from 0x02 (always sent by the server; fallback when events are absent)
  actions: ActionEvent[];      // in arrival order
  metricsEvent: Metrics | null;
  injects: InjectEvent[];
  filler: FillerEvent | null;          // last turn-filler transition (ticker indicator)
  fillerRecvAt: number | null;         // performance.now() when it arrived
  nFills: number;                      // filler "start" events this call
  turnFill: TurnFillEvent | null;      // last mid-call toggle confirmation from the worker
  sessionEnd: SessionEnd | null;
  nEvents: number;
  nUnknown: number;
};

const empty = (): Dep1State => ({
  session: null, texts: [], stockPieces: [], actions: [], metricsEvent: null,
  injects: [], filler: null, fillerRecvAt: null, nFills: 0, turnFill: null, sessionEnd: null, nEvents: 0, nUnknown: 0,
});

export const useDep1Events = (socket: WebSocket | null): Dep1State => {
  const [state, setState] = useState<Dep1State>(empty);
  // batch updates: socket messages arrive at up to ~25/s (audio + text); flush every 100 ms
  const pending = useRef<((s: Dep1State) => Dep1State)[]>([]);

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const msg = decodeMessage(new Uint8Array(e.data));
    if (msg.type === "text") {
      pending.current.push(s => ({ ...s, stockPieces: [...s.stockPieces, msg.data].slice(-MAX_TEXT) }));
      return;
    }
    if (msg.type === "unknown") {
      pending.current.push(s => ({ ...s, nUnknown: s.nUnknown + 1 }));
      return;
    }
    if (msg.type !== "event") {
      return;
    }
    const ev = msg.data ?? {};
    const recvAt = performance.now();
    pending.current.push(s => {
      const n = { ...s, nEvents: s.nEvents + 1 };
      switch (ev.type) {
        case "session":
        case "session_config": // tolerate a server that forwards the config object unchanged
          return { ...n, session: ev as SessionEvent };
        case "text":
          return { ...n, texts: [...s.texts, ev as TextEvent].slice(-MAX_TEXT) };
        case "action":
          return { ...n, actions: [...s.actions, ev as ActionEvent] };
        case "metrics":
          return { ...n, metricsEvent: ev as Metrics };
        case "inject":
          return { ...n, injects: [...s.injects, ev as InjectEvent] };
        case "filler":
          return { ...n, filler: ev as FillerEvent, fillerRecvAt: recvAt, nFills: s.nFills + (ev.state === "start" ? 1 : 0) };
        case "turn_fill":
          return { ...n, turnFill: ev as TurnFillEvent };
        case "session_end":
          return { ...n, sessionEnd: ev as SessionEnd };
        default:
          return n;
      }
    });
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      if (pending.current.length === 0) return;
      const fns = pending.current;
      pending.current = [];
      setState(s => fns.reduce((acc, f) => f(acc), s));
    }, 100);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const current = socket;
    if (!current) {
      return;
    }
    pending.current = [];
    setState(empty());
    current.addEventListener("message", onSocketMessage);
    return () => {
      current.removeEventListener("message", onSocketMessage);
    };
  }, [socket, onSocketMessage]);

  return state;
};
