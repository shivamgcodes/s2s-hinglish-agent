import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";
import { SessionSetup } from "../../dep1/SessionSetup";
import { Dep1Session } from "../../dep1/types";
import { Warmup } from "../../dep1/Warmup";
import { createSession, deleteSession } from "../../dep1/api";

// DEP1 fork of the PersonaPlex Queue page: the stock prompt/voice homepage is replaced by the DEP1
// session-setup panel (records.json demo record -> server-built role prompt + voice). A stock free-prompt
// session is still available from the same page.
// Serverless fork (DESIGN.md 5.4): Connect first gets the microphone + AudioContext (user gesture), then
// POST /api/session; the Warmup panel polls until the Space reports "ready"; only then is <Conversation> mounted
// (it opens /api/chat?sid=… on mount). DELETE /api/session/{sid} on cancel, disconnect and pagehide.
export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [dep1, setDep1] = useState<Dep1Session | null>(null);
  const [started, setStarted] = useState(false);
  const [sid, setSid] = useState<string | null>(null);
  const [phase, setPhase] = useState<"setup" | "warming" | "call">("setup");
  const [startError, setStartError] = useState<string | null>(null);
  const modelParams = useModelParams({ voicePrompt: "NATF2.pt" });

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
    await startProcessor();
    await getMicrophoneAccess();
  }, [startProcessor, getMicrophoneAccess]);

  // pagehide: release the worker if the tab goes away mid-warmup or mid-call
  useEffect(() => {
    if (!sid) return;
    const onHide = () => deleteSession(sid);
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, [sid]);

  const begin = useCallback(async (s: Dep1Session | null, passcode: string) => {
    setStartError(null);
    setDep1(s);
    await startProcessor();           // AudioContext + mic inside the click (user gesture), before waking a GPU
    if (!(await getMicrophoneAccess())) {
      return;                         // SessionSetup shows "Please enable your microphone"
    }
    try {
      const js = await createSession(s ? { record_id: s.recordId, pairing: s.pairing, seed: s.seed, passcode }
                                       : { record_id: null, pairing: "g1", seed: null, passcode });
      setSid(js.sid);
      setPhase("warming");
    } catch (e: any) {
      setStartError(String(e?.message ?? e));
    }
  }, [startProcessor, getMicrophoneAccess]);

  const startDep1 = useCallback((s: Dep1Session, passcode: string) => begin(s, passcode), [begin]);
  const startStock = useCallback((passcode: string) => begin(null, passcode), [begin]);

  const onReady = useCallback(() => {
    setPhase("call");
    setStarted(true);
  }, []);

  const onBack = useCallback(() => {
    setPhase("setup");
    setSid(null);
    setStarted(false);
  }, []);

  return (
    <>
      {(phase === "call" && sid && started && hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        sid={sid}
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        dep1={dep1}
        {...modelParams}
        />
      ) : (phase === "warming" && sid) ? (
        <Warmup sid={sid} dep1={dep1} onReady={onReady} onBack={onBack} />
      ) : (
        <SessionSetup
          onStart={startDep1}
          onStartStock={startStock}
          startError={startError}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          stockTextPrompt={modelParams.textPrompt}
          setStockTextPrompt={modelParams.setTextPrompt}
          stockVoicePrompt={modelParams.voicePrompt}
          setStockVoicePrompt={modelParams.setVoicePrompt}
        />
      )}
    </>
  );
};
