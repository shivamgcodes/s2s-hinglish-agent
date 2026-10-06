import { FC, MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSocket } from "./hooks/useSocket";
import { SocketContext } from "./SocketContext";
import { ServerAudio } from "./components/ServerAudio/ServerAudio";
import { UserAudio } from "./components/UserAudio/UserAudio";
import { TickerIndicator } from "./components/AudioVisualizer/TickerVisualizer";
import { Button } from "../../components/Button/Button";
import { ServerAudioStats } from "./components/ServerAudio/ServerAudioStats";
import { AudioStats } from "./hooks/useServerAudio";
import { MediaContext } from "./MediaContext";
import { ServerInfo } from "./components/ServerInfo/ServerInfo";
import { ModelParamsValues, useModelParams } from "./hooks/useModelParams";
import fixWebmDuration from "webm-duration-fix";
import { getMimeType, getExtension } from "./getMimeType";
import { type ThemeType } from "./hooks/useSystemTheme";
import { Dep1Session, END_REASON_TEXT } from "../../dep1/types";
import { deleteSession } from "../../dep1/api";
import { useDep1Events } from "../../dep1/useDep1Events";
import { ActionsPanel, Disclaimer, MetricsPanel, SamplesPanel, SessionInfo, TextStreamPanel } from "../../dep1/Panels";
import { ScriptPanel } from "../../dep1/ScriptPanel";

type ConversationProps = {
  workerAddr: string;
  workerAuthId?: string;
  sessionAuthId?: string;
  sessionId?: number;
  email?: string;
  theme: ThemeType;
  audioContext: MutableRefObject<AudioContext|null>;
  worklet: MutableRefObject<AudioWorkletNode|null>;
  onConversationEnd?: () => void;
  isBypass?: boolean;
  startConnection: () => Promise<void>;
  dep1?: Dep1Session | null;
  sid?: string | null;
} & Partial<ModelParamsValues>;


const buildURL = ({
  workerAddr,
  params,
  workerAuthId,
  email,
  textSeed,
  audioSeed,
  dep1,
  sid,
}: {
  dep1?: Dep1Session | null;
  sid?: string | null;
  workerAddr: string;
  params: ModelParamsValues;
  workerAuthId?: string;
  email?: string;
  textSeed: number;
  audioSeed: number;
}) => {
  const newWorkerAddr = useMemo(() => {
    if (workerAddr == "same" || workerAddr == "") {
      // serverless fork: host (with port only if there is one). hostname + ":" + port gave "host:" on hf.space.
      const newWorkerAddr = window.location.host;
      console.log("Overriding workerAddr to", newWorkerAddr);
      return newWorkerAddr;
    }
    return workerAddr;
  }, [workerAddr]);
  const wsProtocol = (window.location.protocol === 'https:') ? 'wss' : 'ws';
  const url = new URL(`${wsProtocol}://${newWorkerAddr}/api/chat`);
  if(workerAuthId) {
    url.searchParams.append("worker_auth_id", workerAuthId);
  }
  if(email) {
    url.searchParams.append("email", email);
  }
  url.searchParams.append("text_temperature", params.textTemperature.toString());
  url.searchParams.append("text_topk", params.textTopk.toString());
  url.searchParams.append("audio_temperature", params.audioTemperature.toString());
  url.searchParams.append("audio_topk", params.audioTopk.toString());
  url.searchParams.append("pad_mult", params.padMult.toString());
  url.searchParams.append("text_seed", textSeed.toString());
  url.searchParams.append("audio_seed", audioSeed.toString());
  url.searchParams.append("repetition_penalty_context", params.repetitionPenaltyContext.toString());
  url.searchParams.append("repetition_penalty", params.repetitionPenalty.toString());
  if (sid) {
    url.searchParams.append("sid", sid);   // Space session (DESIGN.md 5.1); the relay adds the worker token itself
  }
  if (dep1) {
    // DEP1 (INTERFACE.md section 3): the server builds role prompt + voice from the record. Prompt and voice are
    // sent as EMPTY strings (not omitted) so the server-side session_config wins.
    url.searchParams.append("text_prompt", "");
    url.searchParams.append("voice_prompt", "");
    url.searchParams.append("seed", String(dep1.seed));
    url.searchParams.append("record_id", dep1.recordId);
    url.searchParams.append("pairing", dep1.pairing);
    url.searchParams.append("agent_type", dep1.agentType);
    url.searchParams.append("events", "1");
  } else {
    url.searchParams.append("text_prompt", params.textPrompt.toString());
    url.searchParams.append("voice_prompt", params.voicePrompt.toString());
    url.searchParams.append("events", "1");
  }
  console.log(url.toString());
  return url.toString();
};


export const Conversation:FC<ConversationProps> = ({
  workerAddr,
  workerAuthId,
  audioContext,
  worklet,
  sessionAuthId,
  sessionId,
  onConversationEnd,
  startConnection,
  isBypass=false,
  email,
  theme,
  dep1=null,
  sid=null,
  ...params
}) => {
  const getAudioStats = useRef<() => AudioStats>(() => ({
    playedAudioDuration: 0,
    missedAudioDuration: 0,
    totalAudioMessages: 0,
    delay: 0,
    minPlaybackDelay: 0,
    maxPlaybackDelay: 0,
  }));
  const isRecording = useRef<boolean>(false);
  const audioChunks = useRef<Blob[]>([]);

  const audioStreamDestination = useRef<MediaStreamAudioDestinationNode>(audioContext.current!.createMediaStreamDestination());
  const stereoMerger = useRef<ChannelMergerNode>(audioContext.current!.createChannelMerger(2));
  const audioRecorder = useRef<MediaRecorder>(new MediaRecorder(audioStreamDestination.current.stream, { mimeType: getMimeType("audio"), audioBitsPerSecond: 128000  }));
  const [audioURL, setAudioURL] = useState<string>("");
  const [isOver, setIsOver] = useState(false);
  const modelParams = useModelParams(params);
  const micDuration = useRef<number>(0);
  const actualAudioPlayed = useRef<number>(0);
  const textSeed = useMemo(() => Math.round(1000000 * Math.random()), []);
  const audioSeed = useMemo(() => Math.round(1000000 * Math.random()), []);

  const WSURL = buildURL({
    workerAddr,
    params: modelParams,
    workerAuthId,
    email: email,
    textSeed: textSeed,
    audioSeed: audioSeed,
    dep1,
    sid,
  });

  const onDisconnect = useCallback(() => {
    setIsOver(true);
    console.log("on disconnect!");
    stopRecording();
  }, [setIsOver]);

  const { socketStatus, sendMessage, socket, start, stop } = useSocket({
    // onMessage,
    uri: WSURL,
    onDisconnect,
  });
  useEffect(() => {
    audioRecorder.current.ondataavailable = (e) => {
      audioChunks.current.push(e.data);
    };
    audioRecorder.current.onstop = async () => {
      let blob: Blob;
      const mimeType = getMimeType("audio");
      if(mimeType.includes("webm")) {
        blob = await fixWebmDuration(new Blob(audioChunks.current, { type: mimeType }));
        } else {
          blob = new Blob(audioChunks.current, { type: mimeType });
      }
      setAudioURL(URL.createObjectURL(blob));
      audioChunks.current = [];
      console.log("Audio Recording and encoding finished");
    };
  }, [audioRecorder, setAudioURL, audioChunks]);


  useEffect(() => {
    start();
    return () => {
      stop();
    };
  }, [start, workerAuthId]);

  const startRecording = useCallback(() => {
    if(isRecording.current) {
      return;
    }
    console.log(Date.now() % 1000, "Starting recording");
    console.log("Starting recording");
    // Build stereo routing for recording: left = server (worklet), right = user mic (connected in useUserAudio)
    try {
      stereoMerger.current.disconnect();
    } catch {}
    try {
      worklet.current?.disconnect(audioStreamDestination.current);
    } catch {}
    // Route server audio (mono) to left channel of merger
    worklet.current?.connect(stereoMerger.current, 0, 0);
    // Connect merger to the MediaStream destination
    stereoMerger.current.connect(audioStreamDestination.current);

    setAudioURL("");
    audioRecorder.current.start();
    isRecording.current = true;
  }, [isRecording, worklet, audioStreamDestination, audioRecorder, stereoMerger]);

  const stopRecording = useCallback(() => {
    console.log("Stopping recording");
    console.log("isRecording", isRecording)
    if(!isRecording.current) {
      return;
    }
    try {
      worklet.current?.disconnect(stereoMerger.current);
    } catch {}
    try {
      stereoMerger.current.disconnect(audioStreamDestination.current);
    } catch {}
    audioRecorder.current.stop();
    isRecording.current = false;
  }, [isRecording, worklet, audioStreamDestination, audioRecorder, stereoMerger]);

  const onPressConnect = useCallback(async () => {
      if (isOver) {
        deleteSession(sid);
        window.location.reload();
      } else {
        audioContext.current?.resume();
        if (socketStatus !== "connected") {
          start();
        } else {
          stop();
          deleteSession(sid);   // the relay already ends the call when the socket closes; this also frees the slot
        }
      }
    }, [socketStatus, isOver, start, stop, sid]);

  const socketColor = useMemo(() => {
    if (socketStatus === "connected") {
      return 'bg-[#76b900]';
    } else if (socketStatus === "connecting") {
      return 'bg-orange-300';
    } else {
      return 'bg-red-400';
    }
  }, [socketStatus]);

  const ev = useDep1Events(socket);
  // turn-filler mode for the ticker indicator: the most recent of the worker's toggle confirmation, the 2 s metrics
  // and the last filler event (each carries the worker's t_wall)
  const fillMode = useMemo(() => {
    const c: [number, string][] = [];
    if (ev.turnFill?.mode) c.push([ev.turnFill.t_wall ?? 0, ev.turnFill.mode]);
    if (ev.metricsEvent?.turn_fill?.mode) c.push([ev.metricsEvent.t_wall ?? 0, ev.metricsEvent.turn_fill.mode]);
    if (ev.filler?.mode) c.push([ev.filler.t_wall ?? 0, ev.filler.mode]);
    c.sort((a, b) => a[0] - b[0]);
    return c.length ? c[c.length - 1][1] : null;
  }, [ev.turnFill, ev.metricsEvent, ev.filler]);
  const [startedAt] = useState(() => Date.now());
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  const phaseMsg = useMemo(() => {
    if (ev.sessionEnd) {
      const r = ev.sessionEnd.reason;
      const why = END_REASON_TEXT[r] ? `${r}: ${END_REASON_TEXT[r]}` : r;
      return `Session ended (${why})${ev.sessionEnd.frames !== null && ev.sessionEnd.frames !== undefined ? ` after ${ev.sessionEnd.frames} frames (${(ev.sessionEnd.frames / 12.5).toFixed(1)} s)` : ""}${ev.sessionEnd.detail ? ` · ${ev.sessionEnd.detail}` : ""}`;
    }
    if (isOver) {
      return "Disconnected.";
    }
    if (socketStatus === "connected") {
      return "Live: speak as the customer.";
    }
    if (socketStatus === "connecting") {
      return `Prompt phase: the GPU worker is loading the role and voice prompt (${((now - startedAt) / 1000).toFixed(0)} s)…`;
    }
    return "Connecting to the GPU worker…";
  }, [ev.sessionEnd, isOver, socketStatus, now, startedAt]);

  const socketButtonMsg = useMemo(() => {
    if (isOver) {
      return 'New Conversation';
    }
    if (socketStatus === "connected") {
      return 'Disconnect';
    } else {
      return 'Connecting...';
    }
  }, [isOver, socketStatus]);

  return (
    <SocketContext.Provider
      value={{
        socketStatus,
        sendMessage,
        socket,
      }}
    >
    <div className="min-h-screen w-full bg-neutral-50 p-3">
      <div className="max-w-screen-xl mx-auto flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            onClick={onPressConnect}
            disabled={socketStatus !== "connected" && !isOver}
            className="border"
          >
            {socketButtonMsg}
          </Button>
          <div className={`h-4 w-4 rounded-full ${socketColor}`} />
          <span data-testid="phase" className={`text-sm ${ev.sessionEnd ? "text-red-700 font-semibold" : "text-gray-700"}`}>{phaseMsg}</span>
          <span className="ml-auto text-xs text-gray-500">{dep1 ? `${dep1.recordId} · ${dep1.pairing} · ${dep1.agentType}` : "stock session"}</span>
        </div>
        {audioContext.current && worklet.current && <MediaContext.Provider value={
          {
            startRecording,
            stopRecording,
            audioContext: audioContext as MutableRefObject<AudioContext>,
            worklet: worklet as MutableRefObject<AudioWorkletNode>,
            audioStreamDestination,
            stereoMerger,
            micDuration,
            actualAudioPlayed,
          }
        }>
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
            <div className="flex flex-col gap-3 min-w-0">
              <section className="bg-white border border-gray-200 rounded-lg p-2">
                <div className="flex justify-around items-center">
                  <div className="h-40 w-40 flex items-center justify-center">
                    <ServerAudio
                      setGetAudioStats={(callback: () => AudioStats) =>
                        (getAudioStats.current = callback)
                      }
                      theme={theme}
                    />
                  </div>
                  <div className="h-40 w-40 flex items-center justify-center">
                    <UserAudio theme={theme}/>
                  </div>
                  <div className="h-40 w-40 flex items-center justify-center">
                    <TickerIndicator filler={ev.filler} fillerRecvAt={ev.fillerRecvAt} mode={fillMode} theme={theme}/>
                  </div>
                </div>
                <div className="flex justify-around text-[11px] text-gray-500"><span>agent</span><span>you (customer)</span><span title="The turn filler: a soft 1 s ticker fed to the model's input when the agent pauses, so it yields the turn. You never hear it.">ticker (model only){ev.nFills ? ` · ${ev.nFills}×` : ""}</span></div>
                <div className="text-sm flex justify-center items-center flex-col">
                  {audioURL && <a href={audioURL} download={`dep1_${dep1?.recordId ?? "stock"}_audio.${getExtension("audio")}`} className="pt-1 text-center block text-blue-700 underline">Download stereo recording</a>}
                </div>
                <div className="hidden md:block">
                  <ServerAudioStats getAudioStats={getAudioStats} />
                </div>
              </section>
              <MetricsPanel metricsEvent={ev.metricsEvent} />
              <SessionInfo session={ev.session} fallback={dep1?.preview} previewPrompt={dep1?.preview?.role_prompt ?? (dep1 ? undefined : params.textPrompt)} />
            </div>
            <div className="flex flex-col gap-3 min-w-0">
              {/* D-SCRIPT-PANEL: the expected conversation sits to the right of the live text (stacked when narrow) */}
              <div className="grid gap-3 lg:grid-cols-2 min-w-0">
                <TextStreamPanel state={ev} className="min-w-0" />
                <ScriptPanel recordId={dep1?.recordId} pairing={dep1?.pairing} texts={ev.texts} className="min-w-0" />
              </div>
              <ActionsPanel actions={ev.actions} />
            </div>
          </div>
          </MediaContext.Provider>}
        <div className="grid md:grid-cols-2 gap-3">
          <SamplesPanel />
          <div className="flex flex-col gap-2">
            <Disclaimer />
            <div className="text-xs text-gray-500"><ServerInfo/></div>
            <p className="text-[11px] text-gray-400">events received: {ev.nEvents} · unknown messages: {ev.nUnknown}</p>
          </div>
        </div>
      </div>
    </div>
    </SocketContext.Provider>
  );
};

        // </MediaContext.Provider> : undefined}
        // 
        // }></MediaContext.Provider>
