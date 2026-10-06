// Turn-filler ("ticker") indicator, next to the agent and customer visuals (user 2026-10-06: "i want to see the ticker
// playing"). The ticker is fed to the MODEL input only (the customer never hears it), so there is no audio to analyse:
// the indicator is driven by the worker's real 0x07 "filler" transition events (start / stop / cancel / skip_check,
// see dep1/types.ts FillerEvent). While a fill runs it pulses like the ticker itself (a click every 200 ms, 5 per s);
// idle = a small grey dot; turn filler Off = greyed out with "off". Same canvas layout as ServerVisualizer.
import { FC, RefObject, useEffect, useRef, useState } from "react";
import { type ThemeType } from "../../hooks/useSystemTheme";
import { FillerEvent } from "../../../../dep1/types";

type Props = {
  filler: FillerEvent | null;
  fillerRecvAt: number | null;   // performance.now() at arrival
  mode: string | null;           // "ticker" | "off" | other demo modes | null (unknown yet)
  parent: RefObject<HTMLElement>;
  theme: ThemeType;
};

const CLICK_MS = 200;     // worker core.py: ticker clicks every 200 ms
const CLICK_TAU_MS = 45;
const GUARD_MS = 1500;    // a start with no stop/cancel after seconds + this is treated as ended (lost event)

export type TickerStatus = "filling" | "idle" | "off" | "skipped" | "cancelled";

export const tickerStatus = (filler: FillerEvent | null, recvAt: number | null, mode: string | null, now: number): TickerStatus => {
  if (mode === "off") return "off";
  if (!filler || recvAt === null) return "idle";
  const age = now - recvAt;
  if (filler.state === "start" && age < (filler.seconds ?? 1) * 1000 + GUARD_MS) return "filling";
  if (filler.state === "skip_check" && age < 1500) return "skipped";
  if (filler.state === "cancel" && age < 1500) return "cancelled";
  return "idle";
};

export const STATUS_TEXT: Record<TickerStatus, string> = {
  filling: "feeding ticker to the model",
  idle: "idle",
  off: "off",
  skipped: "skipped (check-line)",
  cancelled: "stopped: you spoke",
};

export const TickerVisualizer: FC<Props> = ({ filler, fillerRecvAt, mode, parent, theme }) => {
  const [canvasWidth, setCanvasWidth] = useState(parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ filler, fillerRecvAt, mode, theme, canvasWidth });
  live.current = { filler, fillerRecvAt, mode, theme, canvasWidth };

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const L = live.current;
      const width = parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0;
      if (width !== L.canvasWidth) setCanvasWidth(width);
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx || width <= 0) return;
      const now = performance.now();
      const st = tickerStatus(L.filler, L.fillerRecvAt, L.mode, now);
      const c = width / 2;
      const maxR = Math.floor(width * 0.95) / 2;
      ctx.clearRect(0, 0, width, width);
      ctx.fillStyle = L.theme === "dark" ? "#000000" : "#fafafa";
      ctx.fillRect(0, 0, width, width);
      const off = st === "off";
      if (st === "filling" && L.fillerRecvAt !== null) {
        const t = now - L.fillerRecvAt;
        const env = Math.exp(-(t % CLICK_MS) / CLICK_TAU_MS);           // decaying click, like the real buffer
        ctx.beginPath();
        ctx.fillStyle = "#B45309";
        ctx.arc(c, c, maxR * (0.3 + 0.65 * env), 0, 2 * Math.PI);
        ctx.fill();
        ctx.closePath();
        ctx.beginPath();
        ctx.fillStyle = "#F59E0B";
        ctx.arc(c, c, maxR / 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.closePath();
      } else {
        ctx.beginPath();
        ctx.fillStyle = off ? "#D1D5DB" : st === "skipped" ? "#93C5FD" : st === "cancelled" ? "#FCD34D" : "#9CA3AF";
        ctx.arc(c, c, maxR / (off ? 3 : 4), 0, 2 * Math.PI);
        ctx.fill();
        ctx.closePath();
      }
      ctx.beginPath();
      ctx.arc(c, c, maxR, 0, 2 * Math.PI);
      ctx.strokeStyle = off ? "#D1D5DB" : (L.theme === "dark" ? "white" : "black");
      ctx.lineWidth = width / 50;
      ctx.stroke();
      ctx.closePath();
      if (off) {
        ctx.fillStyle = "#6B7280";
        ctx.font = `${Math.max(10, Math.round(width / 7))}px system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("off", c, c);
      }
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [parent]);

  return <canvas data-testid="ticker-canvas" className="max-h-full max-w-full" ref={canvasRef} width={canvasWidth} height={canvasWidth} />;
};

// The box that sits next to ServerAudio / UserAudio (same h-40 w-40 slot).
export const TickerIndicator: FC<Omit<Props, "parent">> = props => {
  const ref = useRef<HTMLDivElement>(null);
  const [now, setNow] = useState(performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(id);
  }, []);
  const st = tickerStatus(props.filler, props.fillerRecvAt, props.mode, now);
  return (
    <div className="flex flex-col items-center justify-center h-full w-full" data-testid="ticker-indicator" data-state={st}>
      <div className="ticker-audio h-4/6 aspect-square" ref={ref}>
        <TickerVisualizer {...props} parent={ref} />
      </div>
      <span className={`text-[10px] mt-1 ${st === "filling" ? "text-amber-700 font-medium" : "text-gray-400"}`}>{STATUS_TEXT[st]}</span>
    </div>
  );
};
