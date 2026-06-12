import type { VoiceEngineStatus } from "../types";
import { formatMs, meter, waveformSlots } from "./voiceHelpers";

export type MeterSample = {
  input: number;
  output: number;
};

export function Meter({ label, value, tone = "emerald" }: { label: string; value: number; tone?: "emerald" | "sky" }) {
  const bar = tone === "sky" ? "bg-sky-400/80" : "bg-emerald-400/80";
  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between text-xs">
        <span className="uppercase tracking-wide text-white/40">{label}</span>
        <span className="font-mono text-white/65">{value}%</span>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-white/10">
        <div className={`h-full rounded-full transition-[width] ${bar}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

export function WaveformMonitor({ samples }: { samples: MeterSample[] }) {
  const bars = Array.from({ length: waveformSlots }, (_, index) => {
    const offset = samples.length - waveformSlots + index;
    return offset >= 0 ? samples[offset] : { input: 0, output: 0 };
  });
  const latest = samples[samples.length - 1] ?? { input: 0, output: 0 };

  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="uppercase tracking-wide text-white/40">Waveform</span>
        <span className="font-mono text-white/55">
          in {meter(latest.input)}% / out {meter(latest.output)}%
        </span>
      </div>
      <div className="mt-3 flex h-24 items-center gap-px overflow-hidden rounded bg-black/25 px-2 py-2">
        {bars.map((sample, index) => {
          const inputHeight = sample.input > 0 ? Math.max(2, sample.input * 46) : 0;
          const outputHeight = sample.output > 0 ? Math.max(2, sample.output * 46) : 0;
          return (
            <div key={index} className="relative h-full min-w-0 flex-1">
              <div className="absolute left-0 right-0 top-1/2 h-px bg-white/10" />
              <div
                className="absolute bottom-1/2 left-0 right-0 rounded-t-sm bg-emerald-400/80"
                style={{ height: `${inputHeight}%` }}
              />
              <div
                className="absolute left-0 right-0 top-1/2 rounded-b-sm bg-sky-400/75"
                style={{ height: `${outputHeight}%` }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

type VoiceMetrics = VoiceEngineStatus["metrics"];

function timingEntries(metrics?: VoiceMetrics): { label: string; value: number }[] {
  const raw = metrics?.timings_ms;
  return Object.entries(raw ?? {})
    .filter(([, value]) => Number.isFinite(value))
    .slice(0, 6)
    .map(([label, value]) => ({ label, value: Number(value) }));
}

export function PerformanceBreakdown({ metrics }: { metrics?: VoiceMetrics }) {
  const timings = timingEntries(metrics);
  const total = metrics?.total_ms ?? null;
  const chunk = metrics?.chunk_ms ?? null;
  const max = Math.max(1, Number(total ?? 0), Number(chunk ?? 0), ...timings.map((entry) => entry.value));
  const overruns = Number(metrics?.overruns ?? 0);
  const underruns = Number(metrics?.underruns ?? 0);

  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="uppercase tracking-wide text-white/40">Timing</span>
        <span className="font-mono text-white/55">{formatMs(total ?? chunk)}</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <MetricPill label="chunk" value={formatMs(chunk)} />
        <MetricPill label="total" value={formatMs(total)} />
        <MetricPill label="overruns" value={String(overruns)} />
        <MetricPill label="underruns" value={String(underruns)} />
      </div>
      <div className="mt-3 flex flex-col gap-2">
        {timings.length === 0 ? (
          <div className="rounded border border-white/10 bg-white/[0.03] px-2 py-1.5 text-xs text-white/35">waiting for stages</div>
        ) : (
          timings.map(({ label, value }) => (
            <div key={`${label}-${value}`} className="min-w-0">
              <div className="flex items-center justify-between gap-3 text-[11px]">
                <span className="truncate uppercase tracking-wide text-white/35">{label}</span>
                <span className="shrink-0 font-mono text-white/55">{formatMs(value)}</span>
              </div>
              <div className="mt-1 h-1 rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-accent/75"
                  style={{ width: `${Math.min(100, Math.max(4, (value / max) * 100))}%` }}
                />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded border border-white/10 bg-white/[0.03] px-2 py-1.5">
      <div className="truncate text-[10px] uppercase tracking-wide text-white/30">{label}</div>
      <div className="truncate font-mono text-xs text-white/65">{value}</div>
    </div>
  );
}
