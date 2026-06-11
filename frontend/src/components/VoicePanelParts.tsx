import type { ReactNode } from "react";
import { Badge } from "./Badge";
import { Select } from "./Select";
import { deviceHint, deviceName, formatBytes, formatMs } from "./voiceHelpers";
import type { VoiceAudioDevice, VoiceModel } from "../types";

export type RoutingApplyState = "idle" | "pending" | "applying" | "applied" | "error";

export function SetupStep({ step, title, aside, children }: { step: string; title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-white/10 bg-surface p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="grid h-5 w-5 place-items-center rounded-full border border-white/15 bg-black/25 text-[10px] font-semibold text-white/55">
            {step}
          </span>
          <div className="text-xs font-medium uppercase tracking-wide text-white/40">{title}</div>
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function RoutingApplyHint({ canReach, state }: { canReach: boolean; state: RoutingApplyState }) {
  if (!canReach) return <span className="text-amber-200/70">Start the engine to list devices</span>;
  if (state === "pending") return <span className="text-white/45">pending...</span>;
  if (state === "applying") return <span className="text-sky-200/75">applying...</span>;
  if (state === "applied") return <span className="text-emerald-200/75">applied</span>;
  if (state === "error") return <span className="text-red-200/75">not applied</span>;
  return <span className="text-white/35">auto apply</span>;
}

export function DeviceSelect({
  label,
  value,
  devices,
  fallback,
  onChange,
}: {
  label: string;
  value: number;
  devices: VoiceAudioDevice[];
  fallback: string;
  onChange: (value: number) => void;
}) {
  const name = deviceName(devices, value, fallback);
  return (
    <label className="min-w-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs uppercase tracking-wide text-white/40">{label}</span>
      </div>
      <Select
        value={String(value)}
        onChange={(v) => onChange(Number(v))}
        placeholder={fallback}
        className="mt-1"
        options={devices.map((d) => ({
          value: d.id,
          label: d.name,
          hint: deviceHint(d.host_api, d.default_sample_rate),
        }))}
      />
      <div className="mt-1 truncate text-[11px] text-white/35" title={name}>{name}</div>
    </label>
  );
}

export function MonitorSelect({ value, devices, onChange }: { value: number; devices: VoiceAudioDevice[]; onChange: (value: number) => void }) {
  const name = deviceName(devices, value, "Off");
  return (
    <label className="min-w-0">
      <div className="text-xs uppercase tracking-wide text-white/40">Monitor</div>
      <Select
        value={String(value)}
        onChange={(v) => onChange(Number(v))}
        className="mt-1"
        options={[
          { value: "-1", label: "Off" },
          ...devices.map((d) => ({
            value: d.id,
            label: d.name,
            hint: deviceHint(d.host_api, d.default_sample_rate),
          })),
        ]}
      />
      <div className="mt-1 truncate text-[11px] text-white/35" title={name}>{name}</div>
    </label>
  );
}

export function OfflineDevice({ label }: { label: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="text-xs uppercase tracking-wide text-white/40">{label}</div>
      <div className="mt-2 rounded border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-sm text-amber-100/75">
        Start the engine to list devices
      </div>
    </div>
  );
}

export function LatencyMeter({ value }: { value: number | null | undefined }) {
  const n = Number(value ?? 0);
  return (
    <div className="rounded-md border border-white/10 bg-black/20 px-3 py-2">
      <div className="flex items-center justify-between text-xs">
        <span className="uppercase tracking-wide text-white/40">Latency</span>
        <span className="font-mono text-white/65">{formatMs(value)}</span>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-white/10">
        <div className="h-full rounded-full bg-accent/80" style={{ width: `${Math.min(100, Math.max(6, n))}%` }} />
      </div>
    </div>
  );
}

export function VoiceSlotList({
  models,
  modelId,
  modelDir,
  onSelect,
}: {
  models: VoiceModel[];
  modelId: string;
  modelDir?: string;
  onSelect: (modelId: string) => void;
}) {
  if (models.length === 0) {
    return (
      <p className="mt-3 text-sm leading-6 text-white/40">
        No voice slots found in <code className="text-white/60">{modelDir ?? "model_dir"}</code>.
      </p>
    );
  }

  return (
    <ul className="mt-3 flex max-h-56 flex-col gap-1.5 overflow-y-auto pr-1">
      {models.map((m) => (
        <li
          key={m.id}
          onClick={() => onSelect(m.id)}
          className={`flex cursor-pointer items-center justify-between gap-2 rounded-md border px-3 py-2 transition ${
            m.id === modelId ? "border-accent/40 bg-accent/10" : "border-white/10 bg-black/20 hover:bg-white/5"
          }`}
        >
          <span className="flex min-w-0 items-center gap-2">
            <span className="text-[10px] text-white/30">#{m.slot}</span>
            <span className="min-w-0 truncate text-sm text-white/80" title={m.name}>{m.name}</span>
          </span>
          <span className="flex shrink-0 items-center gap-1.5">
            <Badge color="bg-accent/50 text-accent-fg">{m.type}{m.version ? ` ${m.version}` : ""}</Badge>
            {m.f0 ? <Badge color="bg-sky-700/50 text-sky-100">f0</Badge> : null}
            {m.has_index ? <Badge color="bg-emerald-700/55 text-emerald-100">index</Badge> : <Badge>no index</Badge>}
            <span className="font-mono text-xs text-white/35">{formatBytes(m.size_bytes)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

export function Row({ label, value, ok, mono = false }: { label: string; value: string; ok?: boolean; mono?: boolean }) {
  return (
    <div className="grid min-w-0 grid-cols-[82px_1fr] gap-2">
      <span className="shrink-0 text-white/40">{label}</span>
      <span
        className={`min-w-0 truncate ${mono ? "font-mono text-xs" : ""} ${
          ok === undefined ? "text-white/70" : ok ? "text-emerald-300/80" : "text-amber-300/70"
        }`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}
