import { useMemo } from "react";

import type { Model } from "../types";
import { Badge } from "./Badge";
import { familyColor, formatVram, isModelAvailable, isNunchaku } from "./imageComposerHelpers";
import { Select } from "./Select";

// A compact model selector (P13.1): the always-expanded card grid ate too much
// vertical space, so we collapse it into the shared Select but keep the card look
// in each option — name + measured VRAM + all badges on a single row.
export function ModelPicker({
  models,
  value,
  onChange,
  placeholder = "select a model",
}: {
  models: Model[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const byId = useMemo(() => new Map(models.map((m) => [m.id, m])), [models]);
  const options = useMemo(
    () => models.map((m) => ({
      value: m.id,
      label: m.name,
      hint: m.unavailable_reason ?? formatVram(m),
      disabled: !isModelAvailable(m),
    })),
    [models],
  );

  return (
    <Select
      value={value}
      options={options}
      onChange={onChange}
      placeholder={placeholder}
      renderOption={(o) => {
        const m = byId.get(o.value);
        if (!m) return <span className="truncate">{o.label}</span>;
        const vram = formatVram(m);
        return (
          <span className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="flex items-center gap-2">
              <span className="min-w-0 flex-1 truncate font-medium" title={m.name}>{m.name}</span>
              {vram ? <span className="shrink-0 font-mono text-[11px] text-white/55">{vram}</span> : null}
            </span>
            <span className="flex items-center gap-1 overflow-hidden">
              <Badge color={familyColor(m.family)}>{m.family}</Badge>
              {m.quant ? <Badge>{m.quant}</Badge> : null}
              {isNunchaku(m) ? <Badge color="bg-emerald-700/55 text-emerald-100">fast</Badge> : null}
              {m.recommendation === "recommended" ? (
                <Badge color="bg-emerald-600/45 text-emerald-100">recommended</Badge>
              ) : null}
              {m.slow ? <Badge color="bg-amber-600/35 text-amber-100">slow</Badge> : null}
              {m.runtime_mode === "stub" ? <Badge color="bg-sky-700/50 text-sky-100">stub</Badge> : null}
              {!isModelAvailable(m) ? <Badge color="bg-red-700/50 text-red-100">disabled</Badge> : null}
              {m.loaded ? (
                <Badge color="bg-emerald-700/55 text-emerald-100">loaded</Badge>
              ) : m.warm ? (
                <Badge color="bg-sky-700/50 text-sky-100">warm</Badge>
              ) : null}
            </span>
            {m.unavailable_reason ? (
              <span className="truncate text-[11px] text-red-200/75" title={m.unavailable_reason}>
                {m.unavailable_reason}
              </span>
            ) : null}
          </span>
        );
      }}
    />
  );
}
