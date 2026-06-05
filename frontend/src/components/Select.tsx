import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

export type SelectOption = { value: string; label: string; hint?: string; disabled?: boolean };

export function Select({
  value,
  options,
  onChange,
  placeholder = "select...",
  className = "",
}: {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const selected = options.find((o) => o.value === value);
  const searchable = options.length > 6;
  const filteredOptions = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => `${o.label} ${o.hint ?? ""}`.toLowerCase().includes(q));
  }, [options, query]);

  useEffect(() => {
    if (!open) return;
    setActive(filteredOptions.findIndex((o) => o.value === value));
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [filteredOptions, open, value]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }
    if (searchable) requestAnimationFrame(() => searchRef.current?.focus());
  }, [open, searchable]);

  const choose = (i: number) => {
    const opt = filteredOptions[i];
    if (!opt || opt.disabled) return;
    onChange(opt.value);
    setOpen(false);
  };

  const step = (dir: 1 | -1) => {
    setActive((cur) => {
      let i = cur;
      for (let n = 0; n < filteredOptions.length; n++) {
        i = (i + dir + filteredOptions.length) % filteredOptions.length;
        if (!filteredOptions[i]?.disabled) return i;
      }
      return cur;
    });
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) setOpen(true);
      else step(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (open) step(-1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (open) choose(active);
      else setOpen(true);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  const onSearchKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      step(1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      step(-1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(active);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
    }
  };

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKeyDown}
        className="flex w-full items-center justify-between gap-2 rounded-md border border-white/10 bg-black/30 px-2.5 py-1.5 text-left text-sm outline-none transition focus:border-violet-500 hover:border-white/20"
      >
        <span className={`min-w-0 truncate ${selected ? "" : "text-white/40"}`}>
          {selected ? selected.label : placeholder}
        </span>
        <svg
          className={`h-3.5 w-3.5 shrink-0 text-white/40 transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
        >
          <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-30 mt-1 w-full overflow-hidden rounded-md border border-white/10 bg-surface-2 shadow-xl shadow-black/60">
          {searchable ? (
            <div className="border-b border-white/10 p-1.5">
              <input
                ref={searchRef}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setActive(0);
                }}
                onKeyDown={onSearchKeyDown}
                placeholder="search..."
                className="w-full rounded border border-white/10 bg-black/30 px-2 py-1 text-xs outline-none transition placeholder:text-white/25 focus:border-violet-500"
              />
            </div>
          ) : null}
          <div className="max-h-56 overflow-y-auto py-1">
            {filteredOptions.length === 0 ? <div className="px-2.5 py-1.5 text-sm text-white/30">no options</div> : null}
            {filteredOptions.map((o, i) => (
              <button
                key={o.value || `opt-${i}`}
                type="button"
                disabled={o.disabled}
                onClick={() => choose(i)}
                onMouseEnter={() => setActive(i)}
                className={`flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left text-sm disabled:cursor-not-allowed disabled:opacity-30 ${
                  o.value === value
                    ? "bg-violet-600/30 text-white"
                    : i === active
                      ? "bg-white/10 text-white/90"
                      : "text-white/80"
                }`}
              >
                <span className="min-w-0 truncate">{o.label}</span>
                {o.hint ? <span className="shrink-0 text-[11px] text-white/40">{o.hint}</span> : null}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
