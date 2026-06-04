import { useEffect, useState } from "react";

// Tiny module-level toast store so any code can fire a toast without prop
// drilling; <ToastHost/> (mounted once in App) subscribes and renders them.
export type ToastKind = "info" | "success" | "error";
export type ToastOpts = { duration?: number; onClick?: () => void };
type ToastItem = { id: number; kind: ToastKind; msg: string; onClick?: () => void };
type Listener = (items: ToastItem[]) => void;

let items: ToastItem[] = [];
const listeners = new Set<Listener>();
let nextId = 1;

function emit() {
  for (const l of listeners) l(items);
}

function dismiss(id: number) {
  items = items.filter((t) => t.id !== id);
  emit();
}

function show(msg: string, kind: ToastKind, opts?: ToastOpts): number {
  const id = nextId++;
  items = [...items, { id, kind, msg, onClick: opts?.onClick }];
  emit();
  const duration = opts?.duration ?? 4500;
  if (duration > 0) setTimeout(() => dismiss(id), duration);
  return id;
}

export const toast = {
  info: (msg: string, opts?: ToastOpts) => show(msg, "info", opts),
  success: (msg: string, opts?: ToastOpts) => show(msg, "success", opts),
  error: (msg: string, opts?: ToastOpts) => show(msg, "error", opts),
  dismiss,
};

const kindStyle: Record<ToastKind, string> = {
  info: "border-white/15 bg-surface-2",
  success: "border-emerald-500/30 bg-emerald-500/10",
  error: "border-red-500/30 bg-red-500/10",
};

export function ToastHost() {
  const [list, setList] = useState<ToastItem[]>(items);
  useEffect(() => {
    const l: Listener = (next) => setList(next);
    listeners.add(l);
    setList(items);
    return () => {
      listeners.delete(l);
    };
  }, []);

  if (!list.length) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-40 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      {list.map((t) => (
        <div
          key={t.id}
          onClick={() => {
            t.onClick?.();
            dismiss(t.id);
          }}
          className={`pointer-events-auto flex animate-fade-in items-start gap-2 rounded-lg border px-3 py-2 text-sm text-white/85 shadow-lg shadow-black/40 transition ${kindStyle[t.kind]} ${t.onClick ? "cursor-pointer hover:brightness-125" : ""}`}
        >
          <span className="min-w-0 flex-1 break-words">{t.msg}</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              dismiss(t.id);
            }}
            className="shrink-0 text-white/40 transition hover:text-white/80"
            aria-label="dismiss"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
