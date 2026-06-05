import type { ReactNode } from "react";

// Shared pill badge (P5.D3 control kit). Pass `color` for type/family accents.
export function Badge({ children, color, className = "" }: { children: ReactNode; color?: string; className?: string }) {
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${color ?? "bg-white/10 text-white/70"} ${className}`}>
      {children}
    </span>
  );
}
