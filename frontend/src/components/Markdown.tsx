import { isValidElement, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

// Lightweight typographic styling via component overrides so we don't pull in a
// full prose plugin. Code blocks get syntax highlighting + a copy button.
export function Markdown({ content }: { content: string }) {
  return (
    <div className="text-sm leading-relaxed text-white/90">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
          h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-semibold">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-2 mt-3 text-base font-semibold">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-1 mt-3 text-sm font-semibold">{children}</h3>,
          ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="marker:text-white/40">{children}</li>,
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer" className="text-sky-400 underline hover:text-sky-300">
              {children}
            </a>
          ),
          img: ({ src, alt }) => (
            <img src={src as string} alt={(alt as string) ?? ""} loading="lazy" className="my-2 max-h-[28rem] w-auto rounded-md border border-white/10" />
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-2 border-l-2 border-white/20 pl-3 text-white/70">{children}</blockquote>
          ),
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="border border-white/15 px-2 py-1 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="border border-white/10 px-2 py-1">{children}</td>,
          pre: PreBlock,
          code: InlineOrBlockCode,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function InlineOrBlockCode({ className, children }: { className?: string; children?: ReactNode }) {
  const text = textFromChildren(children);
  // Block code carries a language-* class (added by rehype-highlight) and is
  // wrapped by <pre>; leave it for the hljs theme. Inline code we style here.
  if (className?.includes("language-") || text.includes("\n")) {
    return <code className={className}>{children}</code>;
  }
  return <CopyableInlineCode>{children}</CopyableInlineCode>;
}

function CopyableInlineCode({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = textFromChildren(children);

  const copy = () => {
    if (!text.trim() || hasActiveSelection()) return;
    navigator.clipboard?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      })
      .catch(() => {});
  };

  return (
    <code
      onClick={copy}
      title={copied ? "copied" : "click to copy"}
      className={`cursor-copy rounded px-1 py-0.5 text-[0.85em] transition ${
        copied ? "bg-emerald-500/20 text-emerald-100" : "bg-white/10 hover:bg-white/15"
      }`}
    >
      {children}
    </code>
  );
}

function PreBlock({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  const copy = (force = false) => {
    if (!force && hasActiveSelection()) return;
    const text = ref.current?.innerText ?? "";
    navigator.clipboard?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };

  return (
    <div className="group relative my-2">
      <button
        onClick={() => copy(true)}
        className={`absolute right-2 top-2 rounded border border-white/15 bg-black/50 px-1.5 py-0.5 text-[10px] text-white/60 transition hover:bg-white/10 ${
          copied ? "opacity-100" : "opacity-0 group-hover:opacity-100"
        }`}
      >
        {copied ? "copied" : "copy"}
      </button>
      <pre
        ref={ref}
        onClick={() => copy(false)}
        title="click to copy"
        className="cursor-copy overflow-x-auto rounded-md border border-white/10 bg-black/50 p-3 text-xs leading-relaxed"
      >
        {children}
      </pre>
    </div>
  );
}

function hasActiveSelection(): boolean {
  const selection = window.getSelection?.();
  return Boolean(selection && !selection.isCollapsed && selection.toString());
}

function textFromChildren(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textFromChildren).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return textFromChildren(node.props.children);
  return "";
}
