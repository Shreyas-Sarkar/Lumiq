"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

interface CodeBlockProps {
  code: string;
}

export default function CodeBlock({ code }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  if (!code) return null;

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700 my-3">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <span className="text-xs text-gray-400 font-medium">Generated Code</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors"
        >
          {copied ? (
            <>
              <Check size={12} />
              Copied!
            </>
          ) : (
            <>
              <Copy size={12} />
              Copy
            </>
          )}
        </button>
      </div>
      <div className="max-h-96 overflow-y-auto">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            padding: "1rem",
            background: "transparent",
            fontSize: "0.8rem",
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
