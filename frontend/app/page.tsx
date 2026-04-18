"use client";

import Link from "next/link";
import { ArrowRight, Terminal, BarChart3, Lightbulb, ShieldCheck, Zap, Database } from "lucide-react";

// ─── Animated terminal demo ───────────────────────────────────────────────────
function HeroDemo() {
  return (
    <div
      className="rounded-xl overflow-hidden border"
      style={{
        borderColor: "var(--border-strong)",
        background: "var(--surface)",
      }}
    >
      {/* Window chrome */}
      <div
        className="flex items-center gap-2 px-4 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}
      >
        <span className="w-3 h-3 rounded-full bg-red-500/70" />
        <span className="w-3 h-3 rounded-full bg-yellow-500/70" />
        <span className="w-3 h-3 rounded-full bg-green-500/70" />
        <span className="ml-3 text-xs" style={{ color: "var(--muted)" }}>
          Lumiq — Analysis Session
        </span>
      </div>

      <div className="p-6 space-y-5">
        {/* Query */}
        <div className="animate-fade-slide-in" style={{ animationDelay: "0.1s", opacity: 0 }}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: "var(--muted)" }}>
              Query
            </span>
          </div>
          <div
            className="rounded-lg px-4 py-3 text-sm"
            style={{ background: "var(--surface-3)", color: "var(--text)", border: "1px solid var(--border)" }}
          >
            What's the revenue trend by product category this quarter?
          </div>
        </div>

        {/* Arrow */}
        <div
          className="flex items-center gap-2 animate-fade-slide-in"
          style={{ animationDelay: "0.3s", opacity: 0 }}
        >
          <div className="flex-1 h-px" style={{ background: "var(--border-strong)" }} />
          <span className="text-xs font-mono flex items-center gap-1.5" style={{ color: "var(--muted)" }}>
            <Terminal size={11} /> Code executed · 42ms
          </span>
          <div className="flex-1 h-px" style={{ background: "var(--border-strong)" }} />
        </div>

        {/* Code */}
        <div className="animate-fade-slide-in" style={{ animationDelay: "0.5s", opacity: 0 }}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium uppercase tracking-widest" style={{ color: "var(--muted)" }}>
              Generated code
            </span>
          </div>
          <div
            className="rounded-lg p-4 text-xs font-mono leading-relaxed overflow-hidden"
            style={{
              background: "#0d0d0d",
              border: "1px solid var(--border)",
              color: "#a8b1c2",
            }}
          >
            <span style={{ color: "#61afef" }}>result</span>
            {" = df.groupby("}
            <span style={{ color: "#98c379" }}>'category'</span>
            {")["}<span style={{ color: "#98c379" }}>'revenue'</span>
            {"].sum().sort_values(ascending="}
            <span style={{ color: "#e5c07b" }}>False</span>
            {")"}
            <br />
            <span style={{ color: "#61afef" }}>result</span>
            {".plot(kind="}
            <span style={{ color: "#98c379" }}>'bar'</span>
            {", title="}
            <span style={{ color: "#98c379" }}>'Revenue by Category'</span>
            {")"}
          </div>
        </div>

        {/* Output row */}
        <div
          className="grid grid-cols-2 gap-3 animate-fade-slide-in"
          style={{ animationDelay: "0.7s", opacity: 0 }}
        >
          {/* Chart */}
          <div
            className="rounded-lg p-4 flex flex-col items-center justify-center gap-3"
            style={{ border: "1px solid var(--border)", background: "var(--surface-3)" }}
          >
            <BarChart3 size={28} style={{ color: "var(--accent)" }} />
            <div className="w-full space-y-1.5">
              {[
                { label: "Electronics", w: "85%" },
                { label: "Clothing", w: "62%" },
                { label: "Home & Garden", w: "45%" },
              ].map((r) => (
                <div key={r.label} className="flex items-center gap-2">
                  <span className="text-[10px] w-20 truncate" style={{ color: "var(--muted)" }}>
                    {r.label}
                  </span>
                  <div className="flex-1 h-1.5 rounded-full" style={{ background: "var(--border-strong)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: r.w, background: "var(--accent)" }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[10px]" style={{ color: "var(--muted)" }}>
              Generated chart
            </p>
          </div>

          {/* Insight */}
          <div
            className="rounded-lg p-4 space-y-2"
            style={{ border: "1px solid var(--border)", background: "var(--surface-3)" }}
          >
            <div className="flex items-center gap-1.5">
              <Lightbulb size={12} style={{ color: "var(--accent)" }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
                Insight
              </span>
            </div>
            <p className="text-xs leading-relaxed" style={{ color: "var(--text)" }}>
              Electronics leads with 85% relative share. Clothing growth is
              flat. Home & Garden is underperforming vs. prior quarter.
            </p>
            <div className="flex items-center gap-1 mt-2">
              <ShieldCheck size={10} style={{ color: "#22c55e" }} />
              <span className="text-[10px]" style={{ color: "#22c55e" }}>
                High confidence · grounded in execution
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Feature item ─────────────────────────────────────────────────────────────
function FeatureItem({
  icon: Icon,
  title,
  desc,
}: {
  icon: React.ElementType;
  title: string;
  desc: string;
}) {
  return (
    <div
      className="rounded-xl p-6 transition-all duration-200 group"
      style={{
        border: "1px solid var(--border)",
        background: "var(--surface)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border-strong)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border)";
      }}
    >
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center mb-4"
        style={{ background: "var(--surface-3)", border: "1px solid var(--border-strong)" }}
      >
        <Icon size={17} style={{ color: "var(--accent)" }} />
      </div>
      <h3 className="font-semibold text-sm mb-2" style={{ color: "var(--text)" }}>
        {title}
      </h3>
      <p className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
        {desc}
      </p>
    </div>
  );
}

// ─── Step flow ────────────────────────────────────────────────────────────────
function FlowStep({
  num,
  title,
  desc,
  last,
}: {
  num: string;
  title: string;
  desc: string;
  last?: boolean;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono font-semibold flex-shrink-0"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
          }}
        >
          {num}
        </div>
        {!last && (
          <div
            className="flex-1 w-px mt-3"
            style={{ background: "var(--border-strong)", minHeight: "40px" }}
          />
        )}
      </div>
      <div className="pb-8">
        <h4 className="font-semibold text-sm mb-1" style={{ color: "var(--text)" }}>
          {title}
        </h4>
        <p className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
          {desc}
        </p>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function LandingPage() {
  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      {/* ── Nav ── */}
      <nav
        className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
        style={{
          background: "rgba(11, 11, 12, 0.85)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-6 h-6 rounded flex items-center justify-center text-xs font-bold"
            style={{ background: "var(--surface-3)", border: "1px solid var(--border-strong)", color: "var(--text)" }}
          >
            L
          </div>
          <span className="font-semibold text-sm" style={{ color: "var(--text)" }}>
            Lumiq
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="text-sm transition-colors px-3 py-1.5 rounded-lg"
            style={{ color: "var(--muted)" }}
          >
            Sign in
          </Link>
          <Link
            href="/chat"
            className="text-sm font-medium flex items-center gap-1.5 px-4 py-1.5 rounded-lg transition-all duration-150"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border-strong)",
              color: "var(--text)",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.background = "var(--surface-3)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.background = "var(--surface-2)";
            }}
          >
            Start analyzing <ArrowRight size={13} />
          </Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="pt-32 pb-24 px-6">
        <div className="max-w-5xl mx-auto">
          {/* Badge */}
          <div className="flex justify-center mb-8">
            <span
              className="inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border-strong)",
                color: "var(--muted)",
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full animate-pulse-dot"
                style={{ background: "#22c55e" }}
              />
              Execution-grounded analysis · no hallucination
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-center text-4xl md:text-5xl font-semibold leading-tight mb-6 tracking-tight"
              style={{ color: "var(--text)", letterSpacing: "-0.02em" }}>
            Your data.{" "}
            <span style={{ color: "var(--accent)" }}>Real computation.</span>
            <br />Zero guesswork.
          </h1>

          {/* Subtext */}
          <p
            className="text-center max-w-2xl mx-auto text-base leading-relaxed mb-10"
            style={{ color: "var(--muted)" }}
          >
            Lumiq executes real code on your dataset to generate answers, charts, and
            insights — every result is grounded in execution, not probability.
          </p>

          {/* CTAs */}
          <div className="flex items-center justify-center gap-4 mb-16">
            <Link
              href="/chat"
              id="cta-start-analyzing"
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-all duration-150"
              style={{
                background: "var(--text)",
                color: "#0b0b0c",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.opacity = "0.85";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.opacity = "1";
              }}
            >
              Start analyzing <ArrowRight size={14} />
            </Link>
            <Link
              href="/login"
              className="px-5 py-2.5 rounded-lg text-sm transition-all duration-150"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border-strong)",
                color: "var(--accent)",
              }}
            >
              Sign in
            </Link>
          </div>

          {/* Demo visual */}
          <HeroDemo />
        </div>
      </section>

      {/* ── How It Works ── */}
      <section
        className="py-24 px-6 border-t"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="max-w-5xl mx-auto">
          <div className="max-w-xs mb-14">
            <p className="text-xs uppercase tracking-widest mb-3 font-medium" style={{ color: "var(--muted)" }}>
              How it works
            </p>
            <h2 className="text-2xl font-semibold tracking-tight" style={{ color: "var(--text)", letterSpacing: "-0.02em" }}>
              Ask → Execute → Output
            </h2>
            <p className="text-sm leading-relaxed mt-3" style={{ color: "var(--muted)" }}>
              A deterministic 4-step pipeline ensures every answer is traceable.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-16">
            <div>
              <FlowStep
                num="01"
                title="Ask in natural language"
                desc="Type your question. Lumiq classifies intent and maps it to the exact columns and data types in your dataset."
              />
              <FlowStep
                num="02"
                title="Code is generated & sandboxed"
                desc="A deterministic Python script is generated, safety-checked, and executed in an isolated sandbox against your real data."
              />
              <FlowStep
                num="03"
                title="Results are verified"
                desc="The output is evaluated for correctness and relevance. If it fails, the system retries — never guesses."
              />
              <FlowStep
                num="04"
                title="Insight is grounded in computation"
                desc="Interpretation is derived strictly from the computed result. No probability, no hallucination — just executed evidence."
                last
              />
            </div>

            <div className="space-y-4">
              {[
                { mode: "executor", label: "Direct computation", desc: "Aggregations, filters, statistical measures executed as Python." },
                { mode: "hybrid", label: "Relationship analysis", desc: "Correlations, comparisons, groupby — all metrics computed before interpretation." },
                { mode: "concept", label: "Conceptual queries", desc: "Explains what a metric means, maps it to your schema — no hallucinated stats." },
                { mode: "irrelevant", label: "Hard rejection", desc: "Off-topic questions are rejected immediately. No speculative answers." },
              ].map((m) => (
                <div
                  key={m.mode}
                  className="rounded-xl px-5 py-4"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                  }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <code
                      className="text-[10px] px-2 py-0.5 rounded font-mono"
                      style={{ background: "var(--surface-3)", color: "var(--accent)", border: "1px solid var(--border-strong)" }}
                    >
                      {m.mode}
                    </code>
                    <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
                      {m.label}
                    </span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                    {m.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Tech Foundation ── */}
      <section
        className="py-24 px-6 border-t"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="max-w-5xl mx-auto">
          <div className="max-w-xs mb-14">
            <p className="text-xs uppercase tracking-widest mb-3 font-medium" style={{ color: "var(--muted)" }}>
              Architecture
            </p>
            <h2 className="text-2xl font-semibold tracking-tight" style={{ color: "var(--text)", letterSpacing: "-0.02em" }}>
              Built on execution,
              not probability
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <FeatureItem
              icon={Terminal}
              title="Sandboxed Python execution"
              desc="Every query generates real Python code that runs against your CSV. Results are deterministic and reproducible."
            />
            <FeatureItem
              icon={ShieldCheck}
              title="Hard-fail on error"
              desc="If code fails after retries, Lumiq returns an explicit failure — never a guessed or fabricated answer."
            />
            <FeatureItem
              icon={Zap}
              title="Schema-aware classification"
              desc="Queries are matched against your actual column names and data types — no generic responses."
            />
            <FeatureItem
              icon={Database}
              title="Query caching"
              desc="Identical queries on identical datasets return cached results instantly with zero additional LLM calls."
            />
            <FeatureItem
              icon={BarChart3}
              title="Native visualizations"
              desc="Charts are generated via matplotlib executed against your data — not mock charts, real data-driven plots."
            />
            <FeatureItem
              icon={Lightbulb}
              title="Evidence-grounded insights"
              desc="Insights are derived strictly from the computation output. Cognitive interpretation never invents statistics."
            />
          </div>
        </div>
      </section>

      {/* ── Bottom CTA ── */}
      <section
        className="py-24 px-6 border-t"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="max-w-2xl mx-auto text-center">
          <h2
            className="text-3xl font-semibold mb-4 tracking-tight"
            style={{ color: "var(--text)", letterSpacing: "-0.02em" }}
          >
            Start with your data
          </h2>
          <p className="text-base leading-relaxed mb-8" style={{ color: "var(--muted)" }}>
            Upload a CSV. Ask a question. Get a verified, computation-backed answer in seconds.
          </p>
          <Link
            href="/chat"
            id="cta-bottom"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium text-sm transition-all duration-150"
            style={{ background: "var(--text)", color: "#0b0b0c" }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.opacity = "0.85";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLAnchorElement).style.opacity = "1";
            }}
          >
            Start analyzing <ArrowRight size={14} />
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer
        className="px-6 py-8 border-t flex items-center justify-between"
        style={{ borderColor: "var(--border)" }}
      >
        <span className="text-xs font-semibold" style={{ color: "var(--muted)" }}>
          Lumiq
        </span>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          Deterministic AI Data Analysis
        </span>
      </footer>
    </div>
  );
}
