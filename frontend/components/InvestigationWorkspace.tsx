"use client";
import BrandMark from "@/components/BrandMark";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowUpRight,
  Database,
  Download,
  FileSpreadsheet,
  LogOut,
  Plus,
  RefreshCw,
  ChevronRight,
  PanelRightClose,
  Pencil,
} from "lucide-react";
import { useSession } from "@/hooks/useSession";
import { signOut } from "@/lib/auth";
import {
  apiError,
  createChat,
  renameWorkspace,
  getChat,
  getChats,
  getInvestigations,
  getInvestigation,
  getProfile,
  startInvestigation,
  resumeInvestigation,
  uploadDataset,
  downloadEvidence,
} from "@/lib/api";
import { Chat, Dataset, Message } from "@/types";
import {
  AnalysisPlan,
  DatasetProfile,
  Evidence,
  Investigation,
} from "@/types/investigations";
import AnalysisBlock from "./AnalysisBlock";
import PlanEditor from "./PlanEditor";

const done = new Set([
  "complete",
  "failed",
  "cancelled",
  "needs_clarification",
  "waiting_for_quota",
  "interpretation_unavailable",
]);
const labels: Record<string, string> = {
  queued: "Queued",
  planning: "Preparing the calculation plan",
  computing: "Computing evidence",
  interpreting: "Interpreting results",
  needs_clarification: "Your input is needed",
  waiting_for_quota: "Waiting for provider capacity",
  interpretation_unavailable: "Evidence ready · interpretation unavailable",
  interrupted: "Interrupted · recovering saved progress",
  complete: "Saved result",
  failed: "Could not complete",
  cancelled: "Cancelled",
};
const number = (value: unknown) =>
  typeof value === "number"
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(
        value,
      )
    : value == null
      ? "—"
      : String(value);

function EvidenceView({
  evidence,
  selected,
  onSelect,
  onDownload,
}: {
  evidence: Evidence;
  selected: boolean;
  onSelect: () => void;
  onDownload: () => void;
}) {
  const chart = evidence.chart;
  const max = Math.max(
    ...(chart?.rows.map((r) => Math.abs(Number(r[chart.y]) || 0)) || []),
    1,
  );
  const columns = evidence.rows.length ? Object.keys(evidence.rows[0]) : [];
  return (
    <section
      id={`evidence-${evidence.id}`}
      className={`evidence-card ${selected ? "selected" : ""}`}
      aria-label={`Evidence ${evidence.id}`}
    >
      <div className="section-heading">
        <div>
          <span className="eyebrow">Computed evidence · {evidence.id}</span>
          <h3>
            {evidence.metric} · {evidence.operation.replaceAll("_", " ")}
          </h3>
        </div>
        <button
          className="icon-button"
          aria-label={`Download ${evidence.id} CSV`}
          onClick={onDownload}
        >
          <Download size={17} />
        </button>
      </div>
      {Object.keys(evidence.summary).length > 0 && (
        <dl className="measurement-grid">
          {Object.entries(evidence.summary).map(([key, value]) => (
            <div key={key}>
              <dt>{key.replaceAll("_", " ")}</dt>
              <dd>
                {number(value)}
                {key === "percent_change" && value !== null ? "%" : ""}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {chart && (
        <figure
          className="chart"
          aria-label={`${evidence.metric} ${chart.y} by ${chart.x}`}
        >
          <figcaption>
            {chart.y.replaceAll("_", " ")} by {chart.x}
          </figcaption>
          {chart.rows.map((row, i) => (
            <div
              className="chart-row"
              key={i}
              title={`${number(row[chart.x])}: ${number(row[chart.y])}`}
            >
              <span>{number(row[chart.x])}</span>
              <div className="bar-track">
                <div
                  className={Number(row[chart.y]) < 0 ? "bar negative" : "bar"}
                  style={{
                    width:
                      row[chart.y] == null
                        ? "0%"
                        : `${Math.max((Math.abs(Number(row[chart.y]) || 0) / max) * 100, 0)}%`,
                  }}
                />
              </div>
              <strong>{number(row[chart.y])}</strong>
            </div>
          ))}
        </figure>
      )}
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label={`${evidence.id} result table`}
      >
        <table>
          <caption className="sr-only">
            Computed {evidence.metric} results
          </caption>
          <thead>
            <tr>
              {columns.map((c) => (
                <th scope="col" key={c}>
                  {c.replaceAll("_", " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {evidence.rows.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c}>{number(r[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {!evidence.rows.length && <p>No matching rows. Try broader filters.</p>}
      </div>
      <button className="text-button" onClick={onSelect}>
        Inspect filters and coverage <ChevronRight size={14} />
      </button>
    </section>
  );
}

export default function InvestigationWorkspace({
  chatId,
}: {
  chatId?: string;
}) {
  const router = useRouter();
  const { token, email, error: sessionError } = useSession();
  const [chats, setChats] = useState<Chat[]>([]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [runs, setRuns] = useState<Investigation[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [clarification, setClarification] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [evidenceId, setEvidenceId] = useState<string | null>(null);
  const [showProfile, setShowProfile] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [editPlan, setEditPlan] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const renameDialog = useRef<HTMLDialogElement>(null);
  const [renameTarget, setRenameTarget] = useState<Chat | null>(null);
  const [workspaceName, setWorkspaceName] = useState("");
  const [renameError, setRenameError] = useState("");
  const [renaming, setRenaming] = useState(false);
  const workspace = chats.find((chat) => chat.id === chatId);
  const workspaceTitle =
    workspace?.title === "New Chat" ? "Untitled workspace" : workspace?.title;
  const loadVersion = useRef(0);
  const request = useRef<{ fingerprint: string; id: string } | null>(null);
  const current = runs.find((r) => r.id === active);
  const numericColumns =
    profile?.columns.filter((column) => column.kind === "number") || [];
  const starterMetric =
    numericColumns.find((column) => /amount|revenue|sales/i.test(column.name))
      ?.name ||
    numericColumns.find((column) => !/(^id$|_id$)/i.test(column.name))?.name;
  const starterCategory =
    profile?.columns.find((column) => /city|region|category/i.test(column.name))
      ?.name ||
    profile?.columns.find((column) => column.kind === "category")?.name;
  const starterDate = profile?.columns.find(
    (column) => column.kind === "date",
  )?.name;
  const starters = [
    "Summarize the main numeric metrics",
    ...(starterMetric && starterCategory
      ? [`Break down ${starterMetric} by ${starterCategory}`]
      : []),
    ...(starterMetric && starterDate
      ? [`Compare ${starterMetric} across two periods using ${starterDate}`]
      : starterMetric
        ? [`What is the average ${starterMetric}?`]
        : []),
  ];
  const status = current?.state.status;
  const running = !!status && !done.has(status);
  const selectedEvidence = current?.state.evidence.find(
    (e) => e.id === evidenceId,
  );

  const load = useCallback(async () => {
    if (!token) return;
    const version = ++loadVersion.current;
    setLoading(true);
    setError("");
    try {
      const list = await getChats(token);
      if (version !== loadVersion.current) return;
      setChats(list);
      if (chatId) {
        const data = await getChat(chatId, token);
        if (version !== loadVersion.current) return;
        setDataset(data.dataset);
        setMessages(data.messages);
        if (data.dataset) {
          const results = await Promise.allSettled([
            getProfile(chatId, token),
            getInvestigations(chatId, token),
          ]);
          if (version !== loadVersion.current) return;
          if (results[0].status === "fulfilled") setProfile(results[0].value);
          else setError(apiError(results[0].reason));
          if (results[1].status === "fulfilled") {
            setRuns(results[1].value);
            setActive((previous) =>
              results[1].status === "fulfilled" &&
              results[1].value.some((r) => r.id === previous)
                ? previous
                : results[1].status === "fulfilled"
                  ? results[1].value[0]?.id || null
                  : null,
            );
          } else setError(apiError(results[1].reason));
        } else {
          setRuns([]);
          setProfile(null);
          setActive(null);
        }
      }
    } catch (e) {
      if (version === loadVersion.current) setError(apiError(e));
    } finally {
      if (version === loadVersion.current) setLoading(false);
    }
  }, [token, chatId]);
  useEffect(() => {
    setDataset(null);
    setRuns([]);
    setMessages([]);
    setProfile(null);
    setActive(null);
    setEvidenceId(null);
    setQuestion("");
    request.current = null;
    setRenameTarget(null);
  }, [chatId]);
  useEffect(() => {
    void load();
    return () => {
      loadVersion.current += 1;
    };
  }, [load]);
  useEffect(() => {
    if (!active || !running || !token) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const run = await getInvestigation(active, token);
        if (!stopped)
          setRuns((prev) => prev.map((r) => (r.id === run.id ? run : r)));
      } catch (e) {
        if (!stopped) setError(apiError(e));
      }
      if (!stopped) timer = setTimeout(poll, 2000);
    };
    timer = setTimeout(poll, 800);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [active, running, token]);
  useEffect(() => {
    if (showUpload) dialog.current?.showModal();
    else dialog.current?.close();
  }, [showUpload]);

  useEffect(() => {
    if (renameTarget) renameDialog.current?.showModal();
    else renameDialog.current?.close();
  }, [renameTarget]);

  async function saveWorkspaceName() {
    if (!renameTarget || !workspaceName.trim() || !token) return;
    setRenaming(true);
    setRenameError("");
    try {
      const renamed = await renameWorkspace(
        renameTarget.id,
        workspaceName.trim(),
        token,
      );
      setChats((previous) =>
        previous.map((chat) => (chat.id === renamed.id ? renamed : chat)),
      );
      setRenameTarget(null);
    } catch (failure) {
      setRenameError(apiError(failure));
    } finally {
      setRenaming(false);
    }
  }

  async function newWorkspace() {
    setBusy(true);
    try {
      const chat = await createChat(token);
      router.push(`/chat/${chat.id}`);
    } catch (e) {
      setError(apiError(e));
    } finally {
      setBusy(false);
    }
  }
  async function upload() {
    if (!file || !chatId) return;
    if (file.size > 5 * 1024 * 1024) {
      setError("CSV exceeds the 5 MB upload limit.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const data = await uploadDataset(chatId, file, token);
      setDataset(data);
      setShowProfile(true);
      setShowUpload(false);
      setFile(null);
      await load();
    } catch (e) {
      setError(apiError(e));
    } finally {
      setBusy(false);
    }
  }
  async function investigate(text = question, plan?: AnalysisPlan) {
    if (!chatId || !text.trim() || busy) return;
    setBusy(true);
    setError("");
    setEvidenceId(null);
    try {
      const fingerprint = JSON.stringify({
        question: text,
        parent_id: current?.id,
        plan,
      });
      if (request.current?.fingerprint !== fingerprint)
        request.current = { fingerprint, id: crypto.randomUUID() };
      const run = await startInvestigation(
        {
          chat_id: chatId,
          question: text,
          request_id: request.current!.id,
          parent_id: current?.id,
          plan,
        },
        token,
      );
      setRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)]);
      setActive(run.id);
      setQuestion("");
      request.current = null;
      setEditPlan(false);
    } catch (e) {
      setError(apiError(e));
    } finally {
      setBusy(false);
    }
  }
  async function resume() {
    if (!current) return;
    setBusy(true);
    setError("");
    try {
      const run = await resumeInvestigation(
        current.id,
        { clarification: clarification.trim() || undefined },
        token,
      );
      setRuns((prev) => prev.map((r) => (r.id === run.id ? run : r)));
      setClarification("");
    } catch (e) {
      setError(apiError(e));
    } finally {
      setBusy(false);
    }
  }
  async function download(e: Evidence) {
    if (!current) return;
    try {
      await downloadEvidence(current.id, e.id, token);
    } catch (err) {
      setError(apiError(err));
    }
  }
  function edit() {
    if (!current?.state.plan) return;
    setEditPlan(!editPlan);
  }

  return (
    <div className="workspace">
      <aside
        className="workspace-nav"
        aria-label="Workspaces and investigations"
      >
        <a className="workspace-brand" href="/chat">
          <BrandMark />
          Lumiq
          <span className="brand-caption">DATA WORKSPACE</span>
        </a>
        <button
          className="primary-button nav-new"
          onClick={newWorkspace}
          disabled={busy || !token}
        >
          <Plus size={16} /> New workspace
        </button>
        <div className="nav-section">
          <span className="eyebrow">Workspaces</span>
          {chats.map((chat) => (
            <button
              key={chat.id}
              title={
                chat.title === "New Chat" ? "Untitled workspace" : chat.title
              }
              className={`nav-item ${chat.id === chatId ? "active" : ""}`}
              onClick={() => router.push(`/chat/${chat.id}`)}
              aria-current={chat.id === chatId ? "page" : undefined}
            >
              <Database size={15} />
              <span>
                {chat.title === "New Chat" ? "Untitled workspace" : chat.title}
              </span>
            </button>
          ))}
          {!chats.length && !loading && (
            <p className="small-muted">Your uploaded data starts here.</p>
          )}
        </div>
        {dataset && (
          <div className="nav-section dataset-summary">
            <span className="eyebrow">Dataset</span>
            <strong>
              <FileSpreadsheet size={16} />
              {dataset.filename}
            </strong>
            <p>
              {number(dataset.row_count)} rows · {dataset.column_count} columns
            </p>
            <button
              className="text-button"
              onClick={() => setShowProfile(!showProfile)}
            >
              Inspect data quality <ChevronRight size={14} />
            </button>
          </div>
        )}
        {runs.length > 0 && (
          <div className="nav-section">
            <span className="eyebrow">Saved investigations</span>
            {runs.map((run) => (
              <button
                key={run.id}
                className={`nav-item investigation-item ${run.id === active ? "active" : ""}`}
                onClick={() => {
                  setActive(run.id);
                  setEvidenceId(null);
                  setEditPlan(false);
                }}
              >
                <span>
                  {run.state.plan?.title || run.state.question}
                  <small>{labels[run.state.status] || run.state.status}</small>
                </span>
              </button>
            ))}
          </div>
        )}
        <div className="nav-account">
          <span title={email}>{email}</span>
          <button
            aria-label="Sign out"
            className="icon-button"
            onClick={signOut}
          >
            <LogOut size={16} />
          </button>
        </div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-header">
          <div>
            <span className="eyebrow">
              {workspaceTitle || "Investigation workspace"}
            </span>
            <p>
              {dataset?.filename || "Turn a CSV into an inspectable answer"}
            </p>
          </div>
          <div className="workspace-actions">
            {workspace && (
              <button
                className="icon-button"
                aria-label="Rename workspace"
                title="Rename workspace"
                disabled={loading || busy}
                onClick={() => {
                  setWorkspaceName(workspaceTitle || "");
                  setRenameError("");
                  setRenameTarget(workspace);
                }}
              >
                <Pencil size={17} />
              </button>
            )}
            <button
              className="icon-button"
              aria-label="Reload workspace"
              onClick={load}
            >
              <RefreshCw size={17} />
            </button>
          </div>
        </header>
        {(error || sessionError) && (
          <div role="alert" className="workspace-error">
            <p>{sessionError || error}</p>
            <button className="text-button" onClick={load}>
              Retry connection
            </button>
            {(sessionError || error).includes("session") && (
              <a href="/login">Sign in again</a>
            )}
          </div>
        )}
        {loading ? (
          <div className="workspace-loading" role="status">
            Loading your saved workspace…
            <p>Free hosting may take a moment to wake up.</p>
          </div>
        ) : !chatId ? (
          <div className="workspace-empty">
            <span className="eyebrow">From question to evidence</span>
            <h1>
              Know what changed.
              <br />
              See how it was calculated.
            </h1>
            <p>
              Upload a CSV, choose a question, and inspect the measurements
              behind the answer. Plans, results, and data quality stay together.
            </p>
            <button
              className="primary-button"
              onClick={newWorkspace}
              disabled={busy}
            >
              Create a workspace <ArrowUpRight size={17} />
            </button>
            <a href="/samples/revenue.csv" download>
              Try the sample revenue CSV
            </a>
          </div>
        ) : !dataset ? (
          <div className="workspace-empty">
            <FileSpreadsheet size={30} />
            <h1>Start with your data.</h1>
            <p>
              Upload one UTF-8 CSV. Lumiq profiles its columns and checks data
              quality before you investigate.
            </p>
            <button
              className="primary-button"
              onClick={() => {
                setError("");
                setShowUpload(true);
              }}
            >
              Upload a CSV <Plus size={16} />
            </button>
            <a href="/samples/revenue.csv" download>
              Download a sample · six rows, known answers
            </a>
            <span className="small-muted">
              Up to 5 MB · 100,000 rows · ISO dates
            </span>
          </div>
        ) : (
          <div className="workspace-canvas">
            {showProfile && profile && (
              <section className="profile-card">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">Dataset profile</span>
                    <h2>Understand the starting point</h2>
                  </div>
                  <button
                    className="icon-button"
                    onClick={() => setShowProfile(false)}
                    aria-label="Close data profile"
                  >
                    <PanelRightClose size={17} />
                  </button>
                </div>
                <p className="small-muted">
                  {profile.rows} rows · {profile.duplicate_rows} duplicate rows
                  retained · {number(profile.memory_bytes / 1024)} KB parsed
                </p>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Dataset columns"
                >
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Column</th>
                        <th scope="col">Type</th>
                        <th scope="col">Missing</th>
                        <th scope="col">Coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {profile.columns.map((c) => (
                        <tr key={c.name}>
                          <th scope="row">{c.name}</th>
                          <td>{c.kind}</td>
                          <td>{c.missing}</td>
                          <td>
                            {c.start
                              ? `${c.start.slice(0, 10)} — ${c.end?.slice(0, 10)}`
                              : `${c.unique} distinct values`}
                            {c.invalid_dates
                              ? ` · ${c.invalid_dates} invalid dates`
                              : ""}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
            {!current ? (
              <section className="investigation-start">
                <span className="eyebrow">Ready to investigate</span>
                <h1>What would you like to understand?</h1>
                <p>
                  Start with a total, compare two periods, or break a metric
                  down by category.
                </p>
                <div className="starter-list">
                  {starters.map((q) => (
                    <button key={q} onClick={() => setQuestion(q)}>
                      {q}
                      <ArrowUpRight size={16} />
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <article className="investigation-result">
                <div className="result-heading">
                  <span
                    className={`run-status ${status === "complete" ? "complete" : ""}`}
                    role="status"
                  >
                    {labels[status || ""] || status}
                  </span>
                  <span className="small-muted">
                    {new Date(current.created_at).toLocaleString()}
                  </span>
                </div>
                <h1>{current.state.plan?.title || "Your investigation"}</h1>
                <p className="investigation-question">
                  {current.state.question}
                </p>
                {current.state.plan && (
                  <details className="plan-panel">
                    <summary>
                      Calculation plan · {current.state.plan.steps.length}{" "}
                      {current.state.plan.steps.length === 1 ? "step" : "steps"}
                    </summary>
                    <p>{current.state.plan.explanation}</p>
                    <ol>
                      {current.state.plan.steps.map((step, i) => (
                        <li key={i}>
                          <strong>{step.operation.replaceAll("_", " ")}</strong>{" "}
                          {step.metric}
                          {step.second_metric
                            ? ` vs ${step.second_metric}`
                            : ""}
                          {step.group_by ? ` by ${step.group_by}` : ""}
                          {step.periods?.length
                            ? ` · ${step.periods.map((p) => p.join(" to ")).join(" vs ")}`
                            : ""}
                          {step.filters?.length
                            ? ` · ${step.filters.length} filters`
                            : ""}
                        </li>
                      ))}
                    </ol>
                    {current.state.plan.assumptions?.map((a, i) => (
                      <p className="small-muted" key={i}>
                        {a}
                      </p>
                    ))}
                    {current.state.plan.steps.length > 0 && (
                      <button className="text-button" onClick={edit}>
                        Edit plan and run a new revision
                      </button>
                    )}
                  </details>
                )}
                {editPlan && current.state.plan && (
                  <PlanEditor
                    key={current.id}
                    plan={current.state.plan}
                    profile={profile}
                    busy={busy}
                    onSubmit={(plan) =>
                      void investigate(
                        `Run the revised calculation plan: ${plan.title}. ${plan.explanation}`,
                        plan,
                      )
                    }
                  />
                )}
                {status === "needs_clarification" && (
                  <form
                    className="clarification-panel"
                    onSubmit={(e) => {
                      e.preventDefault();
                      void resume();
                    }}
                  >
                    <label htmlFor="clarification">
                      {current.state.plan?.clarification}
                    </label>
                    <input
                      id="clarification"
                      value={clarification}
                      maxLength={1000}
                      onChange={(e) => setClarification(e.target.value)}
                      placeholder="e.g. January and February 2026"
                      required
                    />
                    <button
                      className="primary-button"
                      disabled={busy || !clarification.trim()}
                    >
                      Continue investigation
                    </button>
                  </form>
                )}
                {current.state.error && (
                  <div className="run-error">
                    <h2>
                      {current.state.evidence.length
                        ? "Your computed evidence is saved."
                        : "Work paused."}
                    </h2>
                    <p>{current.state.error.message}</p>
                    {current.state.error.retry_at && (
                      <p>
                        Earliest provider retry:{" "}
                        {new Date(
                          current.state.error.retry_at,
                        ).toLocaleString()}
                        . Availability will be checked again.
                      </p>
                    )}
                    {(current.state.error.resumable ||
                      (status === "interpretation_unavailable" &&
                        current.state.http_attempts < 6 &&
                        current.state.error.reasons?.some(
                          (reason) => reason.reason === "invalid_output",
                        ))) && (
                      <button
                        className="secondary-button"
                        disabled={busy}
                        onClick={resume}
                      >
                        Resume saved work
                      </button>
                    )}
                  </div>
                )}
                {current.state.concept_answer && (
                  <p className="concept-answer">
                    {current.state.concept_answer}
                  </p>
                )}
                {current.state.findings.length > 0 && (
                  <section className="findings">
                    <span className="eyebrow">Observations</span>
                    {current.state.findings.map((finding, i) => (
                      <div key={i}>
                        <p>{finding.observation}</p>
                        <button
                          className="evidence-link"
                          onClick={() => {
                            setEvidenceId(finding.evidence_id);
                            document
                              .getElementById(`evidence-${finding.evidence_id}`)
                              ?.scrollIntoView({ block: "nearest" });
                          }}
                        >
                          View {finding.evidence_id} <ChevronRight size={13} />
                        </button>
                        {finding.hypothesis && (
                          <p className="hypothesis">
                            Unverified hypothesis: {finding.hypothesis}
                          </p>
                        )}
                      </div>
                    ))}
                  </section>
                )}
                <div className="evidence-list">
                  {current.state.evidence.map((e) => (
                    <EvidenceView
                      evidence={e}
                      key={e.id}
                      selected={evidenceId === e.id}
                      onSelect={() =>
                        setEvidenceId(evidenceId === e.id ? null : e.id)
                      }
                      onDownload={() => download(e)}
                    />
                  ))}
                </div>
                {selectedEvidence && (
                  <aside
                    className="coverage-panel"
                    aria-label="Evidence details"
                  >
                    <div className="section-heading">
                      <h2>Filters & coverage · {selectedEvidence.id}</h2>
                      <button
                        className="icon-button"
                        aria-label="Close evidence details"
                        onClick={() => setEvidenceId(null)}
                      >
                        <PanelRightClose size={16} />
                      </button>
                    </div>
                    <p>
                      {selectedEvidence.filtered_rows} of{" "}
                      {selectedEvidence.input_rows} input rows match the
                      selected filters.
                    </p>
                    <p>
                      {selectedEvidence.filters.length
                        ? selectedEvidence.filters
                            .map(
                              (f) =>
                                `${f.column} ${f.operator} ${JSON.stringify(f.value)}`,
                            )
                            .join("; ")
                        : "No additional filters."}
                    </p>
                    <ul>
                      {selectedEvidence.limitations.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                    <p className="small-muted">
                      Evidence fingerprint: {selectedEvidence.hash.slice(0, 16)}
                    </p>
                  </aside>
                )}
                {current.state.follow_ups?.length ? (
                  <div className="followup-list">
                    {current.state.follow_ups.map((q) => (
                      <button
                        className="secondary-button"
                        key={q}
                        onClick={() => setQuestion(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                ) : null}
                <details className="run-details">
                  <summary>Run history & model usage</summary>
                  <p>
                    {current.state.http_attempts} of 6 network attempts used.
                    Saved calculations are retained across interpretation
                    retries.
                  </p>
                  {current.state.interpretation_reused && (
                    <p>
                      Interpretation reused from a validated saved run with
                      identical inputs and model configuration.
                    </p>
                  )}
                  <ul>
                    {current.state.events.map((event, i) => (
                      <li key={i}>
                        {labels[event.status] || event.status} ·{" "}
                        {new Date(event.at).toLocaleTimeString()}
                      </li>
                    ))}
                  </ul>
                  {current.state.provider_calls.map((call, i) => (
                    <p key={i}>
                      {call.stage} · {call.provider} / {call.model} ·{" "}
                      {call.outcome.replaceAll("_", " ")}
                    </p>
                  ))}
                </details>
              </article>
            )}
            <form
              className="question-composer"
              onSubmit={(e) => {
                e.preventDefault();
                void investigate();
              }}
            >
              <label htmlFor="investigation-question">
                {current
                  ? "Ask a follow-up using this investigation’s context"
                  : "Ask a question about your dataset"}
              </label>
              <div>
                <textarea
                  id="investigation-question"
                  rows={2}
                  maxLength={1000}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What changed, over which periods, and for which metric?"
                  disabled={busy || running}
                />
                <button
                  className="primary-button"
                  disabled={busy || running || !question.trim()}
                >
                  {busy ? "Saving…" : running ? "Working…" : "Investigate"}
                  <ArrowUpRight size={16} />
                </button>
              </div>
              <p className="small-muted">
                Calculations use your data. Interpretations are checked against
                evidence and may still need your judgment.
              </p>
            </form>
            {messages.length > 0 && (
              <details className="legacy-history">
                <summary>Previous conversation · read only</summary>
                {messages.map((message) =>
                  message.role === "assistant" ? (
                    <AnalysisBlock key={message.id} message={message} />
                  ) : (
                    <p className="legacy-question" key={message.id}>
                      {message.content}
                    </p>
                  ),
                )}
              </details>
            )}
          </div>
        )}
      </main>
      <dialog
        ref={renameDialog}
        className="upload-dialog"
        aria-labelledby="rename-workspace-title"
        onCancel={(event) => {
          if (renaming) event.preventDefault();
          else setRenameTarget(null);
        }}
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void saveWorkspaceName();
          }}
        >
          <h2 id="rename-workspace-title">Rename workspace</h2>
          <p>
            Give this dataset and its investigations a name you’ll recognize.
          </p>
          <label htmlFor="workspace-name">Workspace name</label>
          <input
            id="workspace-name"
            autoFocus
            required
            maxLength={120}
            value={workspaceName}
            onChange={(event) => setWorkspaceName(event.target.value)}
            disabled={renaming}
          />
          {renameError && (
            <p role="alert" className="upload-error">
              {renameError}
            </p>
          )}
          <div className="workspace-actions">
            <button
              type="button"
              className="text-button"
              disabled={renaming}
              onClick={() => setRenameTarget(null)}
            >
              Cancel
            </button>
            <button
              className="primary-button"
              disabled={renaming || !workspaceName.trim()}
            >
              {renaming ? "Saving…" : "Save name"}
            </button>
          </div>
        </form>
      </dialog>
      <dialog
        ref={dialog}
        className="upload-dialog"
        onCancel={() => setShowUpload(false)}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void upload();
          }}
        >
          <div className="section-heading">
            <h2>Upload a dataset</h2>
            <button
              type="button"
              className="text-button"
              onClick={() => setShowUpload(false)}
              disabled={busy}
            >
              Close
            </button>
          </div>
          <p>
            UTF-8 CSV · up to 5 MB and 100,000 rows.
            <br />
            Dates should use YYYY-MM-DD.
          </p>
          <label htmlFor="csv-file">Choose your CSV file</label>
          <input
            id="csv-file"
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            required
          />
          {error && (
            <p role="alert" className="upload-error">
              {error}
            </p>
          )}
          <button className="primary-button" disabled={busy || !file}>
            {busy ? "Uploading…" : "Upload and profile"}
          </button>
        </form>
      </dialog>
    </div>
  );
}
