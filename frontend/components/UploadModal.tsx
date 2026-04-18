"use client";

import { useState, useRef, useCallback } from "react";
import { X, Upload, FileText, Loader2, CheckCircle2 } from "lucide-react";

interface UploadModalProps {
  chatId: string;
  token: string;
  onSuccess: (dataset: unknown) => void;
  onClose: () => void;
}

export default function UploadModal({
  chatId,
  token,
  onSuccess,
  onClose,
}: UploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith(".csv")) {
      setFile(dropped);
      setError("");
    } else {
      setError("Only CSV files are accepted.");
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected && selected.name.endsWith(".csv")) {
      setFile(selected);
      setError("");
    } else if (selected) {
      setError("Only CSV files are accepted.");
    }
  };

  async function handleUpload() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const { uploadDataset } = await import("@/lib/api");
      const dataset = await uploadDataset(chatId, file, token);
      onSuccess(dataset);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(
        axiosErr?.response?.data?.detail || "Upload failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  // Format bytes
  function formatSize(bytes: number) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4"
      style={{ background: "rgba(0, 0, 0, 0.7)", backdropFilter: "blur(4px)" }}
    >
      <div
        className="w-full max-w-md rounded-xl overflow-hidden animate-fade-slide-in"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border-strong)",
          opacity: 0,
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div>
            <h2 className="text-sm font-semibold" style={{ color: "var(--text)" }}>
              Upload dataset
            </h2>
            <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
              CSV files up to 50 MB
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 transition-colors"
            style={{ color: "var(--muted)" }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.color = "var(--text)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.color = "var(--muted)")
            }
          >
            <X size={16} />
          </button>
        </div>

        {/* Drop zone */}
        <div className="p-6">
          <div
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => fileInputRef.current?.click()}
            className="rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-all duration-150"
            style={{
              borderColor: isDragging
                ? "rgba(255,255,255,0.25)"
                : file
                ? "rgba(34, 197, 94, 0.4)"
                : "var(--border-strong)",
              background: isDragging
                ? "var(--surface-3)"
                : file
                ? "rgba(34, 197, 94, 0.04)"
                : "var(--surface-2)",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />

            {file ? (
              <div className="flex flex-col items-center gap-3">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{
                    background: "rgba(34, 197, 94, 0.1)",
                    border: "1px solid rgba(34, 197, 94, 0.3)",
                  }}
                >
                  <CheckCircle2 size={20} style={{ color: "#22c55e" }} />
                </div>
                <div>
                  <p className="text-sm font-medium" style={{ color: "var(--text)" }}>
                    {file.name}
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                    {formatSize(file.size)}
                  </p>
                </div>
                <p className="text-xs" style={{ color: "var(--muted)" }}>
                  Click to replace
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{
                    background: "var(--surface-3)",
                    border: "1px solid var(--border-strong)",
                  }}
                >
                  <Upload size={18} style={{ color: "var(--muted)" }} />
                </div>
                <div>
                  <p className="text-sm font-medium" style={{ color: "var(--text)" }}>
                    Drop your CSV here
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                    or{" "}
                    <span style={{ color: "var(--accent)" }}>click to browse</span>
                  </p>
                </div>
                <div
                  className="flex items-center gap-2 text-[10px] px-3 py-1.5 rounded-full"
                  style={{
                    background: "var(--surface-3)",
                    border: "1px solid var(--border)",
                    color: "var(--muted)",
                  }}
                >
                  <FileText size={10} />
                  .csv only · max 50 MB
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div
              className="mt-3 rounded-lg px-4 py-3 text-xs animate-fade-in"
              style={{
                background: "rgba(239, 68, 68, 0.06)",
                border: "1px solid rgba(239, 68, 68, 0.2)",
                color: "#fca5a5",
              }}
            >
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 mt-5">
            <button
              onClick={onClose}
              className="flex-1 py-2.5 rounded-lg text-sm transition-all duration-150"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border-strong)",
                color: "var(--muted)",
              }}
              onMouseEnter={(e) =>
                ((e.currentTarget as HTMLButtonElement).style.color = "var(--text)")
              }
              onMouseLeave={(e) =>
                ((e.currentTarget as HTMLButtonElement).style.color = "var(--muted)")
              }
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="flex-1 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 flex items-center justify-center gap-2"
              style={{
                background: file && !loading ? "var(--text)" : "var(--surface-3)",
                color: file && !loading ? "var(--bg)" : "var(--muted)",
                border: "1px solid transparent",
                cursor: !file || loading ? "not-allowed" : "pointer",
                opacity: !file || loading ? 0.6 : 1,
              }}
            >
              {loading ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Uploading…
                </>
              ) : (
                "Upload dataset"
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
