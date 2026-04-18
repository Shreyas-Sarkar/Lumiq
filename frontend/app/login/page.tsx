"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { Loader2, ArrowRight } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Step 1: attempt sign-in
      const { data: signInData, error: signInError } =
        await supabase.auth.signInWithPassword({ email, password });

      if (signInData.session) {
        router.push("/chat");
        return;
      }

      // Step 2: sign-in failed — try signup (new user)
      if (signInError) {
        const { data: signUpData, error: signUpError } =
          await supabase.auth.signUp({ email, password });

        if (signUpData.session) {
          router.push("/chat");
          return;
        }

        if (
          signUpError?.message?.toLowerCase().includes("already registered") ||
          signUpError?.message?.toLowerCase().includes("already been registered")
        ) {
          setError("Incorrect password. Please try again.");
        } else if (signUpError) {
          setError(signUpError.message);
        } else {
          setError(
            "Account created but email confirmation is required. Please contact support."
          );
        }
      }
    } catch {
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center px-4"
      style={{ background: "var(--bg)" }}
    >
      {/* Card */}
      <div
        className="w-full max-w-sm rounded-xl overflow-hidden"
        style={{ border: "1px solid var(--border-strong)", background: "var(--surface)" }}
      >
        {/* Header */}
        <div
          className="px-8 py-6"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--surface-2)" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <div
              className="w-6 h-6 rounded flex items-center justify-center text-[11px] font-bold"
              style={{
                background: "var(--surface-3)",
                border: "1px solid var(--border-strong)",
                color: "var(--text)",
              }}
            >
              L
            </div>
            <span className="font-semibold text-sm" style={{ color: "var(--text)" }}>
              Lumiq
            </span>
          </div>
          <h1 className="text-base font-semibold" style={{ color: "var(--text)" }}>
            Sign in to your workspace
          </h1>
          <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
            New users are registered automatically on first login.
          </p>
        </div>

        {/* Form */}
        <div className="px-8 py-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="auth-email"
                className="block text-xs font-medium mb-1.5"
                style={{ color: "var(--muted)" }}
              >
                Email
              </label>
              <input
                id="auth-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="you@example.com"
                className="w-full rounded-lg px-4 py-2.5 text-sm transition-all duration-150"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border-strong)",
                  color: "var(--text)",
                  outline: "none",
                  fontFamily: "inherit",
                }}
                onFocus={(e) =>
                  ((e.currentTarget as HTMLInputElement).style.borderColor =
                    "rgba(255,255,255,0.2)")
                }
                onBlur={(e) =>
                  ((e.currentTarget as HTMLInputElement).style.borderColor =
                    "var(--border-strong)")
                }
              />
            </div>

            <div>
              <label
                htmlFor="auth-password"
                className="block text-xs font-medium mb-1.5"
                style={{ color: "var(--muted)" }}
              >
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                minLength={6}
                placeholder="••••••••"
                className="w-full rounded-lg px-4 py-2.5 text-sm transition-all duration-150"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border-strong)",
                  color: "var(--text)",
                  outline: "none",
                  fontFamily: "inherit",
                }}
                onFocus={(e) =>
                  ((e.currentTarget as HTMLInputElement).style.borderColor =
                    "rgba(255,255,255,0.2)")
                }
                onBlur={(e) =>
                  ((e.currentTarget as HTMLInputElement).style.borderColor =
                    "var(--border-strong)")
                }
              />
            </div>

            {/* Error */}
            {error && (
              <div
                className="rounded-lg px-4 py-3 text-xs leading-relaxed"
                style={{
                  background: "rgba(239, 68, 68, 0.06)",
                  border: "1px solid rgba(239, 68, 68, 0.2)",
                  color: "#fca5a5",
                }}
              >
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              id="auth-submit"
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-medium transition-all duration-150"
              style={{
                background: loading ? "var(--surface-3)" : "var(--text)",
                color: loading ? "var(--muted)" : "var(--bg)",
                border: "1px solid transparent",
                cursor: loading ? "wait" : "pointer",
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Signing in…
                </>
              ) : (
                <>
                  Continue
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>
        </div>
      </div>

      {/* Back to home */}
      <div className="absolute bottom-8 left-0 right-0 text-center">
        <Link
          href="/"
          className="text-xs transition-colors"
          style={{ color: "var(--muted)" }}
        >
          ← Back to Lumiq
        </Link>
      </div>
    </div>
  );
}
