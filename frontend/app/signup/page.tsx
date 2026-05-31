"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, ArrowRight } from "lucide-react";
import { supabase } from "@/lib/supabase";

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
      });

      if (signUpError) {
        const message = signUpError.message.toLowerCase();
        if (message.includes("already registered") || message.includes("already been registered")) {
          setError("That email already exists. Please sign in instead.");
        } else {
          setError(signUpError.message);
        }
        return;
      }

      if (data.session) {
        router.push("/chat");
        return;
      }

      setSuccess("Account created. Check your email to confirm your account, then sign in.");
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
      <div
        className="w-full max-w-sm rounded-xl overflow-hidden"
        style={{ border: "1px solid var(--border-strong)", background: "var(--surface)" }}
      >
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
            Create your account
          </h1>
          <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
            Sign up to start new analysis sessions and resume them later.
          </p>
        </div>

        <div className="px-8 py-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="signup-email"
                className="block text-xs font-medium mb-1.5"
                style={{ color: "var(--muted)" }}
              >
                Email
              </label>
              <input
                id="signup-email"
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
              />
            </div>

            <div>
              <label
                htmlFor="signup-password"
                className="block text-xs font-medium mb-1.5"
                style={{ color: "var(--muted)" }}
              >
                Password
              </label>
              <input
                id="signup-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="new-password"
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
              />
            </div>

            <div>
              <label
                htmlFor="signup-confirm-password"
                className="block text-xs font-medium mb-1.5"
                style={{ color: "var(--muted)" }}
              >
                Confirm password
              </label>
              <input
                id="signup-confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
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
              />
            </div>

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

            {success && (
              <div
                className="rounded-lg px-4 py-3 text-xs leading-relaxed"
                style={{
                  background: "rgba(34, 197, 94, 0.08)",
                  border: "1px solid rgba(34, 197, 94, 0.2)",
                  color: "#86efac",
                }}
              >
                {success}
              </div>
            )}

            <button
              id="signup-submit"
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
                  Creating account…
                </>
              ) : (
                <>
                  Create account
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>
        </div>
      </div>

      <div className="absolute bottom-8 left-0 right-0 text-center">
        <Link
          href="/login"
          className="text-xs transition-colors"
          style={{ color: "var(--accent)" }}
        >
          Already have an account? Sign in
        </Link>
      </div>
    </div>
  );
}
