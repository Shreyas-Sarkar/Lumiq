"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export function useSession() {
  const router = useRouter();
  const [token, setToken] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    const apply = (
      session: { access_token: string; user: { email?: string } } | null,
    ) => {
      if (!alive) return;
      if (!session) {
        setToken("");
        router.replace("/login");
        return;
      }
      setToken(session.access_token);
      setEmail(session.user.email || "");
      setError("");
    };
    supabase.auth
      .getSession()
      .then(({ data, error }) =>
        error
          ? setError("Could not restore your session. Sign in again.")
          : apply(data.session),
      );
    const { data } = supabase.auth.onAuthStateChange((_event, session) =>
      apply(session),
    );
    return () => {
      alive = false;
      data.subscription.unsubscribe();
    };
  }, [router]);
  return { token, email, error };
}
