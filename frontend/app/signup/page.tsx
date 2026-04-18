import { redirect } from "next/navigation";

// Signup is now handled by the unified login page.
// All new users are automatically registered on their first login attempt.
export default function SignupPage() {
  redirect("/login");
}
