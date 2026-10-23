import { test, expect, Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const chat = "22222222-2222-4222-8222-222222222222",
  runId = "55555555-5555-4555-8555-555555555555";
const dataset = {
  id: "33333333-3333-4333-8333-333333333333",
  chat_id: chat,
  filename: "revenue.csv",
  row_count: 6,
  column_count: 5,
  schema_info: { columns: [] },
  sample_rows: [],
  uploaded_at: "2026-10-02T00:00:00Z",
};
const profile = {
  version: 1,
  rows: 6,
  duplicate_rows: 0,
  memory_bytes: 1000,
  columns: [
    {
      name: "date",
      dtype: "object",
      kind: "date",
      missing: 0,
      unique: 6,
      start: "2026-01-05",
      end: "2026-02-28",
    },
    { name: "revenue", dtype: "int64", kind: "number", missing: 0, unique: 6 },
  ],
  limitations: [],
};
const evidence = {
  id: "e1",
  operation: "compare_periods",
  metric: "revenue",
  aggregation: "sum",
  summary: { previous: 300, current: 340, change: 40, percent_change: 13.333 },
  rows: [
    { region: "North", previous: 220, current: 280, change: 60 },
    { region: "South", previous: 80, current: 60, change: -20 },
  ],
  chart: {
    kind: "bar",
    x: "region",
    y: "change",
    rows: [
      { region: "North", change: 60 },
      { region: "South", change: -20 },
    ],
  },
  limitations: [
    "Period coverage may be incomplete. Contributions do not establish business causes.",
  ],
  input_rows: 6,
  filtered_rows: 6,
  filters: [],
  periods: [],
  hash: "1234567890abcdef",
};
function run(status = "complete") {
  return {
    id: runId,
    chat_id: chat,
    investigation_id: runId,
    revision: 1,
    created_at: "2026-10-02T00:00:00Z",
    state: {
      status,
      question: "Compare revenue between January and February 2026",
      plan: {
        title: "Revenue change",
        steps: [
          {
            operation: "compare_periods",
            metric: "revenue",
            group_by: "region",
            date_column: "date",
            periods: [
              ["2026-01-01", "2026-01-31"],
              ["2026-02-01", "2026-02-28"],
            ],
          },
        ],
        clarification:
          status === "needs_clarification"
            ? "Which periods should I compare?"
            : null,
      },
      evidence: status === "needs_clarification" ? [] : [evidence],
      findings:
        status === "complete"
          ? [
              {
                evidence_id: "e1",
                observation: "North offsets the decline in South.",
              },
            ]
          : [],
      http_attempts: 2,
      provider_calls: [],
      events: [],
      error:
        status === "interpretation_unavailable"
          ? {
              category: "providers_unavailable",
              message: "gemini: quota; groq: outage",
              resumable: true,
            }
          : undefined,
    },
  };
}
const env = fs.existsSync(path.join(process.cwd(), ".env"))
  ? fs.readFileSync(path.join(process.cwd(), ".env"), "utf8")
  : "";
const supabaseUrl =
  env.match(/^NEXT_PUBLIC_SUPABASE_URL=["']?([^\s"']+)/m)?.[1] ||
  "https://placeholder.supabase.co";
const host = new URL(supabaseUrl).hostname.split(".")[0];
const user = {
  id: "11111111-1111-4111-8111-111111111111",
  email: "demo@example.test",
  aud: "authenticated",
  role: "authenticated",
  app_metadata: {},
  user_metadata: {},
  created_at: "2026-10-02T00:00:00Z",
};
const token = (suffix = "initial") =>
  `eyJhbGciOiJIUzI1NiJ9.${Buffer.from(JSON.stringify({ sub: user.id, exp: Math.floor(Date.now() / 1000) + 3600, role: "authenticated", suffix })).toString("base64url")}.signature`;
const session = (suffix = "initial") => ({
  access_token: token(suffix),
  refresh_token: "fake-refresh",
  token_type: "bearer",
  expires_in: 3600,
  expires_at: Math.floor(Date.now() / 1000) + 3600,
  user,
});
async function setup(
  page: Page,
  {
    hasDataset = true,
    status = "complete",
    signedIn = true,
    legacy = false,
  } = {},
) {
  let uploaded = hasDataset;
  let current = run(status);
  let empty = !hasDataset;
  const headers: string[] = [];
  if (signedIn)
    await page.addInitScript(
      ({ key, value }) => localStorage.setItem(key, JSON.stringify(value)),
      { key: `sb-${host}-auth-token`, value: session() },
    );
  await page.route("**/auth/v1/**", async (route) =>
    route.fulfill({
      json: route.request().url().includes("/user")
        ? user
        : session("refreshed"),
    }),
  );
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    headers.push(route.request().headers()["authorization"] || "");
    if (url.pathname === "/api/chats") {
      if (route.request().method() === "POST")
        return route.fulfill({ json: { id: chat } });
      return route.fulfill({
        json: [
          {
            id: chat,
            title: "Revenue workspace",
            user_id: user.id,
            status: "active",
            created_at: "2026-10-02T00:00:00Z",
            updated_at: "2026-10-02T00:00:00Z",
          },
        ],
      });
    }
    if (url.pathname === `/api/chat/${chat}`)
      return route.fulfill({
        json: {
          chat: { id: chat },
          dataset: uploaded ? dataset : null,
          messages: legacy
            ? [
                {
                  id: "legacy",
                  chat_id: chat,
                  role: "assistant",
                  content: "Previously computed total: 640",
                  metadata: {answer:'Previously computed total: 640',insight:'',anomalies:[],follow_ups:[],confidence:'high',code:'',result:640,result_type:'scalar',mode:'executor',rag_context_used:[],retry_count:0,plot_base64:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII='},
                  sequence_number: 1,
                  created_at: "2026-10-02T00:00:00Z",
                },
              ]
            : [],
        },
      });
    if (url.pathname === "/api/upload") {
      uploaded = true;
      return route.fulfill({ json: { dataset } });
    }
    if (url.pathname.endsWith("/profile"))
      return route.fulfill({ json: profile });
    if (url.pathname === "/api/investigations") {
      if (route.request().method() === "POST") {
        empty = false;
        current = run("complete");
        return route.fulfill({ status: 202, json: current });
      }
      return route.fulfill({ json: empty ? [] : [current] });
    }
    if (url.pathname.endsWith("/resume")) {
      current = run("queued");
      setTimeout(() => (current = run("complete")), 100);
      return route.fulfill({ status: 202, json: current });
    }
    if (url.pathname.endsWith("/download"))
      return route.fulfill({
        body: "region,change\nNorth,60\nSouth,-20",
        contentType: "text/csv",
      });
    return route.fulfill({ json: current });
  });
  return { headers };
}

test("sign in, create workspace, upload and inspect profile", async ({
  page,
}) => {
  await setup(page, { hasDataset: false, signedIn: false });
  await page.goto("/login");
  await page.getByLabel("Email").fill("demo@example.test");
  await page.getByLabel("Password").fill("not-a-real-password");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page).toHaveURL(/\/chat$/);
  await page.getByRole("button", { name: "Create a workspace" }).click();
  await page.getByRole("button", { name: "Upload a CSV" }).click();
  await page
    .getByLabel("Choose your CSV file")
    .setInputFiles("../samples/revenue.csv");
  await page.getByRole("button", { name: "Upload and profile" }).click();
  await expect(
    page.getByText("revenue.csv", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Dataset columns" }),
  ).toBeVisible();
  await page
    .getByLabel("Ask a question about your dataset")
    .fill("Compare January and February revenue");
  await page.getByRole("button", { name: "Investigate", exact: true }).click();
  await expect(
    page.getByText("North offsets the decline in South."),
  ).toBeVisible();
});
test("inspect measured evidence, download and narrow layout", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/chat/${chat}`);
  await expect(
    page.getByRole("heading", { name: "Revenue change" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "View e1" }).click();
  await expect(
    page.getByRole("complementary", { name: "Evidence details" }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download e1 CSV" }).click();
  expect((await download).suggestedFilename()).toBe("evidence-e1.csv");
  await page.screenshot({
    path: "../output/playwright/workspace-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("button", { name: "Investigate", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../output/playwright/workspace-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Reload workspace" }).focus();
  await expect(
    page.getByRole("button", { name: "Reload workspace" }),
  ).toBeFocused();
});
test("answer clarification and resume actual saved state", async ({ page }) => {
  await setup(page, { status: "needs_clarification" });
  await page.goto(`/chat/${chat}`);
  await page
    .getByLabel("Which periods should I compare?")
    .fill("January and February 2026");
  await page.getByRole("button", { name: "Continue investigation" }).click();
  await expect(
    page.getByText("North offsets the decline in South."),
  ).toBeVisible();
});
test("quota failure retains measurements and resumes interpretation", async ({
  page,
}) => {
  await setup(page, { status: "interpretation_unavailable" });
  await page.goto(`/chat/${chat}`);
  await expect(
    page.getByText("Your computed evidence is saved."),
  ).toBeVisible();
  await expect(page.getByText("gemini: quota; groq: outage")).toBeVisible();
  await page.getByRole("button", { name: "Resume saved work" }).click();
  await expect(
    page.getByText("North offsets the decline in South."),
  ).toBeVisible();
});
test("session refresh supplies new bearer token; legacy messages remain readable", async ({
  page,
}) => {
  const { headers } = await setup(page, { legacy: true });
  await page.goto(`/chat/${chat}`);
  await page.getByText("Previous conversation · read only").click();
  await expect(page.getByText("Previously computed total: 640")).toBeVisible();
  await expect(page.getByAltText("Generated visualization")).toBeVisible();
  const refreshed = session("refreshed");
  await page.evaluate(
    ({ key, value }) => {
      localStorage.setItem(key, JSON.stringify(value));
      const channel = new BroadcastChannel(key);
      channel.postMessage({ event: "TOKEN_REFRESHED", session: value });
      channel.close();
    },
    { key: `sb-${host}-auth-token`, value: refreshed },
  );
  await expect
    .poll(() => headers.includes("Bearer " + refreshed.access_token))
    .toBeTruthy();
});

test("edit period windows and filters without replacing saved results", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/chat/${chat}`);
  await page.getByText("Calculation plan · 1 step", { exact: true }).click();
  await page
    .getByRole("button", { name: "Edit plan and run a new revision" })
    .click();
  await page.getByLabel("Current end").fill("2026-02-20");
  await page.getByRole("button", { name: "Add filter", exact: true }).click();
  await page.getByRole("combobox", { name: "Column", exact: true }).selectOption("revenue");
  await page.getByRole("combobox", { name: "Operator", exact: true }).selectOption("gte");
  await page.getByLabel("Value", { exact: true }).fill("100");
  const outgoing = page.waitForRequest(
    (r) => r.url().endsWith("/api/investigations") && r.method() === "POST",
  );
  await page.getByRole("button", { name: "Run edited plan" }).click();
  const body = (await outgoing).postDataJSON();
  expect(body.parent_id).toBe(runId);
  expect(body.plan.steps[0].periods[1][1]).toBe("2026-02-20");
  expect(body.plan.steps[0].filters[0]).toEqual({
    column: "revenue",
    operator: "gte",
    value: 100,
  });
});
