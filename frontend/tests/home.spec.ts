import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

test("explore the sample, download its source and enter the workspace", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Less guesswork.More evidence.",
  );
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to content" }),
  ).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(
    page.getByRole("img", {
      name: "January revenue 300. February revenue 340. An increase of 40.",
    }),
  ).toBeVisible();
  const drivers = page.getByRole("button", {
    name: "The drivers",
    exact: true,
  });
  await drivers.focus();
  await page.keyboard.press("Enter");
  await expect(drivers).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("img", {
      name: "North adds 60. South subtracts 20. Net revenue change is plus 40.",
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "The caveat", exact: true }).click();
  await expect(
    page.getByText(
      "A few recorded dates cannot establish complete monthly coverage.",
    ),
  ).toBeVisible();
  const downloadEvent = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Check it yourself. Download the CSV" })
    .click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe("revenue.csv");
  const file = await download.path();
  expect(fs.readFileSync(file!, "utf8")).toBe(
    fs.readFileSync(path.resolve("../samples/revenue.csv"), "utf8"),
  );
  await page.getByRole("button", { name: "The change", exact: true }).click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: "../output/playwright/home-desktop.png",
    fullPage: true,
    animations: "disabled",
  });
  for (const width of [320, 390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  expect(
    await page
      .locator(".hero-copy")
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
  await page.screenshot({
    path: "../output/playwright/home-mobile.png",
    fullPage: true,
    animations: "disabled",
  });
  await page
    .getByRole("link", { name: "Open your workspace", exact: true })
    .first()
    .click();
  await expect(page).toHaveURL(/\/login/);
  expect(errors).toEqual([]);
});
