import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { TelemetryEventSchema } from "./contracts";

describe("cross-language contract validation", () => {
  it("successfully parses the gateway telemetry_events.json fixture", () => {
    const fixturePath = resolve(
      __dirname,
      "../../../gateway/tests/fixtures/telemetry_events.json",
    );
    const rawData = readFileSync(fixturePath, "utf-8");
    const json = JSON.parse(rawData) as unknown[];

    expect(Array.isArray(json)).toBe(true);
    expect(json.length).toBeGreaterThan(0);

    for (const item of json) {
      const parsed = TelemetryEventSchema.safeParse(item);
      if (!parsed.success) {
        console.error("Zod parse error:", parsed.error);
      }
      expect(parsed.success).toBe(true);
    }
  });
});

