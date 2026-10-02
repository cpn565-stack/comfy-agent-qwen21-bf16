import { promises as fs } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runColabKit } from "../src/cli/colab.js";
import { buildColabSuggestPayload, loadColabCatalogFile } from "../src/colab/catalog.js";

const root = process.cwd();
const kitRoot = path.join(root, "scripts/colab/qwen_image_2_1_bf16");

afterEach(() => vi.restoreAllMocks());

describe("Qwen Image 2.1 non-quantized BF16", () => {
  it("selects the BF16 kit and its dedicated launcher", async () => {
    vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    const payload = await runColabKit("qwen_image_2_1_bf16", { json: true });
    expect(payload.kit).toMatchObject({
      status: "starter",
      gpu: { minimum: "A100", recommended: "A100" },
      composable: false,
    });
    expect(payload.kit.gpu.verified).toBeUndefined();
    expect(payload.paths.launcher).toBe(path.join(kitRoot, "02_start_comfyui.py"));
    const catalog = await loadColabCatalogFile(path.join(root, "scripts/colab/catalog.yaml"));
    const suggestions = buildColabSuggestPayload(catalog, {
      goal: "Qwen Image 2.1 BF16",
      gpu: "A100",
    });
    expect(suggestions.suggestions[0].kit).toBe("qwen_image_2_1_bf16");
  });

  it("keeps all workflows on BF16 weights and lossless caches", async () => {
    const catalog = await loadColabCatalogFile(path.join(root, "scripts/colab/catalog.yaml"));
    const kit = catalog.kits.find((item) => item.name === "qwen_image_2_1_bf16")!;
    for (const workflow of kit.workflows) {
      const graph = JSON.parse(await fs.readFile(path.join(kitRoot, workflow.file), "utf8"));
      expect(graph["1"].inputs).toMatchObject({
        unet_name: "qwen_image_2.1_bf16.safetensors",
        weight_dtype: "default",
      });
      expect(graph["2"].inputs.clip_name).toBe("qwen3vl_8b_bf16.safetensors");
      expect(graph["3"].inputs.vae_name).toBe("qwen_image_2.1_vae_bf16.safetensors");
      if (workflow.task === "image_edit") {
        expect(graph["6"].inputs).toMatchObject({ device: "cpu", dtype: "default" });
      }
      expect(JSON.stringify(graph)).not.toMatch(/int8|int4|fp8|gguf|convrot/i);
    }
  });

  it("does not substitute a quantized kit when BF16 or A100 requirements cannot be met", async () => {
    const catalog = await loadColabCatalogFile(path.join(root, "scripts/colab/catalog.yaml"));
    for (const goal of [
      "Qwen Image 2.1 非量化",
      "qwen_image_2_1_bf16",
      "Qwen Image 2.1 unquantized",
    ]) {
      const payload = buildColabSuggestPayload(catalog, { goal, gpu: "A100" });
      expect(payload.suggestions.length).toBeGreaterThan(0);
      expect(payload.suggestions.every((item) => item.kit === "qwen_image_2_1_bf16")).toBe(true);
    }
    for (const gpu of ["L4", "T4", "G4"] as const) {
      const payload = buildColabSuggestPayload(catalog, { goal: "Qwen Image 2.1 BF16", gpu });
      expect(payload.suggestions).toEqual([]);
      expect(payload.alternatives.every((item) => item.kit === "qwen_image_2_1_bf16")).toBe(true);
      expect(payload.alternatives[0].unmet_requirements).toContain("requires_gpu:A100");
    }
  });

  it("rejects unsupported hardware before setup and checks BF16 file headers", () => {
    const result = spawnSync("python3", [path.join(root, "test/qwen21_bf16_preflight.py")], {
      encoding: "utf8",
    });
    expect(result.status, result.stderr || result.stdout).toBe(0);
  });
});
