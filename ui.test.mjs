// node --test            （計算機の上のコード前半は numpy のある python3 で試す。GRAPHY_TEST_PYTHON で変えられる）
// モデル本体（torch / transformers）は使わない。確かめるのは「画像の準備・環境の判定・画面の往復」。
// モデルの実行は実機（README の検証）で確かめる。
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { DISCLAIMER, PY, PY_PREP, REVISION, TRANSFORMERS, activate, buildScript, explain, failureText, targetProblem } from "./ui.js";

const PYTHON = process.env.GRAPHY_TEST_PYTHON || "python3";
const py = (code, cwd) => spawnSync(PYTHON, ["-I", "-c", code], { cwd, encoding: "utf8" });

test("targetProblem: 通る画像と止める画像", () => {
  const ok = { kind: "image", modality: "DX", sliceCount: 1 };
  assert.equal(targetProblem(ok), null);
  assert.equal(targetProblem({ ...ok, modality: "cr" }), null);
  assert.match(targetProblem(null), /タイル/);
  assert.match(targetProblem({ ...ok, kind: "video" }), /動画/);
  assert.match(targetProblem({ ...ok, modality: "CT" }), /CR\/DX.*CT/);
  assert.match(targetProblem({ ...ok, modality: "MG" }), /CR\/DX.*MG/);
  assert.match(targetProblem({ ...ok, sliceCount: 3 }), /3 枚/);
});

test("buildScript: 固定の 3 値だけが入り、本体の上限と埋め込み検査の目安に収まる", () => {
  const s = buildScript();
  assert.ok(s.startsWith("GRAPHY = __import__('json').loads("));
  assert.ok(s.includes(REVISION) && s.includes(TRANSFORMERS));
  assert.ok(s.length < 64 * 1024);
  assert.ok(!/[A-Za-z0-9+/=]{200}/.test(s), "base64 らしい長い連なりがある");
  assert.ok(!/[\d.,\s-]{400}/.test(s.replace(/\s{2,}/g, " ")), "数字の長い並びがある");
  assert.ok(/revision=rev/.test(PY) && /trust_remote_code=True/.test(PY), "重みとコードを revision で固定して読む");
});

test("PY 全体は Python として構文が正しい", () => {
  const r = spawnSync(PYTHON, ["-I", "-c", "import ast,sys; ast.parse(sys.stdin.read())"], { input: PY, encoding: "utf-8" });
  assert.equal(r.status, 0, r.stderr);
});

function withNpz(meta, volume, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cxr-"));
  try {
    fs.mkdirSync(path.join(dir, "inputs"));
    const make = py(`
import json, numpy as np
rows = json.loads(${JSON.stringify(JSON.stringify(volume))})
v = np.array(rows['v'], dtype=np.float32).reshape(rows['shape'])
np.savez('inputs/0.npz', volume=v, spacing=np.array([np.nan]*3), meta=np.array(0),
         **{'meta.json': np.frombuffer(json.dumps(json.loads(${JSON.stringify(JSON.stringify(meta))})).encode(), dtype=np.uint8)})
`, dir);
    assert.equal(make.status, 0, make.stderr);
    return fn(dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const prepare = (dir) => py(`
${PY_PREP}
import numpy as np
g, info = load_image('inputs/0.npz')
print(json.dumps({'dtype': str(g.dtype), 'shape': list(g.shape), 'min': int(g.min()), 'max': int(g.max()),
                  'first': int(g[0, 0]), 'last': int(g[-1, -1]), 'info': info}))
`, dir);

const ramp = { shape: [1, 2, 2], v: [0, 100, 200, 400] };

test("load_image: 窓なしは最小〜最大で 8bit にする（正例）", () => {
  withNpz({ modality: "DX", photometricInterpretation: "MONOCHROME2" }, ramp, (dir) => {
    const r = prepare(dir);
    assert.equal(r.status, 0, r.stderr);
    const o = JSON.parse(r.stdout);
    assert.deepEqual([o.dtype, o.min, o.max, o.first, o.last], ["uint8", 0, 255, 0, 255]);
    assert.equal(o.info.window, "minmax");
  });
});

test("load_image: DICOM の窓を当てる。窓の外は飽和する", () => {
  // c=100 w=101 → 50〜150 が 0〜255。値 0 は 0、200 と 400 は 255
  withNpz({ modality: "CR", photometricInterpretation: "MONOCHROME2", windowCenter: [100], windowWidth: [101] }, ramp, (dir) => {
    const o = JSON.parse(prepare(dir).stdout);
    assert.equal(o.info.window, "dicom");
    assert.deepEqual([o.first, o.last], [0, 255]);
  });
  withNpz({ modality: "CR", photometricInterpretation: "MONOCHROME2", windowCenter: [100], windowWidth: [101] },
    { shape: [1, 1, 3], v: [100, 100, 100] }, (dir) => {
      const o = JSON.parse(prepare(dir).stdout);
      assert.ok(o.first > 120 && o.first < 135, `窓の中心は中間の明るさのはず: ${o.first}`);
    });
});

test("load_image: MONOCHROME1 は反転する（負例との対）", () => {
  withNpz({ modality: "DX", photometricInterpretation: "MONOCHROME1" }, ramp, (dir) => {
    const o = JSON.parse(prepare(dir).stdout);
    assert.deepEqual([o.first, o.last], [255, 0]);
    assert.equal(o.info.monochrome1, true);
  });
});

test("load_image: 使えない画像は not-applicable で止まる（CT・複数枚・一定値）", () => {
  const cases = [
    [{ modality: "CT" }, ramp, /not-applicable: .*CT/],
    [{ modality: "MG" }, ramp, /not-applicable: .*MG/],
    [{ modality: "DX" }, { shape: [2, 1, 2], v: [0, 1, 2, 3] }, /not-applicable: .*2 枚/],
    [{ modality: "DX" }, { shape: [1, 1, 2], v: [5, 5] }, /not-applicable: .*一定/],
  ];
  for (const [meta, vol, re] of cases) {
    withNpz(meta, vol, (dir) => {
      const r = prepare(dir);
      assert.notEqual(r.status, 0);
      assert.match(r.stderr, re);
    });
  }
});

test("env_problem: 足りない・版違いは止め、揃っていれば通す", () => {
  const run = (installed) => JSON.parse(py(`
${PY_PREP}
need = {'torch': None, 'transformers': '${TRANSFORMERS}'}
print(json.dumps(env_problem(${JSON.stringify(JSON.stringify(installed))} and json.loads(${JSON.stringify(JSON.stringify(installed))}), need)))
`).stdout);
  assert.equal(run({ torch: "2.14.1+cpu", transformers: TRANSFORMERS }), null);
  assert.match(run({ torch: null, transformers: TRANSFORMERS }), /^missing-packages: .*torch/);
  assert.match(run({ torch: "2", transformers: null }), new RegExp(`^missing-packages: .*transformers==${TRANSFORMERS.replaceAll(".", "\\.")}`));
  const bad = run({ torch: "2", transformers: "4.57.6" });
  assert.match(bad, /^unsupported-version: transformers 4\.57\.6.*4\.46\.3/);
  assert.ok(!/torch /.test(bad.split("。")[0]), "版の指定が無い torch は版違いに数えない");
});

test("failureText / explain", () => {
  assert.equal(failureText("missing-packages: x を入れてください"), "x を入れてください");
  assert.equal(failureText("unsupported-version: y"), "y");
  assert.match(failureText("KeyError: 'a'"), /^失敗: KeyError/);
  assert.match(explain("dataset-local-only"), /この PC の計算機/);
  assert.equal(explain("unknown-code"), "unknown-code");
});

// --- 画面（最小の偽 DOM） ---
class El {
  constructor(tag) { this.tag = tag; this.kids = []; this.style = {}; this.dataset = {}; this.handlers = {}; this.textContent = ""; }
  append(...c) { this.kids.push(...c); }
  replaceChildren(...c) { this.kids = c; }
  addEventListener(n, f) { this.handlers[n] = f; }
  text() { return this.textContent + this.kids.map((k) => (k instanceof El ? k.text() : String(k))).join(""); }
  find(id) { if (this.dataset.testid === id) return this; for (const k of this.kids) { const f = k instanceof El ? k.find(id) : null; if (f) return f; } return null; }
}
function fakeEnv(outcome, target = { kind: "image", modality: "DX", sliceCount: 1, studyUid: "S", seriesUid: "R", seriesLabel: "CXR PA" }) {
  globalThis.window = {};
  globalThis.document = { createElement: (t) => new El(t) };
  const calls = [], notes = [];
  const container = new El("div");
  const host = {
    getTargets: () => [target],
    notify: (m) => notes.push(m),
    openWindow: () => ({ container }),
    compute: { runJob: async (opts) => { calls.push(opts); return outcome; } },
  };
  return { host, calls, notes, container };
}
const okOutcome = (rep) => ({ ok: true, status: "ok", readFile: async () => new TextEncoder().encode(JSON.stringify(rep)) });

test("画面: 免責は常に出ており、実行は local へ npz 1 件で渡す。結果は所見と印象に分かれる", async () => {
  const rep = { findings: "No pneumothorax.", impression: "Normal.", model: "m", revision: "84dfcba8aaaa", device: "cpu", seconds: 61.2, window: "minmax" };
  const { host, calls, container } = fakeEnv(okOutcome(rep));
  await activate(host);
  assert.equal(container.find("cxr-disclaimer").textContent.includes(DISCLAIMER), true);
  await container.find("cxr-run").handlers.click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].endpointId, "local");
  assert.deepEqual(calls[0].inputs, [{ studyUid: "S", seriesUid: "R", format: "npz" }]);
  assert.equal(calls[0].script, buildScript());
  assert.equal(container.find("cxr-findings").value, "No pneumothorax.");
  assert.equal(container.find("cxr-impression").value, "Normal.");
  assert.equal(container.find("cxr-status").textContent, "できました");
  assert.equal(container.find("cxr-disclaimer").textContent.includes(DISCLAIMER), true, "結果が出た後も免責は残る");
  assert.match(container.find("cxr-meta").text(), /84dfcba8/);
});

test("画面: 版違いは理由を出して止まり、結果は出さない（負例）", async () => {
  const out = { ok: true, status: "error", errorName: "RuntimeError", errorValue: "unsupported-version: transformers 4.57.6（動作を確かめた版は 4.46.3）", readFile: async () => null };
  const { host, container } = fakeEnv(out);
  await activate(host);
  await container.find("cxr-run").handlers.click();
  assert.match(container.find("cxr-status").textContent, /^transformers 4\.57\.6/);
  assert.equal(container.find("cxr-result").style.display, "none");
  assert.equal(globalThis.window.__cxrReportState.phase, "failed");
});

test("画面: 本体の拒否（旧版の本体）は理由つきで出る", async () => {
  const { host, container } = fakeEnv({ ok: false, error: "burnin-risky-modality" });
  await activate(host);
  await container.find("cxr-run").handlers.click();
  assert.match(container.find("cxr-status").textContent, /更新してください（burnin-risky-modality）/);
});

test("画面: CT のタイルでは窓を開かず、理由を通知する", async () => {
  const { host, notes, container } = fakeEnv(null, { kind: "image", modality: "CT", sliceCount: 100 });
  await activate(host);
  assert.equal(container.kids.length, 0);
  assert.match(notes[0], /CR\/DX/);
});
