// node --test            （計算機の上のコード前半は numpy のある python3 で試す。GRAPHY_TEST_PYTHON で変えられる）
// モデル本体（torch / transformers）は使わない。確かめるのは「画像の準備・環境の判定・画面の往復」。
// モデルの実行は実機（README の検証）で確かめる。
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { DISCLAIMER, PY, PY_PREP, REVISION, TRANSFORMERS, activate, buildScript, explain, failureText, parseNpyF32, targetProblem, upsampleCam } from "./ui.js";

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

test("targetProblem: CR/DX 以外のモダリティはすべて理由つきで止める（CT・MR・PT・US ほか）", () => {
  for (const m of ["CT", "MR", "PT", "US", "XA", "NM", "MG", "RF", "SC", "OT", ""]) {
    assert.match(targetProblem({ kind: "image", modality: m, sliceCount: 1 }) ?? "", /CR\/DX/, m);
  }
  for (const m of ["CR", "DX", "dx"]) assert.equal(targetProblem({ kind: "image", modality: m, sliceCount: 1 }), null, m);
});

test("免責と説明文に『胸部の正面像』の限定が書かれている", () => {
  assert.match(DISCLAIMER, /胸部の正面像（PA\/AP）の 1 枚だけ/);
  assert.match(DISCLAIMER, /部位と撮影方向を判定しません/);
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
    ...["CT", "MR", "PT", "US", "XA", "NM"].map((m) => [{ modality: m }, ramp, new RegExp(`not-applicable: .*${m}`)]),
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


// --- Grad-CAM ---
test("crop_box: 正方形はそのまま、横長・縦長は中央の正方形（切り落とした端が出る）", () => {
  const run = (rows, cols) => JSON.parse(py(`
${PY_PREP}
print(json.dumps(crop_box(${rows}, ${cols}, 384)))
`).stdout);
  assert.deepEqual(run(1024, 1024), { x0: 0, y0: 0, w: 1024, h: 1024 });
  assert.deepEqual(run(2000, 1000), { x0: 0, y0: 500, w: 1000, h: 1000 });
  assert.deepEqual(run(1000, 2000), { x0: 500, y0: 0, w: 1000, h: 1000 });
});

test("sentence_spans: '.' で閉じ、続く sep / eos は同じ文に含める。全トークンを 1 度ずつ覆う", () => {
  // ids: bos a b . c . SEP d . EOS（dot=9 sep=3 eos=2）
  const ids = [101, 5, 6, 9, 7, 9, 3, 8, 9, 2];
  const r = JSON.parse(py(`
${PY_PREP}
print(json.dumps(sentence_spans(${JSON.stringify(ids)}, 9, (3, 2), ${ids.length})))
`).stdout);
  assert.deepEqual(r, [[0, 3], [3, 6], [6, 9]]);
  assert.equal(r[0][0], 0);
  for (let i = 1; i < r.length; i++) assert.equal(r[i][0], r[i - 1][1], "隙間も重なりも無い");
  assert.equal(r.at(-1)[1], ids.length - 1);
  // '.' で終わらない末尾も 1 文になる
  const tail = JSON.parse(py(`
${PY_PREP}
print(json.dumps(sentence_spans([101, 5, 9, 7, 8], 9, (3, 2), 5)))
`).stdout);
  assert.deepEqual(tail, [[0, 2], [2, 4]]);
});

test("parseNpyF32: numpy が書いた float32 を読む。別の型は拒否（負例）", () => {
  const dump = (dtype) => spawnSync(PYTHON, ["-I", "-c", `import numpy as np,sys,io;b=io.BytesIO();np.save(b,np.arange(24,dtype=np.${dtype}).reshape(2,3,4));sys.stdout.buffer.write(b.getvalue())`]).stdout;
  const r = parseNpyF32(new Uint8Array(dump("float32")));
  assert.deepEqual(r.shape, [2, 3, 4]);
  assert.deepEqual([r.data[0], r.data[5], r.data[23]], [0, 5, 23]);
  assert.throws(() => parseNpyF32(new Uint8Array(dump("float64"))), /float32/);
});

test("upsampleCam: 正方形は全面・横長の画像は切り落とした端が NaN・角と中央の値が合う", () => {
  const cam = Float32Array.from([0, 1, 2, 3]); // 2×2: 左上 0・右上 1・左下 2・右下 3
  const sq = upsampleCam(cam, [2, 2], { x0: 0, y0: 0, w: 8, h: 8 }, 8, 8);
  assert.ok(!sq.some(Number.isNaN));
  assert.equal(sq[0], 0);
  assert.equal(sq[7], 1);
  assert.equal(sq[7 * 8], 2);
  assert.equal(sq[63], 3);
  // 縦 4・横 8 の画像で、中央の 4×4（x0=2）だけを見た場合: 左右の端 2 列ずつが NaN
  const wide = upsampleCam(cam, [2, 2], { x0: 2, y0: 0, w: 4, h: 4 }, 4, 8);
  for (let y = 0; y < 4; y++) {
    assert.ok(Number.isNaN(wide[y * 8]) && Number.isNaN(wide[y * 8 + 1]) && Number.isNaN(wide[y * 8 + 6]) && Number.isNaN(wide[y * 8 + 7]));
    assert.ok(!Number.isNaN(wide[y * 8 + 2]) && !Number.isNaN(wide[y * 8 + 5]));
  }
  assert.equal(wide[2], 0);
  assert.equal(wide[5], 1);
  assert.equal(wide[3 * 8 + 2], 2);
  assert.equal(wide[3 * 8 + 5], 3);
  // 定数のマップは定数のまま
  const flat = upsampleCam(new Float32Array(9).fill(0.5), [3, 3], { x0: 0, y0: 0, w: 10, h: 10 }, 10, 10);
  assert.ok(flat.every((v) => v === 0.5));
});

function npyOf(shape, values) {
  const header0 = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}), }`;
  const pad = (64 - ((10 + header0.length + 1) % 64)) % 64;
  const header = header0 + " ".repeat(pad) + "\n";
  const out = new Uint8Array(10 + header.length + values.length * 4);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  const dv = new DataView(out.buffer);
  values.forEach((v, i) => dv.setFloat32(10 + header.length + i * 4, v, true));
  return out;
}

function camEnv(rep, camBytes) {
  const e = fakeEnv({ ok: true, status: "ok", readFile: async (n) => (n === "report.json" ? new TextEncoder().encode(JSON.stringify(rep)) : n === "cam.npy" ? camBytes : null) });
  const overlays = [], cleared = [];
  e.host.showOverlay = (tile, o) => { overlays.push({ tile, ...o }); return true; };
  e.host.clearOverlay = (tile) => cleared.push(tile);
  e.host.getTargets = () => [{ kind: "image", modality: "DX", sliceCount: 1, studyUid: "S", seriesUid: "R", seriesLabel: "CXR", tileId: "T1" }];
  return { ...e, overlays, cleared };
}
const baseRep = { findings: "F.", impression: "I.", model: "m", revision: "84dfcba8aaaa", device: "cpu", seconds: 1, window: "dicom", rows: 8, cols: 8 };

test("画面: 文を押すと、その文の cam が元画像の大きさで重なり、消すと外れる", async () => {
  const rep = { ...baseRep, camError: null, cam: { shape: [2, 2], crop: { x0: 0, y0: 0, w: 8, h: 8 }, gap: 0.0001,
    sentences: [{ section: "findings", text: "Clear lungs.", strength: 1 }, { section: "impression", text: "Normal.", strength: 0.2 }] } };
  const { host, container, overlays, cleared } = camEnv(rep, npyOf([2, 2, 2], [0, 1, 2, 3, 3, 2, 1, 0]));
  await activate(host);
  await container.find("cxr-run").handlers.click();
  assert.match(container.find("cxr-cam-sentence-1").textContent, /印象｜Normal\.（強さ 0\.20）/);
  assert.match(container.find("cxr-cam-legend").text(), /効いていない.*強く効いた/, "色の凡例が出る");
  container.find("cxr-cam-sentence-1").handlers.click();
  assert.equal(overlays.length, 1);
  assert.equal(overlays[0].tile, "T1");
  assert.equal(overlays[0].rows, 8);
  assert.equal(overlays[0].cols, 8);
  assert.equal(overlays[0].data.length, 64);
  assert.equal(overlays[0].data[0], 3, "2 文目のマップの左上は 3（1 文目の 0 ではない）");
  assert.equal(globalThis.window.__cxrReportState.cam.shown, 1);
  container.find("cxr-cam-clear").handlers.click();
  assert.deepEqual(cleared.slice(-1), ["T1"]);
  assert.equal(globalThis.window.__cxrReportState.cam.shown, null);
});

test("画面: Grad-CAM が作れなかったときは理由を出し、レポートは出す（負例）", async () => {
  const rep = { ...baseRep, cam: null, camError: "RuntimeError: gradcam-mismatch: 再計算した確率が生成時と合いません（差 0.3000）" };
  const { host, container, overlays } = camEnv(rep, null);
  await activate(host);
  await container.find("cxr-run").handlers.click();
  assert.match(container.find("cxr-cam-error").text(), /作れませんでした: .*gradcam-mismatch/);
  assert.equal(container.find("cxr-findings").value, "F.", "レポートは出る");
  assert.equal(container.find("cxr-cam-sentence-0"), null);
  assert.equal(overlays.length, 0);
});

test("画面: 本体が重ねを断ったら（大きさ不一致）理由を出す", async () => {
  const rep = { ...baseRep, camError: null, cam: { shape: [2, 2], crop: { x0: 0, y0: 0, w: 8, h: 8 }, sentences: [{ section: "findings", text: "A.", strength: 1 }] } };
  const { host, container } = camEnv(rep, npyOf([1, 2, 2], [0, 1, 2, 3]));
  host.showOverlay = () => false;
  await activate(host);
  await container.find("cxr-run").handlers.click();
  container.find("cxr-cam-sentence-0").handlers.click();
  assert.match(container.find("cxr-status").textContent, /重ねられませんでした/);
  assert.equal(globalThis.window.__cxrReportState.cam.error, "overlay-refused");
});
