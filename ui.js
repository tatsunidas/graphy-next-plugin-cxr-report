/// <reference path="./graphy-plugin.d.ts" />
// @ts-check
/*
 * 胸部X線 1 枚から所見・印象の下書きを作る GRAPHY-Next の公式プラグイン。モデルは CXRMate（aehrc・Apache-2.0）。
 *
 * 流れ（「実行」1 回・同意 1 回）:
 *   1. 表示中の CR/DX 1 枚を、本体が匿名化した npz として **この PC の計算機（endpointId "local"）だけ**へ渡す。
 *      焼き込み文字は塗らないので、外部の計算機（Colab など）へは本体が送らせない（dataset-local-only）
 *   2. 計算機の上で窓を当てて 8bit にし、CXRMate（Hugging Face）で生成する。重みは同梱しない
 *   3. 所見（findings）と印象（impression）を、免責つきで見せる
 *
 * 計算機の上のコード（PY）は、同意画面に全文が出る。モデルの版と revision だけが差し込まれる。
 */

export const MODEL = "aehrc/cxrmate-single-tf";
/** trust_remote_code で読む modelling_single.py ごと、この revision に固定する（動作を確かめたコミット） */
export const REVISION = "84dfcba8125c9b9296bfc1ee317749788b6b59b4";
/** 4.57.6 では generate が落ちる（cache_position が None）。動作を確かめた版だけを許す */
export const TRANSFORMERS = "4.46.3";

export const DISCLAIMER =
  "研究用の下書きです。診断・治療の判断に使わないでください。生成された文は誤りを含み、画像にない内容" +
  "（撮影方向・年齢・性別・過去画像との比較など）が入ることがあります。" +
  "対象は胸部の正面像（PA/AP）の 1 枚だけです。側面像や胸部以外の部位の画像でも実行はできますが、結果は無意味です" +
  "（このプラグインは部位と撮影方向を判定しません）。";

// ---------------------------------------------------------------------------
// 計算機の上で走る Python。先頭に GRAPHY（model・revision・transformers）が足される。
// 前半（PY_PREP）は numpy だけで試験できる。後半（PY_MODEL）が torch / transformers を使う。
// ---------------------------------------------------------------------------
export const PY_PREP = String.raw`
import json, os, sys, time
from importlib import metadata as _md

T0 = time.time()


def stage(name, p):
    print('__progress__', p, name, flush=True)


def version_of(dist):
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return None


def env_problem(installed, need):
    """入っていない・版が違うパッケージがあれば理由（先頭が missing-packages: / unsupported-version:）。無ければ None"""
    missing = [d for d in need if installed.get(d) is None]
    if missing:
        return 'missing-packages: この計算機に ' + ' '.join(missing) + ' を入れてください（pip install ' + ' '.join(
            d + ('==' + need[d] if need[d] else '') for d in missing) + '）'
    bad = [d for d, want in need.items() if want and installed[d].split('+')[0] != want]
    if bad:
        return 'unsupported-version: ' + '、'.join(
            d + ' ' + installed[d] + '（動作を確かめた版は ' + need[d] + '）' for d in bad
        ) + '。環境を書き換えないので、専用の環境で pip install ' + ' '.join(d + '==' + need[d] for d in bad) + ' を入れてください'
    return None


def load_image(path):
    """npz の volume[1, y, x] を窓で 8bit にする。使えない画像は not-applicable: で止める"""
    import numpy as np
    z = np.load(path)
    meta = json.loads(bytes(z['meta.json']))
    if meta.get('modality') not in ('CR', 'DX'):
        raise RuntimeError('not-applicable: 胸部X線（CR/DX）の 1 枚だけに使えます（このシリーズは ' + str(meta.get('modality')) + '）')
    vol = z['volume']
    if vol.ndim != 3 or vol.shape[0] != 1:
        raise RuntimeError('not-applicable: 1 枚の画像だけに使えます（このシリーズは ' + str(vol.shape[0]) + ' 枚・多フレームを含む）')
    a = vol[0].astype(np.float32)
    wc, ww = meta.get('windowCenter'), meta.get('windowWidth')
    func = meta.get('voiLutFunction') or 'LINEAR'
    window = 'minmax'
    lo, hi = float(a.min()), float(a.max())
    if wc and ww and func == 'LINEAR' and float(ww[0]) > 1:
        c, w = float(wc[0]), float(ww[0])
        lo, hi, window = c - 0.5 - (w - 1) / 2, c - 0.5 + (w - 1) / 2, 'dicom'
    if not hi > lo:
        raise RuntimeError('not-applicable: 画素値が一定で、画像として使えません')
    y = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if meta.get('photometricInterpretation') == 'MONOCHROME1':
        y = 1.0 - y
    return (y * 255.0 + 0.5).astype(np.uint8), {'window': window, 'rows': int(a.shape[0]), 'cols': int(a.shape[1]),
                                                 'monochrome1': meta.get('photometricInterpretation') == 'MONOCHROME1'}
`;

export const PY_MODEL = String.raw`

stage('setup', 0.02)
NEED = {'torch': None, 'torchvision': None, 'transformers': GRAPHY['transformers'], 'pillow': None, 'numpy': None}
problem = env_problem({d: version_of(d) for d in NEED}, NEED)
if problem:
    raise RuntimeError(problem)
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
gray, info = load_image('inputs/0.npz')
img = Image.fromarray(gray).convert('RGB')
stage('image', 0.1)

name, rev = GRAPHY['model'], GRAPHY['revision']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = transformers.AutoTokenizer.from_pretrained(name, revision=rev)
stage('tokenizer', 0.2)
model = transformers.AutoModel.from_pretrained(name, revision=rev, trust_remote_code=True).eval().to(device)
fe = transformers.AutoImageProcessor.from_pretrained(name, revision=rev)
stage('model', 0.5)
side = fe.size['shortest_edge']
prep = T.Compose([T.Resize(side), T.CenterCrop(side), T.ToTensor(), T.Normalize(fe.image_mean, fe.image_std)])
px = prep(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = model.generate(
        pixel_values=px, special_token_ids=[tok.sep_token_id], bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
        use_cache=True, max_length=256, num_beams=4)
findings, impression = model.split_and_decode_sections(out.sequences, [tok.sep_token_id, tok.eos_token_id], tok)
stage('generated', 0.95)
json.dump({
    'findings': findings[0], 'impression': impression[0],
    'model': name, 'revision': rev, 'device': device,
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'seconds': round(time.time() - T0, 1), **info,
}, open('outputs/report.json', 'w', encoding='utf-8'), ensure_ascii=False)
stage('done', 1.0)
`;

export const PY = PY_PREP + PY_MODEL;

/** 計算機に送るコード。差し込むのは固定の 3 値だけ。 */
export function buildScript() {
  const cfg = { model: MODEL, revision: REVISION, transformers: TRANSFORMERS };
  return `GRAPHY = __import__('json').loads(${JSON.stringify(JSON.stringify(cfg))})\n` + PY;
}

/** 本体がデータを作らなかった・送らなかった理由（よく出るものだけ）。 */
const REFUSALS = {
  "burnin-risky-modality": "この本体は、胸部X線を計算機へ渡す機能を持っていません。新しい版の GRAPHY-Next に更新してください",
  "dataset-local-only": "胸部X線はこの PC の計算機にしか渡せません（外部の計算機は選べません）",
  "npz-too-large": "画像が大きすぎます",
  "permission-denied": "このプラグインに計算機を使う許可がありません",
  "no-endpoint": "環境設定 ＞ 外部の計算機 で、この PC の計算機（local）を使えるようにしてください",
};
export const explain = (/** @type {string | undefined} */ code) =>
  code && REFUSALS[code] ? `${REFUSALS[code]}（${code}）` : String(code);

/** 対象のタイルを使えるか。使えなければ理由を返す。 */
export function targetProblem(/** @type {any} */ t) {
  if (!t) return "画像のタイルを選んでから開いてください";
  if (t.kind !== "image") return "動画は対象外です";
  const m = String(t.modality ?? "").toUpperCase();
  if (m !== "CR" && m !== "DX") return `胸部X線（CR/DX）の 1 枚だけに使えます（このシリーズは ${m || "?"}）`;
  if (t.sliceCount > 1) return `1 枚の画像だけに使えます（このシリーズは ${t.sliceCount} 枚）`;
  return null;
}

/** 計算機の失敗（errorName: errorValue）を、画面に出す文にする。 */
export function failureText(/** @type {string | undefined} */ value) {
  const v = String(value ?? "");
  const m = /^(missing-packages|unsupported-version|not-applicable): ?(.*)$/s.exec(v);
  return m ? m[2] : `失敗: ${v}`;
}

// ---------------------------------------------------------------------------
// 画面
// ---------------------------------------------------------------------------

/** 明るい・暗いどちらのテーマでも読める（背景と文字色の両方を固定する） */
const BANNER_STYLE = "padding: 8px 10px; background: #fff4e5; border: 1px solid #f0b36b; border-radius: 4px; color: #5c3300";

/** @param {any} host */
export async function activate(host) {
  const target = (host.getTargets?.() ?? []).find((/** @type {any} */ t) => t.kind === "image");
  const problem = targetProblem(target);
  if (problem) {
    host.notify("胸部X線レポート: " + problem);
    return;
  }
  const win = host.openWindow({ title: "胸部X線レポートの下書き（CXRMate）", width: 640, height: 640 });
  const root = win.container;
  root.style.cssText = "font: 13px system-ui, sans-serif; padding: 12px; overflow: auto; display: flex; flex-direction: column; gap: 8px;";
  /** @type {Record<string, any>} */
  const state = (/** @type {any} */ (window).__cxrReportState = { phase: "idle" });

  const el = (/** @type {string} */ tag, /** @type {Record<string, any>} */ props = {}, /** @type {any[]} */ ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "testid") e.dataset.testid = v;
      else if (k === "style") e.style.cssText = v;
      else /** @type {any} */ (e)[k] = v;
    }
    for (const c of kids) e.append(c);
    return e;
  };

  const banner = el("div", {
    testid: "cxr-disclaimer",
    textContent: "⚠ " + DISCLAIMER,
    style: BANNER_STYLE,
  });
  const about = el("div", {
    testid: "cxr-about",
    style: "font-size: 12px; opacity: 0.8; line-height: 1.5",
    textContent:
      `胸部の正面像（PA/AP）の 1 枚だけを入れてください。部位と撮影方向は判定しません。` +
      `モデル: ${MODEL}（Apache-2.0。MIMIC-CXR で学習）。画像はこの PC の計算機だけに渡り、外へは出ません。` +
      "初回は重み（約 430 MB）を Hugging Face から取得します。CPU では 1 枚に 10 秒〜1 分ほどかかります。" +
      "計算機の Python に torch・torchvision・transformers==" + TRANSFORMERS + " が必要です（環境は書き換えません）。",
  });
  const runBtn = el("button", { textContent: "実行", testid: "cxr-run", style: "align-self: flex-start; padding: 4px 18px" });
  const status = el("div", { testid: "cxr-status", style: "min-height: 1.4em; opacity: 0.9" });
  const result = el("div", { testid: "cxr-result", style: "display: none; flex-direction: column; gap: 8px" });
  root.append(banner, about, el("div", { style: "opacity: 0.8" }, `対象: ${target.seriesLabel ?? ""}`), runBtn, status, result);

  const setStatus = (/** @type {string} */ t) => { status.textContent = t; state.status = t; };
  const onProgress = (/** @type {number} */ p, /** @type {string | undefined} */ m) => setStatus(`${Math.round(p * 100)}% ${m ?? ""}`);

  const section = (/** @type {string} */ label, /** @type {string} */ text, /** @type {string} */ id) => {
    const area = el("textarea", { value: text, readOnly: true, testid: id, rows: id === "cxr-findings" ? 8 : 3, style: "width: 100%; box-sizing: border-box; font: 13px system-ui, sans-serif" });
    const copy = el("button", { textContent: "コピー", style: "font-size: 11px" });
    copy.addEventListener("click", () => { void navigator.clipboard?.writeText(text); });
    return el("div", {}, el("div", { style: "display: flex; gap: 8px; align-items: center; font-weight: 600" }, label, copy), area);
  };

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    result.style.display = "none";
    result.replaceChildren();
    state.phase = "running"; state.error = undefined; state.report = undefined;
    setStatus("送っています…");
    const r = await host.compute.runJob({
      endpointId: "local",
      inputs: [{ studyUid: target.studyUid, seriesUid: target.seriesUid, format: "npz" }],
      script: buildScript(),
      timeoutSec: 3600,
    }, { onProgress });
    if (!r.ok) {
      state.phase = "failed"; state.error = r.error;
      setStatus(r.cancelled ? "取り消しました" : explain(r.error));
      runBtn.disabled = false;
      return;
    }
    if (r.status !== "ok") {
      state.phase = "failed"; state.error = `${r.errorName}: ${r.errorValue}`; state.stderr = r.stderr;
      setStatus(failureText(r.errorValue));
      runBtn.disabled = false;
      return;
    }
    const raw = await r.readFile("report.json");
    if (!raw) {
      state.phase = "failed"; state.error = "no-report";
      setStatus("失敗: 結果が返りませんでした");
      runBtn.disabled = false;
      return;
    }
    const rep = JSON.parse(new TextDecoder().decode(raw));
    state.phase = "done"; state.report = rep;
    result.replaceChildren(
      el("div", { style: BANNER_STYLE, testid: "cxr-disclaimer-result" }, "⚠ " + DISCLAIMER),
      section("所見（findings）", rep.findings, "cxr-findings"),
      section("印象（impression）", rep.impression, "cxr-impression"),
      el("div", { testid: "cxr-meta", style: "font-size: 11px; opacity: 0.75" },
        `${rep.model} @ ${String(rep.revision).slice(0, 8)}・${rep.device}・${rep.seconds} 秒・窓: ${rep.window === "dicom" ? "DICOM の窓" : "最小〜最大"}`),
    );
    result.style.display = "flex";
    setStatus("できました");
    runBtn.disabled = false;
  });
}
