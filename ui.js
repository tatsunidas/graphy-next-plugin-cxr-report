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


def need_packages():
    """必要なパッケージ（版は指定があるものだけ）。事前チェックと本処理で同じ表を使う"""
    return {'torch': None, 'torchvision': None, 'transformers': GRAPHY['transformers'], 'pillow': None, 'numpy': None}


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


def crop_box(rows, cols, side):
    """Resize(side)（短辺を side に）→ CenterCrop(side) が元画像で見ている範囲 {x0, y0, w, h}（元画像の画素座標）"""
    short = min(rows, cols)
    scale = side / short
    new_h, new_w = (side, int(side * cols / rows)) if rows <= cols else (int(side * rows / cols), side)
    top, left = int(round((new_h - side) / 2.0)), int(round((new_w - side) / 2.0))
    return {'x0': left / scale, 'y0': top / scale, 'w': side / scale, 'h': side / scale}


def sentence_spans(ids, dot, specials, n):
    """生成列 ids[0:n] を文に分ける。返す (a, b) は対数確率 lp[a:b]（= 予測する ids[a+1 .. b]）。
    '.' で閉じ、直後の sep / eos は同じ文に含める（単独の文にしない）"""
    last = n - 1
    spans, a, t = [], 0, 0
    while t < last:
        end = t + 1
        if ids[end] == dot:
            while end < last and ids[end + 1] in specials:
                end += 1
            spans.append((a, end))
            a = t = end
        else:
            t += 1
    if a < last:
        spans.append((a, last))
    return spans


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
NEED = need_packages()
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
        use_cache=True, max_length=256, num_beams=4, output_scores=True)
findings, impression = model.split_and_decode_sections(out.sequences, [tok.sep_token_id, tok.eos_token_id], tok)
stage('generated', 0.85)

# --- Grad-CAM（文ごと）。対象は CvT の最終段の特徴マップ（射影の手前）。生成済みの系列で teacher forcing をやり直し、
# 文の対数確率の和を特徴マップで微分する。やり直した確率が generate の sequences_scores と合わなければ、別の確率を
# 微分していることになるので cam は出さない（理由を返す）
cam_info = {'cam': None, 'camError': None}
try:
    seq = out.sequences
    ids = seq[0].tolist()
    n = sum(1 for i in ids if i != tok.pad_token_id)
    fm = model.encoder.cvt(px).last_hidden_state
    tokens = model.encoder.projection_head(torch.permute(torch.flatten(fm, 2), [0, 2, 1]))
    din = seq[:, :n - 1]
    tt = model.token_ids_to_token_type_ids(din, [tok.sep_token_id])
    o = model(encoder_outputs=(tokens,), decoder_input_ids=din, decoder_token_type_ids=tt, return_dict=True)
    lp = torch.log_softmax(o.logits, -1).gather(-1, seq[:, 1:n].unsqueeze(-1)).squeeze(-1)[0]
    gap = abs(lp.sum().item() / (n - 1) - out.sequences_scores[0].item())
    if gap > 0.01:
        raise RuntimeError('gradcam-mismatch: 再計算した確率が生成時と合いません（差 %.4f）' % gap)
    spans = sentence_spans(ids, tok.convert_tokens_to_ids('.'), (tok.sep_token_id, tok.eos_token_id), n)
    sep_at = ids.index(tok.sep_token_id) if tok.sep_token_id in ids else None
    cams, sentences = [], []
    for a, b in spans:
        g, = torch.autograd.grad(lp[a:b].sum(), fm, retain_graph=True)
        cams.append(torch.relu((g.mean(dim=(2, 3), keepdim=True) * fm).sum(1))[0].detach())
        sentences.append({
            'section': 'findings' if (sep_at is None or a + 1 <= sep_at) else 'impression',
            'text': tok.decode(ids[a + 1:b + 1], skip_special_tokens=True).strip(),
            'logprob': round(lp[a:b].sum().item(), 3), 'tokens': b - a,
        })
    peak = max(c.max().item() for c in cams)
    for c, sent in zip(cams, sentences):
        m = c.max().item()
        sent['strength'] = round(m / peak, 3) if peak > 0 else 0.0
    np.save('outputs/cam.npy', np.stack([(c / c.max()).cpu().numpy() if c.max() > 0 else c.cpu().numpy() for c in cams]).astype(np.float32))
    cam_info = {'cam': {'shape': list(cams[0].shape), 'crop': crop_box(info['rows'], info['cols'], side), 'sentences': sentences,
                        'gap': round(gap, 5)}, 'camError': None}
except Exception as e:
    cam_info = {'cam': None, 'camError': type(e).__name__ + ': ' + str(e)}
stage('gradcam', 0.95)
json.dump({
    'findings': findings[0], 'impression': impression[0],
    'model': name, 'revision': rev, 'device': device,
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'seconds': round(time.time() - T0, 1), **info, **cam_info,
}, open('outputs/report.json', 'w', encoding='utf-8'), ensure_ascii=False)
stage('done', 1.0)
`;

/** 画像を送らない事前チェック。入っているパッケージと Python の場所を env.json に書くだけ（何も入れない・書き換えない） */
export const PY_PROBE = String.raw`
stage('probe', 0.3)
need = need_packages()
installed = {d: version_of(d) for d in need}
torch_error = None
cuda = None
if installed['torch']:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception as e:
        torch_error = type(e).__name__ + ': ' + str(e)[:200]
weights = None
try:
    from huggingface_hub import try_to_load_from_cache
    weights = isinstance(try_to_load_from_cache(GRAPHY['model'], 'config.json', revision=GRAPHY['revision']), str)
except Exception:
    weights = None
problem = env_problem(installed, need)
if problem is None and torch_error:
    problem = 'broken-package: torch を読み込めません（' + torch_error + '）。入れ直してください'
os.makedirs('outputs', exist_ok=True)
json.dump({'python': sys.executable, 'pyVersion': sys.version.split()[0], 'installed': installed, 'need': need,
           'problem': problem, 'torchError': torch_error, 'cuda': cuda, 'weightsCached': weights},
          open('outputs/env.json', 'w', encoding='utf-8'), ensure_ascii=False)
stage('done', 1.0)
`;

export const PY = PY_PREP + PY_MODEL;

/** 計算機に送るコード。差し込むのは固定の 3 値だけ。 */
export function buildScript() {
  const cfg = { model: MODEL, revision: REVISION, transformers: TRANSFORMERS };
  return `GRAPHY = __import__('json').loads(${JSON.stringify(JSON.stringify(cfg))})\n` + PY;
}

/** 事前チェックのコード（画像は送らない）。差し込むのは buildScript と同じ固定の 3 値だけ。 */
export function buildProbeScript() {
  const cfg = { model: MODEL, revision: REVISION, transformers: TRANSFORMERS };
  return `GRAPHY = __import__('json').loads(${JSON.stringify(JSON.stringify(cfg))})\n` + PY_PREP + PY_PROBE;
}

/**
 * 不足・版違いを入れる pip のコマンド。計算機が実際に使う Python（sys.executable）を名指しする
 * （Windows では python と py が別物のことがある）。空白などを含むパスは引用符で囲む。
 */
export function installCommand(/** @type {string} */ python, /** @type {Record<string, string | null>} */ installed, /** @type {Record<string, string | null>} */ need) {
  const wanted = Object.keys(need).filter((d) => installed[d] == null || (need[d] && String(installed[d]).split("+")[0] !== need[d]));
  if (!wanted.length) return null;
  const exe = /^[\w./:\\-]+$/.test(python) ? python : `"${python}"`;
  return `${exe} -m pip install ${wanted.map((d) => d + (need[d] ? "==" + need[d] : "")).join(" ")}`;
}

/** float32・C 順・little-endian の .npy を読む（cam.npy 用）。 */
export function parseNpyF32(/** @type {Uint8Array} */ bytes) {
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.subarray(1, 6)) !== "NUMPY") throw new Error("npy ではありません");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const hlen = bytes[6] === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const off = (bytes[6] === 1 ? 10 : 12) + hlen;
  const header = new TextDecoder().decode(bytes.subarray(off - hlen, off));
  if (!/'descr': *'<f4'/.test(header) || /'fortran_order': *True/.test(header)) throw new Error("float32・C 順の npy だけ読めます: " + header);
  const shape = [.../\(([^)]*)\)/.exec(header)?.[1].matchAll(/\d+/g) ?? []].map((m) => Number(m[0]));
  const n = shape.reduce((a, b) => a * b, 1);
  if (bytes.byteLength - off !== n * 4) throw new Error("npy の大きさが合いません");
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = dv.getFloat32(off + i * 4, true);
  return { shape, data };
}

/**
 * 特徴マップ 1 枚（h×w・0〜1）を、モデルが見た範囲 crop（元画像の画素座標）に双線形で伸ばし、元画像の rows×cols に置く。
 * 範囲の外（中央切り出しで落ちた端）は NaN（オーバーレイで透明）。
 */
export function upsampleCam(/** @type {Float32Array} */ cam, /** @type {[number, number]} */ [h, w], /** @type {{x0:number,y0:number,w:number,h:number}} */ crop, rows, cols) {
  const out = new Float32Array(rows * cols).fill(NaN);
  const x1 = crop.x0 + crop.w, y1 = crop.y0 + crop.h;
  for (let y = Math.max(0, Math.floor(crop.y0)); y < Math.min(rows, Math.ceil(y1)); y++) {
    const v = Math.min(h - 1, Math.max(0, ((y + 0.5 - crop.y0) / crop.h) * h - 0.5));
    if (y + 0.5 < crop.y0 || y + 0.5 > y1) continue;
    const v0 = Math.floor(v), v1 = Math.min(h - 1, v0 + 1), fv = v - v0;
    for (let x = Math.max(0, Math.floor(crop.x0)); x < Math.min(cols, Math.ceil(x1)); x++) {
      if (x + 0.5 < crop.x0 || x + 0.5 > x1) continue;
      const u = Math.min(w - 1, Math.max(0, ((x + 0.5 - crop.x0) / crop.w) * w - 0.5));
      const u0 = Math.floor(u), u1 = Math.min(w - 1, u0 + 1), fu = u - u0;
      out[y * cols + x] =
        cam[v0 * w + u0] * (1 - fu) * (1 - fv) + cam[v0 * w + u1] * fu * (1 - fv) +
        cam[v1 * w + u0] * (1 - fu) * fv + cam[v1 * w + u1] * fu * fv;
    }
  }
  return out;
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
  const envBox = el("div", { testid: "cxr-env", style: `display: none; flex-direction: column; gap: 6px; ${BANNER_STYLE}` });
  root.append(banner, about, el("div", { style: "opacity: 0.8" }, `対象: ${target.seriesLabel ?? ""}`), runBtn, status, envBox, result);

  const setStatus = (/** @type {string} */ t) => { status.textContent = t; state.status = t; };
  const onProgress = (/** @type {number} */ p, /** @type {string | undefined} */ m) => setStatus(`${Math.round(p * 100)}% ${m ?? ""}`);

  const section = (/** @type {string} */ label, /** @type {string} */ text, /** @type {string} */ id) => {
    const area = el("textarea", { value: text, readOnly: true, testid: id, rows: id === "cxr-findings" ? 8 : 3, style: "width: 100%; box-sizing: border-box; font: 13px system-ui, sans-serif" });
    const copy = el("button", { textContent: "コピー", style: "font-size: 11px" });
    copy.addEventListener("click", () => { void navigator.clipboard?.writeText(text); });
    return el("div", {}, el("div", { style: "display: flex; gap: 8px; align-items: center; font-weight: 600" }, label, copy), area);
  };

  /** Grad-CAM の節: 文を押すと、その文の生成に効いた領域を画像に重ねる。 */
  async function camSection(/** @type {any} */ r, /** @type {any} */ rep) {
    const box = el("div", { testid: "cxr-cam", style: "display: flex; flex-direction: column; gap: 6px" });
    box.append(el("div", { style: "font-weight: 600" }, "Grad-CAM（文を押すと画像に重ねます）"));
    if (!rep.cam) {
      box.append(el("div", { testid: "cxr-cam-error", style: "opacity: 0.85" }, `Grad-CAM は作れませんでした: ${rep.camError ?? "不明"}`));
      state.cam = { error: rep.camError ?? null };
      return box;
    }
    const raw = await r.readFile("cam.npy");
    if (!raw) {
      box.append(el("div", { testid: "cxr-cam-error" }, "Grad-CAM は作れませんでした: cam.npy が返りませんでした"));
      return box;
    }
    const npy = parseNpyF32(raw);
    const [h, w] = rep.cam.shape;
    const sentences = rep.cam.sentences;
    box.append(el("div", { style: "font-size: 12px; opacity: 0.8; line-height: 1.5" },
      "その文の生成に効いた領域です。所見が正しいことの根拠ではありません。" +
      `解像度は粗く（${h}×${w}）、画像にない内容（年齢・撮影方向など）には意味のある領域が出ません。` +
      "「強さ」は全文の最大に対する比で、小さい文は画像にほとんど依存していません。"));
    // 本体の LUT「Hot_Iron」の見た目（低=青 → 赤 → 黄 → 高=白）。LUT が変わったらここも合わせる
    box.append(el("div", { testid: "cxr-cam-legend", style: "display: flex; gap: 6px; align-items: center; font-size: 11px; opacity: 0.9" },
      "効いていない",
      el("div", { style: "width: 160px; height: 10px; border-radius: 2px; background: linear-gradient(to right, #0101fe, #fa0104 50%, #fefa01 75%, #fefef5)" }),
      "強く効いた"));
    const opacity = el("input", { type: "range", min: "0.2", max: "0.8", step: "0.05", value: "0.5", testid: "cxr-cam-opacity" });
    const clear = el("button", { textContent: "消す", testid: "cxr-cam-clear", style: "font-size: 11px" });
    box.append(el("div", { style: "display: flex; gap: 8px; align-items: center; font-size: 12px" }, "不透明度", opacity, clear));
    const rowsEl = [];
    const show = (/** @type {number} */ k) => {
      const data = upsampleCam(npy.data.subarray(k * h * w, (k + 1) * h * w), [h, w], rep.cam.crop, rep.rows, rep.cols);
      const ok = host.showOverlay(target.tileId, { data, rows: rep.rows, cols: rep.cols, colormap: "Hot_Iron", opacity: Number(opacity.value), window: { center: 0.5, width: 1 } });
      state.cam = { shown: ok ? k : null, error: ok ? null : "overlay-refused" };
      if (!ok) setStatus("重ねられませんでした（画像の大きさが表示中のものと合いません）");
      rowsEl.forEach((e, i) => { e.style.outline = ok && i === k ? "2px solid #f0b36b" : "none"; });
    };
    sentences.forEach((/** @type {any} */ sent, /** @type {number} */ k) => {
      const row = el("div", {
        testid: `cxr-cam-sentence-${k}`, textContent: `${sent.section === "findings" ? "所見" : "印象"}｜${sent.text}（強さ ${sent.strength.toFixed(2)}）`,
        style: "cursor: pointer; padding: 4px 6px; border: 1px solid #8884; border-radius: 4px; font-size: 12px",
      });
      row.addEventListener("click", () => show(k));
      rowsEl.push(row);
      box.append(row);
    });
    opacity.addEventListener("change", () => { if (Number.isInteger(state.cam?.shown)) show(state.cam.shown); });
    clear.addEventListener("click", () => { host.clearOverlay(target.tileId); state.cam = { shown: null, error: null }; rowsEl.forEach((e) => { e.style.outline = "none"; }); });
    state.cam = { shown: null, error: null, sentences: sentences.length };
    return box;
  }

  /** 事前チェックを通ったか（この窓のあいだだけ覚える。通ったあとは同意の窓を増やさない） */
  let envOk = false;

  /** 不足・版違いの案内。入れるのは利用者（環境は書き換えない）。 */
  const showEnv = (/** @type {any} */ env) => {
    const rows = Object.keys(env.need).map((d) => {
      const v = env.installed[d];
      const want = env.need[d];
      const ok = v != null && (!want || String(v).split("+")[0] === want);
      return el("div", { style: "display: flex; gap: 12px; font-size: 12px" },
        el("span", { style: "width: 110px; font-weight: 600" }, d),
        el("span", { style: "width: 110px" }, want ? "== " + want : "（版は問わない）"),
        el("span", {}, v == null ? "✗ 入っていません" : ok ? "✓ " + v : "✗ " + v + "（版が違います）"));
    });
    const cmd = installCommand(env.python, env.installed, env.need);
    const kids = [
      el("div", { style: "font-weight: 600" }, "実行の前に、この PC の計算機の Python に足りないものがあります"),
      el("div", { testid: "cxr-env-python", style: "font-size: 12px; word-break: break-all" }, `使われる Python: ${env.python}（${env.pyVersion}）`),
      ...rows,
    ];
    if (env.problem && /^broken-package/.test(env.problem)) kids.push(el("div", { style: "font-size: 12px" }, failureText(env.problem)));
    if (cmd) {
      const area = el("textarea", { value: cmd, readOnly: true, rows: 4, testid: "cxr-env-command", style: "width: 100%; box-sizing: border-box; font: 12px ui-monospace, monospace" });
      const copy = el("button", { textContent: "コマンドをコピー", testid: "cxr-env-copy", style: "font-size: 11px" });
      copy.addEventListener("click", () => { void navigator.clipboard?.writeText(cmd); });
      kids.push(
        el("div", { style: "font-size: 12px" }, "ターミナル（Windows は PowerShell かコマンドプロンプト）で、次を実行してください。" +
          "このプラグインは環境を書き換えません。"),
        area, copy,
        el("div", { style: "font-size: 11px; opacity: 0.85; line-height: 1.5" },
          "GPU（CUDA）で動かしたいときは、torch だけ pytorch.org の案内の版を入れてください。" +
          "パスを引用符で囲んだ場合、PowerShell では先頭に & を付けてください。"));
    }
    const again = el("button", { textContent: "再確認", testid: "cxr-env-recheck", style: "align-self: flex-start; padding: 2px 14px" });
    again.addEventListener("click", async () => {
      again.disabled = true;
      if (await checkEnv()) setStatus("環境は整いました。「実行」を押してください");
      again.disabled = false;
    });
    kids.push(again, el("div", { style: "font-size: 11px; opacity: 0.85" }, "入れたら［再確認］を押してください（本体の再起動は要りません）。"));
    envBox.replaceChildren(...kids);
    envBox.style.display = "flex";
  };

  /** 画像を送らない小さなジョブで、計算機の Python を確かめる。整っていれば true */
  const checkEnv = async () => {
    setStatus("環境を確認しています…");
    const r = await host.compute.runJob({ endpointId: "local", inputs: [], script: buildProbeScript(), timeoutSec: 300 }, { onProgress });
    if (!r.ok) {
      state.error = r.error;
      setStatus(r.cancelled ? "取り消しました" : explain(r.error));
      return false;
    }
    if (r.status !== "ok") {
      state.error = `${r.errorName}: ${r.errorValue}`;
      setStatus(failureText(r.errorValue));
      return false;
    }
    const raw = await r.readFile("env.json");
    if (!raw) {
      state.error = "no-env";
      setStatus("失敗: 環境の確認結果が返りませんでした");
      return false;
    }
    const env = JSON.parse(new TextDecoder().decode(raw));
    state.env = env;
    if (env.problem) {
      envOk = false;
      showEnv(env);
      setStatus("足りないものがあります。下の手順で入れて、［再確認］を押してください");
      return false;
    }
    envOk = true;
    envBox.style.display = "none";
    envBox.replaceChildren();
    return true;
  };

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    if (!envOk && !(await checkEnv())) {
      runBtn.disabled = false;
      return;
    }
    result.style.display = "none";
    result.replaceChildren();
    host.clearOverlay?.(target.tileId);
    state.phase = "running"; state.error = undefined; state.report = undefined; state.cam = undefined;
    setStatus(state.env?.weightsCached === false ? "送っています…（初回は重み約 430 MB を取得します）" : "送っています…");
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
      const envLost = /^(missing-packages|unsupported-version): /.test(String(r.errorValue));
      if (envLost) envOk = false; // 確認のあとで環境が変わった。次の「実行」でもう一度確かめて手順を出す
      setStatus(failureText(r.errorValue) + (envLost ? "（「実行」を押すと環境を確認して、導入の手順を示します）" : ""));
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
      await camSection(r, rep),
    );
    result.style.display = "flex";
    setStatus("できました");
    runBtn.disabled = false;
  });
}
