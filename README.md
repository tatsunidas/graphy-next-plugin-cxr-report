# Chest X-ray Report Draft (CXRMate) — GRAPHY-Next plugin

胸部X線 1 枚（CR/DX）から、所見（findings）と印象（impression）の**下書き**を作る GRAPHY-Next の公式プラグインです。
モデルは [CXRMate](https://huggingface.co/aehrc/cxrmate-single-tf)（aehrc・Apache-2.0）。

> ⚠ **研究用です。診断・治療の判断に使わないでください。** 生成された文は誤りを含み、画像にない内容
> （撮影方向・年齢・性別・過去画像との比較など）が入ることがあります。医療機器ではありません。

## 使い方
1. 2D ビューアで胸部X線（CR/DX・1 枚）を開く。
2. 「解析 ＞ AI ＞ Chest X-ray Report Draft」を開き、「実行」を押す。
3. 同意の窓で、送るデータと実行されるコードの全文を確かめて承認する。
4. 所見と印象が表示される（コピーできる。DICOM への保存はしない）。

## 動く場所と条件
- **この PC の計算機（環境設定 ＞ 外部の計算機の「この PC」）でだけ動きます。** 胸部X線は画素に焼き込まれた文字を
  塗らずに渡すため、本体が外部の計算機（Colab など）への送信を断ります（`dataset-local-only`）。画像は PC の外へ出ません。
- GRAPHY-Next 0.5.1 以降（マスク無しの CR/DX を local にだけ渡す変更を含む版）。
- 計算機の Python に **torch・torchvision・`transformers==4.46.3`**・pillow・numpy。環境は書き換えません。足りなければ
  名前を示して止まります。🔴 **transformers は 4.46.3 だけ**です（4.57.6 では `generate` が落ちることを確かめた。
  それ以外の版は未確認なので止めます）。
- 重みは同梱しません。初回に Hugging Face から取得します（キャッシュ約 430 MB）。`trust_remote_code=True` で
  モデルの Python コードを読むため、**モデルとコードを固定の revision（`84dfcba8125c9b9296bfc1ee317749788b6b59b4`）で取得**します。
  同意画面のコードに revision が出ます。
- 対象は 1 枚の 2D 画像だけ。多フレーム・CR/DX 以外（MG を含む）は、理由を出して止まります。
  表示の窓（DICOM の WindowCenter/Width）を当てて 8bit にして渡します。窓が無ければ最小〜最大です。

## 確かめたこと・確かめていないこと
- 確かめた（Linux・CPU・Electron の実機）: NIH ChestX-ray14 の 1 枚（1024×1024）で、所見と印象が返る・同意画面に
  注意と revision が出る・SHA-256 が出る・`transformers` が別の版の環境では理由を出して止まる。CPU 16 コアで 1 枚 約 10 秒（実機・重みはキャッシュ済み・読込を含む。別の直接実行では約 58 秒だった）。
- **未確認**: NVIDIA GPU での実行、Windows・macOS、初回の重みの取得（検証時はキャッシュ済み）、生成文の臨床的な正しさ。
  実際の出力は「PA and lateral views」「___-year-old female」「Comparison is made with…」のように、1 枚の画像からは
  言えない定型文を含みました。生成文の正確さは評価していません。

## ライセンスと出典
- このプラグインのコード: MIT（`LICENSE`）。
- モデル: CXRMate（[aehrc/cxrmate-single-tf](https://huggingface.co/aehrc/cxrmate-single-tf)）は Apache-2.0（ゲート無し）。
  **学習データは MIMIC-CXR** です。本プラグインは MIMIC-CXR のデータを同梱・使用しません（モデルの重みだけを使います）。
  論文: Nicolson et al., *Longitudinal data and a semantic similarity reward for chest X-ray report generation*,
  Informatics in Medicine Unlocked (2024), doi:10.1016/j.imu.2024.101585。
- 重みは再配布せず、利用者の計算機が Hugging Face から取得します。

## 今後
- RaDialog は、基盤の LLaVA-1.5-7B が Llama 2 Community License のため、条件を全文読んでから別の版で検討します（初版には入れていません）。
- 縦断版（`cxrmate-multi-tf` など）は前回レポートが要るため対象外です。

## 開発
```
npm test        # 純関数・計算機の上のコード前半（numpy のある python3）・画面の往復
```
モデル本体の実行は GRAPHY-Next の実機検証（`automator/src/spike/computeCxrReportCheck.ts`）で確かめます。
