# リポジトリの構造

Baphomet（Qwen3.5-4B ベースの 8 ドナーを学習なしでマージした 4B モデル群）の作業記録。番号順が時系列。重み・データセット本体・API キーは置かない（`AGENTS.md` の「Not in this repo」参照）。

## ルート

| ファイル | 中身 |
|---|---|
| `README.md` | 入口。まず `AGENTS.md` と `PROGRESS.md` を読む |
| `AGENTS.md` | エージェント（Hark / Claude Code / Codex など）向けの引き継ぎ。モデルの系譜、結果と注意書き、計算資源 |
| `CLAUDE.md` | Claude Code 用（`AGENTS.md` と同趣旨） |
| `PROGRESS.md` | 完了したこと・保留中のこと |
| `REPO_STRUCTURE.md` | このファイル |
| `merge-papers-sent.md` | 送ったマージ論文のメモ |
| `.gitignore` | `*.safetensors` `*.gguf` `*.bin` `*.pt`、venv、鍵類を除外 |

## ディレクトリ

| ディレクトリ | 時期 | 中身 | 結論 |
|---|---|---|---|
| `01-baphomet-1.0-merge/` | 2026-11 初旬 | 初期マージのログ、ドナーのキー一覧、`slerp_merge.py` | Baphomet-1.0 は GoatMerge（別リポ）で 0.1375×8 の線形マージ |
| `02-bench180/` | 11-09 | 180 問の対比較ベンチ（`items.jsonl`、`score.py`、BFCL AST チェッカー、1.0 vs 基盤の結果とグラフ） | 1.0 が基盤に勝ち。以後すべてのテストはこの 180 問 |
| `03-bench90/` | 11-10 | 90 問ベンチ（MERGE³ 込み） | |
| `04-merge3/` | 11-10 | MERGE³ 方式の進化マージ（IRT 項目選択、カーネル、ログ）。`NOTES.md` に設計 | 1.0 に負け、ツール呼び出しも劣化 |
| `05-evogm/` | 11-12〜13 | EvoGM（8 次元 λ）。`NOTES.md`、`kernels/`、`kout/`、`final*/`、`report/`（日英 PDF と TeX）、`repo/`（上流 EvoGM のコピー） | ラウンド 3 の勝者 = Baphomet-1.1-4B |
| `06-baphomet-1.1/` | 11-13〜15 | 1.1 の CPU 再構築、一致検査、モデルカード、HF アップロード | HF `CloudGoat/Baphomet-1.1-4B` で公開 |
| `07-evogm2/` | 11-15〜16 | EvoGM v2（ドナーごとにアテンション系 / MLP 系の 16 次元 λ） | 勝者 r1i1 は 134/180 で B11（140/180）に負け、公開見送り |
| `08-evogm-dfs/` | 11-17 | EvoGM-DFS（r1i1 / B11 / B10 の層を 4 層ブロックで並べ替える Data Flow Space 探索） | 空間内の最良は素の B11。**知見のまとめは `08-evogm-dfs/report/findings_ja.pdf`** |

## 07-evogm2 の中

| パス | 中身 |
|---|---|
| `NOTES.md` | 設計・テスト・実行ログ（DFS の提案と結果も末尾にある） |
| `build_sets.py` `rebalance_search.py` `build_bfcl_cands.py` `score_pre.py` | 探索 200 / 検証 150 の問題セット作成、BFCL の事前選別 |
| `search_items*.json` `val_items.json` `bfcl_cands.json` `bfcl_pre_p.json` | 問題セット（GPQA 本文を含む。公開しないこと） |
| `kernel_bfclpre/` | BFCL 事前選別用 Kaggle カーネル |
| `pre_out/` | 事前選別の出力 |
| `src/` | カーネルのソース（`part_a/b/c.py`、`gpuworker4.py`、`build.py`）。`src_bak_20261116/` は修正前のバックアップ |
| `kernel_main/` | 本番カーネル `cloudgoat/evogm2-main`（`build.py` の生成物） |
| `kernel_final/` | 足切り緩和（BFCL 基盤 −2/30）で再判定し 180 問テストを走らせたカーネル |
| `kds/` | Kaggle データセット `cloudgoat/evogm2-data`（`linmerge.py` = 16 次元マージ、`fitscore.py`、問題セット、テスト 180） |
| `kout/main/` `kout/final/` | カーネル出力（`evo_log.jsonl`、`val_result.json`、`evo_result.json`、IRT、生成出力 `*_outputs/`、ログ） |
| `tests/` `mock/` | 単体テスト、GPU なしで制御フローを検証するモック |
| `check.sh` `push.sh` | 状態確認・出力取得、データセット作成とカーネル push |

## 08-evogm-dfs の中

| パス | 中身 |
|---|---|
| `NOTES.md` | DFS の設計・数値検証・モック試験・結果（`07-evogm2/NOTES.md` 末尾と同じ内容） |
| `report/` | **知見のまとめ** `findings_ja.pdf`、原稿 `findings_ja.md`、`style.css`、`build.sh`（pandoc + weasyprint） |
| `src/` | `dfslib.py`（ブロック経路と W の畳み込み）、`gpuworker_dfs.py`、`part_a/b/c.py`、`build.py`、`evogm_dfs_src.py` |
| `kernel/` | 本番カーネル `cloudgoat/evogm-dfs`（40 層パディングエンジン、md5 1ddf5a16） |
| `kds_dfs/` | データセット `cloudgoat/evogm-dfs-data`（v2 の結果を引き継ぐ `v2_prev.json` など） |
| `build_prev.py` | `07-evogm2/kout/main` から `v2_prev.json` を作る |
| `kout/` | カーネル出力（`dfs_log.jsonl`、`val_result.json`、`evo_result.json`、IRT 再推定、生成出力、ログ）。状態ファイル `dfs_state.pt` は除外 |
| `tests/` `mock/` | パディング・W 畳み込みの数値テスト、中断再開を含むモック試験 |
| `check.sh` `push.sh` | 状態確認・出力取得、push |

## 呼び名の対応

| 呼び名 | 実体 |
|---|---|
| B10 | Baphomet-1.0-4B（0.1375×8） |
| B11 | Baphomet-1.1-4B（EvoGM ラウンド 3 勝者） |
| r1i1 | EvoGM v2 でラウンド 1・イテレーション 1 に生成された候補（v2 の勝者） |
| 180 問テスト | `02-bench180/items.jsonl`（MMLU-Pro 80 / GPQA 50 / BFCL 50、上限 8,192） |
