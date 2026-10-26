# fly_culture_spontaneous

自発的な神経活動の「点火点」を、**分散培養した大脳皮質ニューロン**と**ハエの全脳配線図（FlyWire）**で同じ物差しで調べた研究です。

> ノイズだけで動いている神経回路で、自発的な集団発火（ネットワークバースト）はどこから始まるのか。何がそこを点火点にするのか。

2026 年 10 月 27 日（1 日）で実施。テーマを選んだ経緯は [notes/theme_selection.md](notes/theme_selection.md)。

## 結果の要約

**培養神経（公開 MaxOne HD-MEA データ: 7 培養・114 セッション・バースト 30,349 回。再現用に 59 電極の 30 培養）**
- バーストは局所的に始まり、点火のホットスポットは日をまたいで同じ場所に戻る（連続する日の点火マップの相関 r = 0.43、別の培養どうしは −0.06。7/7 培養）
- 点火点を予測するのは「よく発火する場所」ではなく、バースト間の協調したゆらぎの**主モード**（機能的結合行列の主固有ベクトル: ρ = 0.25 対 発火率 ρ ≈ 0、7/7 培養）
- 分岐比 m ≈ 0.99（暴走の少し手前）。発達とともにバーストが増え、m は 1 から離れる。30 分の記録の中で m は上がらない（疲労型で、SHY 型ではない）

**ハエの脳（FlyWire v783、138,639 ニューロンのスパイキングモデル）**
- 培養と同じ条件（弱いノイズ＋短期シナプス抑圧）を与えると、行動で較正した脳はバーストしない。シナプスを約 2 倍に強めるとバーストする
- 興奮性の配線の主モードは、**触角葉のアセチルコリン性局所ニューロン**（中心は片側 15 個の lLN1_bc）と、その相手（投射ニューロン、ケニヨン細胞）にある（伝達物質の 3 通りの仮定すべて）
- lLN1_bc は脳で最も自発発火の多いニューロンの一つ（全体の上位 0.15%）。バーストの 30〜45 ms 前に発火する（最小補正の仮定）か、常時発火して脳を駆動する（他の仮定）
- 触角葉のアセチルコリン性局所ニューロン 111 個（脳の 0.08%）を黙らせると、6 条件すべてで自発バーストが 2.3〜9 分の 1 に減る。同数のランダムな細胞では減らない
- **本物のハエ**（安静時の全脳イメージング、20 匹）でも、触角葉のゆらぎは 74 領域で最も遅い（右 1 位、左 4 位。18/20 匹で他の領域より遅い）
- 予測: 触角葉のアセチルコリン性局所ニューロンは、ハエの自発活動の「ハブ」である（本物のハエで検証できる）

否定的・弱い結果（睡眠回路には特に届かない、領域ごとのモデルと実データの対応は弱い、など）と限界も論文に書いています。数値の一次資料は [notes/results_summary.md](notes/results_summary.md)。

## 成果物

| 成果物 | ファイル |
|---|---|
| ① 論文（Nature 投稿形式、英語） | [paper/paper_en.pdf](paper/paper_en.pdf)（原稿: [paper/paper_en_src.html](paper/paper_en_src.html)。`paper/build_paper.py` で引用番号を付けて paper_en.html を生成） |
| ① 論文（日本語版） | [paper/paper_ja.pdf](paper/paper_ja.pdf) |
| ② 発表資料（10 分＋Appendix、デジタル庁デザインシステム準拠） | [slides/slides.pdf](slides/slides.pdf)、発表原稿と想定問答 [slides/script.md](slides/script.md) |
| ③ 予備知識なしで読める解説 | [explainer/explainer.pdf](explainer/explainer.pdf)（元: [explainer/explainer.html](explainer/explainer.html)） |
| ④ 研究動画（日本語字幕、約 2.5 分） | [video/fly_culture_ignition_ja.mp4](video/fly_culture_ignition_ja.mp4) |
| 図（Nature 形式、PDF/SVG/PNG） | [figures/](figures/) |
| 数値のまとめ・文献メモ | [notes/results_summary.md](notes/results_summary.md)、[notes/culture_results.md](notes/culture_results.md)、[notes/literature.md](notes/literature.md) |

## フォルダ構成

```
data/          公開データ（リポジトリには含めない。下の「再現のしかた」で取得）
src/
  sim.py            GPU 版の全脳 LIF シミュレータ（Shiu ら 2024 の Brian 2 版と一致。前の研究から流用）
  spont.py          自発活動用の拡張（全ニューロンへのノイズ、ゲイン、短期シナプス抑圧、試行ごとの沈黙化）と分岐比の MR 推定量
  variants.py       伝達物質の符号の仮定（最小補正・細胞タイプ多数決・モノアミンの速い作用なし）
  culture/          培養データの解析（バースト検出、点火点、予測因子、分岐比、発達、ドリフト）
  fly/              ハエのモデルの実験と解析
    bursts.py         ゲインごとの自発バースト（cell type × 半球ごとの活動、仮想 MEA）
    ignition.py       全スパイク記録からの点火解析（開始前の過剰発火、閾値を越える時刻）
    structure.py      配線の主モード（ペロン固有ベクトル、符号つき行列の主モード）
    ladder.py         短期抑圧なしで自力で続く活動が最初に現れる回路
    silence.py, silence_confirm.py  黙らせる実験（予備・確認）
    eln_pn_control.py 投射ニューロンへの化学シナプスを除いた対照
    functional_eig.py 培養と同じ「活動の主モード」の検定をモデルに適用
    turner_*.py       本物のハエの安静時イメージング（Turner ら 2021）の解析
    record_*.py       点火解析・動画用の全スパイク記録
  figures.py        論文の図（Nature の体裁、style.py）
results/       解析結果（CSV、parquet、npz）
figures/       論文の図
paper/ slides/ explainer/ video/  成果物
notes/         テーマ選定、数値のまとめ、培養の解析メモ、文献メモ
```

## 再現のしかた

入力データはすべて公開データで、リポジトリには含めていません（計 約 4.5 GB）。

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git data/shiu_model
```

```bash
curl -L -o data/neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
```

```bash
mkdir -p data/flywire && for f in per_neuron_neuropil_count_pre_783.feather per_neuron_neuropil_count_post_783.feather; do curl -L -o data/flywire/$f https://zenodo.org/api/records/10676866/files/$f/content; done
```

```bash
mkdir -p data/culture/dandi002020 && cd data/culture/dandi002020 && uvx --from dandi dandi download DANDI:002020/0.261004.1410
```

Wagenaar ら 2006 の dense 培養のスパイク（約 0.7 GB）は https://potterlab.bme.gatech.edu/development-data/ の「daily / spont / dense」の `.spk.txt.bz2` を data/culture/potter_dense/ に置きます。Turner ら 2021 のデータは figshare（https://doi.org/10.6084/m9.figshare.13349282）の tar から `data/ito_responses`、`data/ito_68_atlas` を data/fly_imaging/turner2021/ に展開します（pickle は src/fly/turner_load.py の制限つき unpickler で読みます）。

```bash
uv sync
```

```bash
cd src/fly && ../../.venv/bin/python regions.py && ../../.venv/bin/python structure.py && ../../.venv/bin/python ladder.py && for v in fixAL typevote nomod; do ../../.venv/bin/python bursts.py $v 1.0 1.25 1.5 1.75 2.0 2.5; done
```

その後、`record_video.py`（例: `record_video.py fixAL 2.5`）、`record_robust.py`、`ignition.py`、`functional_eig.py`、`silence.py`、`silence_confirm.py`、`eln_pn_control.py`、`analyze_bursts.py`、`turner_metrics.py`、`turner_homolog.py` を実行し、培養は `src/culture/run_sessions.py` → `summarize.py` → `extra_controls.py` → `c3_eigen.py`、図は `src/figures.py fig1 fig2 fig3 fig4 ed1 ed2 ed3 ed4` です。GPU（CUDA）が必要です。RTX 5070 Ti で全体 約 2 時間です。PDF は `src/topdf.sh 入力.html 出力.pdf`（Google Chrome のヘッドレス印刷）で作ります。

## 使ったデータとコード

- 培養: DANDI 002020（Nakamuta, Akita, Zhang, Kawahara, Takahashi, Teramae; CC-BY-4.0）、Wagenaar, Pine & Potter 2006
- ハエ: FlyWire v783 の結合表と Shiu らのモデル（https://github.com/philshiu/Drosophila_brain_model）、FlyWire の細胞注釈（https://github.com/flyconnectome/flywire_annotations）、FlyWire のニューロピルごとのシナプス数（Zenodo 10676866）、Turner, Mann & Clandinin 2021 の安静時イメージング（MIT License）
- デジタル庁デザインシステムのトークン（MIT License）: https://github.com/digital-go-jp/design-tokens
