# UNGA Resolution Citation Network (UN General Assembly)

国連総会（UNGA）で採択される決議文書は、前文部で過去の決議に対して **recalling / reaffirming / welcoming** 等の定型句（clause starter）を用いて言及する。本リポジトリは、決議文書をノード、言及を有向エッジとして抽出し、言及ネットワークの概観・集中度・時系列推移・年次ホットトピックを可視化するためのスクリプト群を提供する。

---

## ディレクトリ構成（想定）
undl_dump/
symbols.txt                 # 決議シンボル一覧（例：A/RES/79/330）
symbol_title.tsv            # symbol と title（+ optional columns）
records.jsonl               # 取得済みメタデータ（任意）
pdf_en/                     # 英語PDF格納
edges.csv                   # 抽出エッジ（raw）
edges_reclassified.csv      # 再分類済みエッジ（relation_group列あり）
overview_out/               # overview出力（json/png/tsv）
pareto_out/                 # パレート図出力
lineplot/                   # 折れ線・hot_topics・ヒートマップ出力

解析の流れ（Pipeline）

1) 決議シンボル（A/RES/…）とタイトルを取得
	•	download_resolution_symbols.py

出力（例）：
	•	undl_dump/symbols.txt
	•	undl_dump/symbol_title.tsv

⸻

2) 決議PDFをダウンロード
	•	download_pdfs_incremental.py

挙動：
	•	symbols.txt を読み込み、未ダウンロード分のみ差分で取得
	•	records.jsonl に英語PDF URLがあればそれを優先（無ければODS APIへフォールバック）
	•	PDFは undl_dump/pdf_en/ に保存

⸻

3) エッジ抽出（PDF→テキスト→Clause切り出し→決議番号抽出）
	•	extract_edges_from_pdfs_pypdf.py

ポイント：
	•	clause starter から節を切り出し、その節中に現れる決議番号のみを抽出することで脚注等のノイズを抑制
	•	VALUE_NEUTRAL / STANCE の両カテゴリに含まれる表現を抽出対象にする

出力：
	•	undl_dump/edges.csv（raw。relation列は生の表現）

⸻

4) エッジの再分類（raw→relation_group付与）
	•	reclassify_edges.py

出力：
	•	undl_dump/edges_reclassified.csv

列（例）：
	•	citing_symbol
	•	cited_symbol
	•	relation（生の表現）
	•	relation_group（value_neutral / stance / other / unspecified）

⸻

5) 生の言及表現の内訳（lemma正規化した比率）
	•	analyze_relation_breakdown_raw.py

目的：
	•	edges.csv を読み、affirm/affirming/affirmed のような活用差を affirm に寄せて集計
	•	表現の構成比を可視化（例：ツリーマップ）

⸻

分析・可視化

A) グラフ全体の概観（in-degree分布など）
	•	analyze_graph_overview.py

入力：
	•	undl_dump/edges_reclassified.csv

出力（例）：
	•	undl_dump/overview_out/overview_summary.json
	•	undl_dump/overview_out/in_degree_all.png
	•	undl_dump/overview_out/in_degree_value_neutral.png
	•	undl_dump/overview_out/in_degree_stance.png
	•	undl_dump/overview_out/top10_in_degree_*.tsv

⸻

B) パレート図（Top N + 累積割合 + 上位10件の凡例）
	•	plot_indegree_pareto.py

入力：
	•	undl_dump/edges_reclassified.csv
	•	undl_dump/symbol_title.tsv

出力：
	•	undl_dump/pareto_out/pareto_top50_all.png
	•	undl_dump/pareto_out/pareto_top50_value_neutral.png
	•	undl_dump/pareto_out/pareto_top50_stance.png

注：
	•	cumulative percentage は 全被言及数（全ノード） を分母として計算する設定を推奨

⸻

C) 直近X年の推移（Top10の折れ線）+ 年次ホットトピック抽出
	•	analyze_recent_trends_lineplot.py

入力：
	•	undl_dump/edges_reclassified.csv
	•	undl_dump/symbol_title.tsv

出力：
	•	undl_dump/lineplot/top10_all_last_10_years.png 等
	•	undl_dump/lineplot/hot_topics_all_last_10_years.csv 等（期間常連は除外）

⸻

D) 年次ホットトピックのヒートマップ
	•	heatmap_hot_topics.py

入力：
	•	undl_dump/lineplot/hot_topics_all_last_10_years.csv
	•	undl_dump/lineplot/hot_topics_stance_last_10_years.csv
	•	undl_dump/lineplot/hot_topics_value_neutral_last_10_years.csv

出力：
	•	undl_dump/lineplot/heatmap_hot_topics_*.png

<<<<<<< Updated upstream
## Data source
UN Official Document System (ODS)
=======

>>>>>>> Stashed changes
