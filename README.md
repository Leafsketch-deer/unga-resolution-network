# UN General Assembly Resolution Citation Network

This repository contains code for extracting and analyzing citation
relationships among United Nations General Assembly resolutions.

Resolutions are modeled as nodes and inter-resolution references
(e.g. "Recalling", "Reaffirming") as directed edges.

## Data source
UN Official Document System (ODS)

## Main scripts
- extract_edges_from_pdfs_pypdf.py
- analyze_graph_overview.py
- analyze_recent_trends.py
- analyze_recent_trends_lineplot.py
