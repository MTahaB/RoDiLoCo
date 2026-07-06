# Paper — `rodiloco.tex`

The arXiv-style writeup (Phase 5). Compiles out-of-the-box with the standard `article` class;
no external style file needed to draft.

## Build

```bash
cd paper
pdflatex rodiloco
bibtex   rodiloco
pdflatex rodiloco
pdflatex rodiloco
```

Or, if you have `latexmk`:

```bash
latexmk -pdf rodiloco.tex
```

## Figures

Drop the plots produced by `scripts/plot_results.py` into `paper/figures/` as PDF, then
uncomment the `\includegraphics` lines (the `\fbox{...}` placeholders sit right above them):

| In the paper | Produced by |
|--------------|-------------|
| `figures/plotP2_comm.pdf` | `plot_results.py comm` (Fig. 1) |
| `figures/plot1_fragility.pdf` | `plot_results.py fragility` (Fig. 2 / plot #1) |
| `figures/plot2_defense.pdf` | `plot_results.py defense` (Fig. 3 left / plot #2) |
| `figures/plot3_tax.pdf` | `plot_results.py tax` (Fig. 3 right / plot #3) |

Save as PDF for vector figures, e.g. `plt.savefig("figures/plot1_fragility.pdf")`.

## Switching to the official template

Replace `\documentclass[11pt]{article}` + the `geometry` line with the ICML or NeurIPS style
file (`icml2024.sty` / `neurips_2024.sty`, a single drop-in file each) when you pick a venue.
The section structure already matches an 8-page workshop paper.

## Filling it in

Every number/result is a red `\todo{...}` marker. Search for `\todo` — each one maps to a
value your runs produce (`final_ppl`, `comm_total_bytes`, the HP table). The prose framing is
final; only the numbers, figures, and headline sentences remain.
