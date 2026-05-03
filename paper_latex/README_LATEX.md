# Lost in Compaction --- LaTeX project

This directory contains the LaTeX sources for the paper
**"Lost in Compaction: Measuring Information Loss in LLM Context Summaries"**.

The project compiles with **pdflatex** (Overleaf default) on UTF-8 sources
and uses the Kourgeorge `arxiv.sty` preprint style.

## Files

```
paper_latex/
  main.tex                     # top-level driver
  arxiv.sty                    # arXiv preprint style (Kourgeorge)
  references.bib               # BibTeX entries
  README_LATEX.md              # this file
  figures/                     # PNG figures referenced from sections/
  sections/
    00_abstract.tex
    01_introduction.tex
    02_framework.tex
    03_calibration_results.tex
    04_controlled_compaction.tex
    05_singlepass_results.tex
    06_strategy_comparison.tex
    07_discussion.tex
    08_future_work.tex
    09_conclusion.tex
    10_reproducibility.tex
    appendix_a_prompts.tex
    appendix_b_predictive_model.tex
```

## Compiling on Overleaf

1. **Zip the project**

   From the parent directory (`COMPACT_BENCHMARK/`):

   ```bash
   cd paper_latex
   zip -r ../paper_latex.zip .
   ```

   (Or just right-click the `paper_latex/` folder in Windows Explorer
   and choose *Send to* / *Compressed (zipped) folder*.)

2. **Upload to Overleaf**

   - Sign in at <https://www.overleaf.com>.
   - Click **New Project** $\to$ **Upload Project**.
   - Drag the ZIP into the dialog.
   - Overleaf unpacks the archive into a new project.

3. **Compile**

   - Make sure the **Compiler** is set to *pdfLaTeX* (Menu $\to$ Settings).
   - The **Main document** should be `main.tex` (auto-detected).
   - Click *Recompile*. The first pass writes `.aux` files; the second
     pass resolves cross-references and citations. Overleaf does the
     two passes automatically when you click *Recompile* once.

   You should see warnings about undefined references on the first
   build --- they disappear on the second. Bibliography errors mean
   `references.bib` was not picked up; check it sits next to `main.tex`.

## Producing the arXiv tarball

When the PDF compiles cleanly:

1. Menu $\to$ **Submit** $\to$ **Download Source**. Overleaf produces
   a `.zip` containing all source files (`.tex`, `.sty`, `.bib`,
   `figures/...`).
2. arXiv accepts a `.tar.gz` or a `.zip`. The Overleaf zip is fine
   for direct upload to <https://arxiv.org/submit>.
3. arXiv recompiles your sources on its servers --- it must be able
   to find every `\input{...}` file and every figure. The compile log
   on arXiv is verbose; if it fails, the message usually points at
   the offending line in `main.tex` or one of the `sections/*.tex`
   files.

## Local pdflatex (optional)

If you ever install TeX Live locally:

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

The standard four-step incantation. On Overleaf the tooling collapses
this into a single *Recompile* button.

## Editing tips

- Each section is a standalone file under `sections/`. Edit just that
  file; `main.tex` rarely needs to change.
- Figures live in `figures/` and are referenced by **filename only**
  (no path) thanks to `\graphicspath{{figures/}}`.
- Add a new reference: append a `@article{...}` entry to
  `references.bib`, then cite with `\citep{key}` or `\citet{key}`
  (natbib).
- The `\code{...}` macro typesets inline code in monospace. Use it for
  filenames, command names, identifiers. Inside a `lstlisting` block,
  you don't need `\code{}`.
- Special characters needing escaping in body text: `& % $ # _ { } ~ ^ \`
  (the underscore is the most common gotcha in filenames).
