# Manuscript sources

Build the main manuscript and supplement with a TeX installation that includes
`latexmk`:

```sh
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error manuscript/manuscript.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error manuscript/supplementary.tex
```

## References

The main manuscript contains its reference list directly in `manuscript.tex`,
using the Nature Portfolio bibliography style and superscript numerical citations.
Its normal build does not need BibTeX, a `.bib` file or a `.bbl` file.
The journal's [final-source requirements](https://www.nature.com/nmeth/submission-guidelines/aip-and-formatting)
request embedded references; initial submission allows the compiled PDF.

Keep `bibliography.bib` as the editable reference database. After changing a
reference or adding, removing or reordering citations, run from the repository root:

```sh
python3 manuscript/update_bibliography.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error manuscript/manuscript.tex
```

The update command requires BibTeX, generates the bibliography in a temporary
directory and replaces only the marked bibliography block. It checks citation
order and rejects BibTeX warnings or unresolved placeholders. Do not edit the
generated block directly. Citation keys are stable identifiers; their year suffix
may refer to an earlier version, e.g. `CZI2023` now cites the 2025 journal article.

`sn-nature.bst` contains a local correction for conference entries: it handles
missing editors without a BibTeX stack error, renders editors once, and keeps
conference-paper titles in roman type. Publisher locations are omitted when not
provided, rather than filled with placeholders.

Metadata corrections were checked against publisher records for
[CELLxGENE](https://doi.org/10.1093/nar/gkae1142),
[scIB](https://doi.org/10.1038/s41592-021-01336-8),
[Human Cell Atlas](https://doi.org/10.7554/eLife.27041),
[Cell Ontology](https://doi.org/10.1186/gb-2005-6-2-r21),
[Scanpy](https://doi.org/10.1186/s13059-017-1382-0),
[scTab](https://doi.org/10.1038/s41467-024-51059-5),
[zero-shot evaluation](https://doi.org/10.1186/s13059-025-03574-x), and
[Pascanu et al.](https://proceedings.mlr.press/v28/pascanu13.html).
Other preprint citations retain the selected preprint versions and are explicitly
labelled as such; this is not a claim that every preprint lacks a later publication.
