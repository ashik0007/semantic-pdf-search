# pdfs/

Place your PDF files here before running `python index_docs.py`.

This directory is listed in `.gitignore` and will **not** be committed to git.
Each PDF becomes part of the searchable index.

## Supported file types

- `.pdf` — primary format
- Other formats (`.txt`, `.docx`) can be added by modifying `index_docs.py`
  to remove the `required_exts=[".pdf"]` filter in `SimpleDirectoryReader`.

## Naming tips

Use descriptive filenames — they appear in query results as the source label.

Good:    `3GPP_TS_38_214_v17.4.0.pdf`
Generic: `spec.pdf`
