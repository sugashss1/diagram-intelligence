# Knowledge-base materials

Place only local, project-approved retrieval materials in this folder. The VQA
stage indexes this folder recursively and does not pull information from the
internet or any other repository location.

Suggested organisation:

- symbols/ — JSON or Markdown symbol descriptions and libraries.
- rules/ — JSON or Markdown engineering/design rules.
- documents/ — PDFs, text manuals, and DOCX documentation.

Supported inputs are PDF, JSON, Markdown, plain text, reStructuredText, CSV,
YAML, and DOCX. Run VQA with:

    python main.py "test/full-adder-circuit.png" --question "What does this circuit do?"

The pipeline persists a deterministic local hybrid-search index at
output/retrieval/knowledge_index.json. Use --rebuild-knowledge-index after
changing these materials.
