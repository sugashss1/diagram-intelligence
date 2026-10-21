"""Local knowledge-base ingestion and dependency-free hybrid retrieval.

The retriever combines BM25 keyword scoring with TF-IDF cosine vector scoring.
It is deterministic, persists only extracted text/chunks, and does not require a
remote embedding service. A future embedding backend can be added without
changing the returned chunk contract.
"""

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree


SUPPORTED_SUFFIXES = {".pdf", ".json", ".md", ".txt", ".rst", ".csv", ".yml", ".yaml", ".docx"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenise(text):
    return TOKEN_PATTERN.findall(text.lower())


def _json_to_text(value, prefix=""):
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            lines.append(_json_to_text(item, child))
        return "\n".join(line for line in lines if line)
    if isinstance(value, list):
        return "\n".join(_json_to_text(item, f"{prefix}[{index}]") for index, item in enumerate(value))
    return f"{prefix}: {value}" if prefix else str(value)


def _read_docx(path):
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in document.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def _read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF knowledge-base ingestion requires the pypdf package.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_knowledge_document(path):
    """Read a supported document into plain text with actionable failures."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".json":
        return _json_to_text(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".docx":
        return _read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text, chunk_words, overlap_words):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def ingest_knowledge_base(root, chunk_words=250, overlap_words=40):
    """Ingest PDF, JSON, Markdown, text, and DOCX files beneath ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory does not exist: {root}")
    chunks = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES):
        text = read_knowledge_document(path).strip()
        if not text:
            continue
        for index, chunk in enumerate(_chunk_text(text, chunk_words, overlap_words), start=1):
            chunks.append(
                {
                    "id": f"{path.relative_to(root).as_posix()}#{index}",
                    "source": path.relative_to(root).as_posix(),
                    "chunk_index": index,
                    "text": chunk,
                }
            )
    if not chunks:
        raise ValueError(f"No readable supported knowledge documents were found under: {root}")
    return chunks


class HybridRetriever:
    """BM25 + TF-IDF cosine retriever over a local, serialisable chunk index."""

    def __init__(self, chunks, bm25_weight=0.5, vector_weight=0.5):
        if not chunks:
            raise ValueError("HybridRetriever requires at least one knowledge chunk.")
        if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight == 0:
            raise ValueError("Retrieval weights must be non-negative and not both zero.")
        self.chunks = chunks
        self.bm25_weight = bm25_weight / (bm25_weight + vector_weight)
        self.vector_weight = vector_weight / (bm25_weight + vector_weight)
        self.term_counts = [Counter(_tokenise(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = max(1.0, sum(self.lengths) / len(self.lengths))
        self.document_frequency = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())
        self.idf = {
            term: math.log((len(chunks) + 1) / (frequency + 1)) + 1
            for term, frequency in self.document_frequency.items()
        }
        self.vector_norms = [
            math.sqrt(sum((count * self.idf[term]) ** 2 for term, count in counts.items())) or 1.0
            for counts in self.term_counts
        ]

    def _bm25(self, query_counts, index, k1=1.5, b=0.75):
        score = 0.0
        counts, length = self.term_counts[index], self.lengths[index]
        for term, query_count in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * length / self.average_length)
            score += query_count * self.idf.get(term, 0.0) * numerator / denominator
        return score

    def _vector_cosine(self, query_counts, index):
        query_norm = math.sqrt(sum((count * self.idf.get(term, 0.0)) ** 2 for term, count in query_counts.items()))
        if query_norm == 0:
            return 0.0
        dot = sum(
            query_count * self.idf.get(term, 0.0) * self.term_counts[index].get(term, 0) * self.idf.get(term, 0.0)
            for term, query_count in query_counts.items()
        )
        return dot / (query_norm * self.vector_norms[index])

    @staticmethod
    def _normalise(scores):
        maximum = max(scores, default=0.0)
        return [score / maximum if maximum else 0.0 for score in scores]

    def search(self, query, top_k=5):
        """Return source-attributed chunks ranked by fused retrieval score."""
        query_counts = Counter(_tokenise(query))
        if not query_counts:
            return []
        bm25 = [self._bm25(query_counts, index) for index in range(len(self.chunks))]
        vectors = [self._vector_cosine(query_counts, index) for index in range(len(self.chunks))]
        bm25_normalised, vector_normalised = self._normalise(bm25), self._normalise(vectors)
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: self.bm25_weight * bm25_normalised[index] + self.vector_weight * vector_normalised[index],
            reverse=True,
        )[: max(0, int(top_k))]
        return [
            {
                **self.chunks[index],
                "bm25_score": round(bm25[index], 6),
                "vector_score": round(vectors[index], 6),
                "hybrid_score": round(self.bm25_weight * bm25_normalised[index] + self.vector_weight * vector_normalised[index], 6),
            }
            for index in ranked
            if bm25[index] > 0 or vectors[index] > 0
        ]


def save_index(chunks, path, source_root):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "knowledge-index/v1", "source_root": str(Path(source_root).resolve()), "chunks": chunks}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_index(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError(f"Knowledge index has no chunks: {path}")
    return chunks


def build_or_load_index(knowledge_base_path, index_path, rebuild=False):
    """Reuse a persisted index unless a caller explicitly requests rebuilding."""
    index_path = Path(index_path)
    if index_path.is_file() and not rebuild:
        return HybridRetriever(load_index(index_path))
    chunks = ingest_knowledge_base(knowledge_base_path)
    save_index(chunks, index_path, knowledge_base_path)
    return HybridRetriever(chunks)
