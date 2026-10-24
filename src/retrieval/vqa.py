"""Multimodal RAG question answering over a recovered circuit."""
from openai import OpenAI
import base64
import json
import os
from pathlib import Path
from urllib import error, request

from image.connectivity_recovery import DEFAULT_INVOKE_URL, DEFAULT_MODEL
from text.circuit_json import topology_to_mermaid


def _image_part(path):
    path = Path(path)
    mime = "jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}


def _post_json(url, api_key, payload, timeout_seconds):
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

    completion = client.chat.completions.create(
    model="z-ai/glm-5.3-flash",
    messages=payload["messages"],
    temperature=1,
    top_p=0.95,
    max_tokens=262144,
    stream=False
    )

    return completion

def _response_text(payload):
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
    return str(content).strip()


def answer_circuit_question(
    question,
    circuit,
    topology,
    image_path,
    retriever,
    api_key=None,
    invoke_url=None,
    model_name=None,
    top_k=5,
    timeout_seconds=180,
):
    """Answer a user question using the original image, JSON graph, and retrieved KB text."""
    if not str(question).strip():
        raise ValueError("A non-empty VQA question is required.")
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Circuit image does not exist: {image_path}")
    api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY or NVAPI_KEY before invoking VQA.")
    invoke_url = invoke_url or os.environ.get("NVIDIA_INVOKE_URL", DEFAULT_INVOKE_URL)
    model_name = model_name or os.environ.get("NVIDIA_VLM_MODEL", DEFAULT_MODEL)
    mermaid = topology_to_mermaid(circuit, topology)
    retrieval_query = f"{question}\nComponents: {' '.join(component['kind'] for component in circuit.get('components', []))}\n{mermaid}"
    retrieved = retriever.search(retrieval_query, top_k=top_k)
    context = "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in retrieved)
    prompt = f"""You are an engineering-diagram assistant. Answer the user using the circuit image, the recovered graph, and retrieved engineering knowledge.

    User question:
    {question}

    Recovered Mermaid graph:
    ```mermaid
    {mermaid}```

    Canonical topology summary:
    {json.dumps(topology, ensure_ascii=False)}

    Retrieved knowledge:
    {context or '[No matching knowledge-base passages were retrieved.]'}

    Requirements:
    - Distinguish directly observed diagram facts from inferences.
    - State when unresolved crossings or incomplete recovery make an answer uncertain.
    - Do not claim electrical correctness solely from internal graph consistency.
    - Give a concise general-text answer with source filenames when knowledge-base evidence is used."""

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1200,
        "stream": False,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, _image_part(image_path)]}],
    }
    try:
        response = _post_json(invoke_url, api_key, payload, timeout_seconds)
        answer = response.choices[0].message.content
    except (OSError, KeyError, ValueError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"VQA invocation failed: {exc}") from exc
    return {"question": question, "answer": answer, "model": model_name, "retrieved_sources": retrieved, "mermaid": mermaid}


def save_vqa_response(response, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
