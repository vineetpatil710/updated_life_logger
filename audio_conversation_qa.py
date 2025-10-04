#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio → Conversation → Question Answering (Whisper + Lightweight RAG)

Usage:
  python audio_conversation_qa.py --audio "/path/to/audio_file.(wav|mp3|m4a|flac)" \
                                  --question "What did the speaker say about timelines?"

python audio_conversation_qa.py --audio "audio2.mp3" --question "explain iphone 17 pro"

What it does:
1) Transcribes the conversation from an audio file using OpenAI Whisper (offline).
2) Chunks the transcript with timestamps.
3) Retrieves the most relevant chunks for your question via semantic search.
4) Attempts extractive QA on each top chunk; if not confident, uses a generative fallback.
5) Prints the best answer and shows timestamped sources from the audio.

Dependencies (install if needed):   
  pip install openai-whisper transformers sentence-transformers torch numpy

Tip: For faster inference on supported hardware, install PyTorch with GPU support.
"""

import argparse #lets your script read command-line options like --input file.wav.
import re #“find/replace by pattern” (regular expressions).
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict #type hints to make code clearer.

import numpy as np #– fast math on arrays (vectors/matrices).
import torch #– PyTorch; runs neural nets on CPU/GPU.
import whisper #OpenAI’s speech-to-text (transcribes audio files).
from sentence_transformers import SentenceTransformer #turns sentences into vectors (embeddings) you can compare.
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------ Utilities ------------------------------ #

def human_time(s: float) -> str:
    """Seconds → hh:mm:ss.mmm"""
    if s is None or np.isnan(s):
        return "?:?:?.???"
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------ Data Types ------------------------------ #

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    seg_indices: List[int]


# ------------------------------ ASR (Whisper) ------------------------------ #

def transcribe_audio(
    audio_path: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe audio to segments (start, end, text) using openai-whisper.
    """
    device = device or device_auto()
    model = whisper.load_model(whisper_model, device=device)
    # Better timestamps & punctuation handling
    options = {"task": "transcribe", "fp16": (device == "cuda"), "language": language}
    result = model.transcribe(audio_path, **{k: v for k, v in options.items() if v is not None})

    segments = []
    for seg in result.get("segments", []):
        text = normalize_space(seg.get("text", ""))
        if text:
            segments.append(Segment(start=float(seg.get("start", 0.0)),
                                    end=float(seg.get("end", 0.0)),
                                    text=text))
    return segments


# ------------------------------ Chunking ------------------------------ #

def chunk_segments(
    segments: List[Segment],
    max_chars: int = 1200,
    overlap_chars: int = 200
) -> List[Chunk]:
    """
    Merge consecutive segments into text windows ~max_chars with controlled overlap.
    """
    chunks: List[Chunk] = []
    buf = []
    buf_len = 0
    buf_start = None
    buf_indices = []

    def flush():
        nonlocal chunks, buf, buf_len, buf_start, buf_indices
        if not buf:
            return
        text = normalize_space(" ".join(buf))
        start = buf_start
        end = current_end if buf_indices else start
        chunks.append(Chunk(start=start, end=end, text=text, seg_indices=buf_indices.copy()))
        buf, buf_len, buf_start, buf_indices = [], 0, None, []

    current_end = 0.0
    for i, seg in enumerate(segments):
        seg_text = seg.text
        seg_len = len(seg_text)
        if buf_len == 0:
            buf_start = seg.start
        if buf_len + seg_len <= max_chars:
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices.append(i)
            current_end = seg.end
        else:
            flush()
            # Overlap: carry last overlap_chars chars into new buffer start
            if chunks:
                carry_text = chunks[-1].text[-overlap_chars:]
                # Re-anchor carry within new chunk without timestamps (just for text continuity)
                buf.append(carry_text)
                buf_len = len(carry_text)
                buf_start = seg.start  # new chunk timestamp starts at current seg
            else:
                buf_len = 0
                buf_start = seg.start
            buf.append(seg_text)
            buf_len += seg_len + 1
            buf_indices = [i]
            current_end = seg.end

    flush()
    return chunks


# ------------------------------ Retrieval ------------------------------ #

class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device_auto() if device is None else device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embs

    def top_k(self, query: str, chunks: List[Chunk], k: int = 5) -> List[Tuple[int, float]]:
        chunk_texts = [c.text for c in chunks]
        q = self.encode([query])  # (1, d)
        c = self.encode(chunk_texts)  # (n, d)
        sims = (c @ q.T).squeeze(-1)  # cosine sims since normalized
        top_idx = np.argsort(-sims)[:max(1, min(k, len(chunks)))]
        return [(int(i), float(sims[i])) for i in top_idx]


# ------------------------------ Readers (Extractive + Generative) ------------------------------ #

class QAReader:
    def __init__(
        self,
        extractive_model: str = "deepset/roberta-base-squad2",
        generative_model: str = "google/flan-t5-large",
        device: Optional[str] = None
    ):
        self.device = device_auto() if device is None else device

        # Extractive pipeline (fast and precise when answer span exists)
        self.qa = pipeline(
            "question-answering",
            model=extractive_model,
            device=0 if self.device == "cuda" else -1
        )

        # Generative fallback (handles synthesis across multiple chunks)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generative_model)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model)
        if self.device != "cpu":
            self.gen_model = self.gen_model.to(self.device)

    def extractive_best(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]]
    ) -> Optional[Dict]:
        """
        Try extractive QA over top chunks; return best if confident.
        """
        best = None
        for chunk, sim in candidates:
            try:
                out = self.qa(question=question, context=chunk.text)
                score = float(out.get("score", 0.0))
                answer = normalize_space(out.get("answer", ""))
                if not answer:
                    continue
                entry = {
                    "answer": answer,
                    "score": score,
                    "start_char": int(out.get("start", -1)),
                    "end_char": int(out.get("end", -1)),
                    "chunk": chunk,
                    "retrieval_sim": sim
                }
                if best is None or entry["score"] > best["score"]:
                    best = entry
            except Exception:
                continue
        # Heuristic threshold: accept only confident spans
        if best and best["score"] >= 0.35:
            return best
        return None

    def generative_synthesize(
        self,
        question: str,
        candidates: List[Tuple[Chunk, float]],
        max_context_chars: int = 3000,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Build a compact prompt from top chunks and synthesize an answer.
        """
        # Concatenate contexts until limit
        acc = []
        used = 0
        sources = []
        for chunk, sim in candidates:
            t = f"[{human_time(chunk.start)}–{human_time(chunk.end)}] {chunk.text}"
            if used + len(t) > max_context_chars and acc:
                break
            acc.append(t)
            used += len(t)
            sources.append((chunk.start, chunk.end))
        context = "\n\n".join(acc)

        prompt = (
            "You are an expert conversation analyst. Answer the question ONLY using the context from the transcript.\n"
            "If you are not sure, provide the best supported answer from the given excerpts. Be concise and precise.\n\n"
            f"Question: {question}\n\n"
            f"Transcript excerpts:\n{context}\n\n"
            "Answer:"
        )

        inputs = self.gen_tokenizer([prompt], return_tensors="pt", truncation=True, max_length=4096)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                #  temperature=0.2,
                #  top_p=0.95
              
            )
        ans = self.gen_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        ans = ans.split("Answer:", 1)[-1].strip()
        return {"answer": ans, "score": None, "chunk": None, "sources": sources}


# ------------------------------ Orchestration ------------------------------ #

def map_span_to_timestamp(span_text: str, chunk: Chunk, segments: List[Segment]) -> Tuple[float, float]:
    """
    Approximate timestamps for an extractive span by locating it inside the chunk's segments.
    """
    # Simple heuristic: find the first segment inside chunk that contains a substantial part of the span.
    span = span_text.lower().strip()
    for idx in chunk.seg_indices:
        seg = segments[idx]
        txt = seg.text.lower()
        # Consider partial containment to be more robust
        if span in txt or any(w in txt for w in span.split()[:3]):
            return seg.start, seg.end
    # Fallback to chunk window
    return chunk.start, chunk.end


def answer_from_audio(
    audio_path: str,
    question: str,
    whisper_model: str = "small",
    language: Optional[str] = None,
    retriever_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    extractive_model: str = "deepset/roberta-base-squad2",
    generative_model: str = "google/flan-t5-large",
    top_k: int = 5
) -> Dict:
    # 1) ASR
    segments = transcribe_audio(audio_path, whisper_model=whisper_model, language=language)

    if not segments:
        return {"answer": "(No transcript produced.)", "sources": []}

    # 2) Chunking
    chunks = chunk_segments(segments, max_chars=1200, overlap_chars=200)

    # 3) Retrieval
    retriever = Retriever(model_name=retriever_name)
    ranked = retriever.top_k(question, chunks, k=top_k)
    candidates = [(chunks[i], sim) for i, sim in ranked]

    # 4) Readers
    reader = QAReader(extractive_model=extractive_model, generative_model=generative_model)

    # Try extractive first
    ext = reader.extractive_best(question, candidates)

    if ext:
        start_ts, end_ts = map_span_to_timestamp(ext["answer"], ext["chunk"], segments)
        return {
            "answer": ext["answer"],
            "confidence": round(float(ext["score"]), 4),
            "sources": [{
                "window": [human_time(ext["chunk"].start), human_time(ext["chunk"].end)],
                "precise_span": [human_time(start_ts), human_time(end_ts)]
            }]
        }

    # Fallback to generative synthesis
    gen = reader.generative_synthesize(question, candidates)
    return {
        "answer": gen["answer"],
        "confidence": None,
        "sources": [{
            "window": [human_time(s), human_time(e)]
        } for (s, e) in gen["sources"]]
    }


# ------------------------------ CLI ------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Ask questions about an audio conversation.")
    p.add_argument("-a", "--audio", help="Path to audio file (wav, mp3, m4a, flac, etc.)")
    p.add_argument("-q", "--question", help="Your question about the conversation.")
    p.add_argument("--whisper_model", default="small",
                   help="Whisper model size: tiny|base|small|medium|large-v2|large-v3 (default: small)")
    p.add_argument("--language", default=None, help="Force language code (e.g., 'en'); otherwise auto-detect.")
    p.add_argument("--top_k", type=int, default=5, help="Top K chunks to retrieve (default: 5)")
    p.add_argument("--extractive_model", default="deepset/roberta-base-squad2", help="HF model for extractive QA")
    p.add_argument("--generative_model", default="google/flan-t5-large", help="HF model for generative fallback")
    p.add_argument("--retriever", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer name")

    args, _ = p.parse_known_args()

    # Interactive fallback if launched without flags (e.g., “Run” button)
    if not args.audio:
        args.audio = input("Path to audio file: ").strip().strip('"')
    if not args.question:
        args.question = input("Your question: ").strip()

    return args



def main():
    args = parse_args()
    result = answer_from_audio(
        audio_path=args.audio,
        question=args.question,
        whisper_model=args.whisper_model,
        language=args.language,
        retriever_name=args.retriever,
        extractive_model=args.extractive_model,
        generative_model=args.generative_model,
        top_k=args.top_k
    )

    print("\n=== Answer ===")
    print(result.get("answer", "").strip() or "(No answer.)")
    if result.get("confidence") is not None:
        print(f"\nConfidence: {result['confidence']:.4f}")
    print("\n=== from the audio 1) The timestamp that is the source of the text answer. 2) The timestamp in the audio where the answer appears. ===")
    for i, src in enumerate(result.get("sources", []), 1):
        win = src.get("window", ["?", "?"])
        if "precise_span" in src:
            ps = src["precise_span"]
            print(f"{i}")
            print(f"{i} {win[0]} → {win[1]}   |   Span {ps[0]} → {ps[1]}")
        else:
            print(f"{i} {win[0]} → {win[1]}")
    print()


if __name__ == "__main__":
    main()
