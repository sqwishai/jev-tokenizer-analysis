from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

import regex
import tiktoken
from huggingface_hub import hf_hub_download


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_U2B: dict[str, int] = {v: k for k, v in _bytes_to_unicode().items()}


def _decode_byte_level(token: str) -> bytes:
    return bytes(_U2B[ch] for ch in token)


def load_qwen35_tokenizer_json(
    repo_id: str = "Qwen/Qwen3.5-9B",
    local_path: str | Path | None = None,
) -> dict:
    path = Path(local_path) if local_path is not None else Path(hf_hub_download(repo_id, "tokenizer.json"))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def all_substrings_before_last_byte_in(cand: bytes, allowed_set: set[bytes]) -> bool:
    n = len(cand)
    for i in range(n):
        for j in range(i + 1, n):
            if cand[i:j] not in allowed_set:
                return False
    return True


class JevBillingTokenizer:
    def __init__(
        self,
        qwen_repo_id: str = "Qwen/Qwen3.5-9B",
        qwen_tokenizer_path: str | Path | None = None,
    ) -> None:
        qwen_json = load_qwen35_tokenizer_json(qwen_repo_id, qwen_tokenizer_path)
        self.o200k = tiktoken.get_encoding("o200k_base")
        self.o200k_ranks: dict[bytes, int] = dict(self.o200k._mergeable_ranks)

        pat_str = qwen_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pretokenizer = regex.compile(pat_str)

        self.qwen_vocab: dict[bytes, int] = {
            _decode_byte_level(tok): idx for tok, idx in qwen_json["model"]["vocab"].items()
        }
        self.qwen_reachable: set[bytes] = {bytes([b]) for b in range(256)}
        for merge in qwen_json["model"]["merges"]:
            a, b = merge if isinstance(merge, list) else merge.split(" ")
            self.qwen_reachable.add(_decode_byte_level(a) + _decode_byte_level(b))

        self.pass1_lookup: set[bytes] = set(self.o200k_ranks) & self.qwen_reachable
        self.pass2_formed_ranks: dict[bytes, int] = {
            tok: rank
            for tok, rank in self.o200k_ranks.items()
            if all_substrings_before_last_byte_in(tok, self.qwen_reachable)
        }

    def encode_pretoken(self, piece: bytes) -> list[bytes]:
        if piece in self.pass1_lookup:
            return [piece]

        parts = [piece[i : i + 1] for i in range(len(piece))]
        formed_ranks = self.pass2_formed_ranks
        while len(parts) > 1:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(parts) - 1):
                cand = parts[i] + parts[i + 1]
                rank = formed_ranks.get(cand)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts

    def encode_pieces(self, text: str) -> list[bytes]:
        normalized = unicodedata.normalize("NFC", text)
        out: list[bytes] = []
        for chunk in self.pretokenizer.findall(normalized):
            out.extend(self.encode_pretoken(chunk.encode("utf-8")))
        return out

    def count(self, text: str) -> int:
        return len(self.encode_pieces(text))

    def encode_many(self, texts: Iterable[str]) -> list[list[bytes]]:
        return [self.encode_pieces(t) for t in texts]
