"""Local tokenizer: learned text pieces plus a fixed UTF-8 byte fallback."""
import json
import re
from collections import Counter
from pathlib import Path
from .schema import QueryError

# Every character is covered. Digits remain separate so unseen numeric IDs work.
PIECES = re.compile(r"[A-Za-z]+|[0-9]|[ \t]+|\n|[^A-Za-z0-9 \t\n]")


class LocalTokenizer:
    pad_id, bos_id, eos_id = 0, 1, 2
    byte_offset = 3
    piece_offset = 259

    def __init__(self, pieces=None):
        self.pieces = list(pieces or [])
        if len(set(self.pieces)) != len(self.pieces):
            raise QueryError("Duplicate tokenizer pieces")
        self.lookup = {piece: i + self.piece_offset for i, piece in enumerate(self.pieces)}

    @classmethod
    def fit(cls, texts, max_vocab=4096):
        if max_vocab < cls.piece_offset:
            raise QueryError("Vocabulary must allow 256 bytes and 3 control tokens")
        counts = Counter(piece for text in texts for piece in PIECES.findall(text))
        # Single-byte pieces already exist in the byte alphabet.
        learned = [p for p, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                   if count >= 2 and len(p.encode("utf-8")) > 1]
        return cls(learned[:max_vocab - cls.piece_offset])

    def __len__(self):
        return self.piece_offset + len(self.pieces)

    def encode(self, text, max_tokens=None):
        ids = []
        for piece in PIECES.findall(text):
            known = self.lookup.get(piece)
            ids.extend([known] if known is not None else [b + self.byte_offset for b in piece.encode("utf-8")])
        ids.append(self.eos_id)
        if max_tokens is not None and len(ids) > max_tokens:
            raise QueryError(f"Text contains {len(ids)} tokens, exceeding the limit of {max_tokens}. Increase the appropriate input/output limit or reduce schema scope; truncation is disabled.")
        return ids

    def decode(self, ids):
        result = bytearray()
        for token in ids:
            if token == self.eos_id:
                break
            if token in (self.pad_id, self.bos_id):
                continue
            if self.byte_offset <= token < self.piece_offset:
                result.append(token - self.byte_offset)
            elif self.piece_offset <= token < len(self):
                result.extend(self.pieces[token - self.piece_offset].encode("utf-8"))
            else:
                raise QueryError("Invalid output token")
        try:
            return result.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QueryError("Model generated invalid UTF-8") from exc

    def save(self, path):
        Path(path).write_text(json.dumps({"format": "local-pieces-bytes-v1", "pieces": self.pieces}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "local-pieces-bytes-v1":
            raise QueryError("Unsupported tokenizer format")
        return cls(data["pieces"])
