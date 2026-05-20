"""
Lightweight Word-Level Tokenizer for Dragon Fruit VQA.

Designed for a narrow agricultural question domain (~500 tokens).
Supports serialization to JSON for edge deployment.

Special tokens:
    <PAD> = 0
    <UNK> = 1
"""

from __future__ import annotations
import json, re, os
from typing import Dict, List, Optional


# ─── SPECIAL TOKENS ─────────────────────────────────────────────────────────
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
PAD_ID     = 0
UNK_ID     = 1


class VQATokenizer:
    """
    Minimal word-level tokenizer.

    Splits on whitespace + punctuation, lowercases, maps to integer IDs.
    Total vocabulary is typically ~400–500 tokens for our question templates.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.word2id: Dict[str, int] = dict(vocab)
        else:
            self.word2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2word: Dict[int, str] = {v: k for k, v in self.word2id.items()}

    # ── Vocabulary Construction ──────────────────────────────────────────
    @classmethod
    def build_from_texts(cls, texts: List[str]) -> "VQATokenizer":
        """
        Build vocabulary from a list of raw question strings.

        Args:
            texts: List of question strings (e.g., all question templates).

        Returns:
            Fitted VQATokenizer instance.
        """
        vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        next_id = 2

        for text in texts:
            for token in cls._tokenize(text):
                if token not in vocab:
                    vocab[token] = next_id
                    next_id += 1

        return cls(vocab)

    # ── Tokenization ─────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Lowercase + split on non-alphanumeric characters.
        Keeps apostrophes inside words (e.g., "what's" → ["what's"]).
        """
        text = text.lower().strip()
        # Split on whitespace and non-word chars, but keep apostrophes in words
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        return tokens

    def encode(
        self,
        text: str,
        max_len: int = 32,
        padding: bool = True,
    ) -> List[int]:
        """
        Encode a question string to a list of integer token IDs.

        Args:
            text:    Raw question string.
            max_len: Maximum sequence length (pad or truncate to this).
            padding: If True, pad with PAD_ID to max_len.

        Returns:
            List of integer token IDs with length exactly max_len.
        """
        tokens = self._tokenize(text)
        ids = [self.word2id.get(t, UNK_ID) for t in tokens]

        # Truncate if too long
        ids = ids[:max_len]

        # Pad if too short
        if padding:
            ids += [PAD_ID] * (max_len - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a human-readable string."""
        tokens = [self.id2word.get(i, UNK_TOKEN) for i in ids if i != PAD_ID]
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    # ── Serialization ────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save vocabulary to JSON for edge deployment."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "VQATokenizer":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Ensure integer values
        vocab = {k: int(v) for k, v in vocab.items()}
        return cls(vocab)

    def __repr__(self) -> str:
        return f"VQATokenizer(vocab_size={self.vocab_size})"


# ─── QUESTION TEMPLATES ─────────────────────────────────────────────────────
# All question templates used for synthetic dataset generation.
# These also define the vocabulary for the tokenizer.

QUESTION_TEMPLATES = {
    "diagnosis": [
        "What disease is this?",
        "Can you identify the problem?",
        "Is this plant sick?",
        "What is wrong with this dragon fruit?",
        "What disease does this plant have?",
        "Can you diagnose this plant?",
        "What condition is this fruit in?",
        "Identify the disease in this image.",
        "Is this plant infected?",
        "What's the matter with this plant?",
        "Tell me what disease this is.",
        "What infection does this show?",
    ],
    "severity": [
        "How serious is this?",
        "Is this rot severe?",
        "What is the risk level?",
        "How bad is the damage?",
        "Is this dangerous for the plant?",
        "Should I be worried?",
        "What is the severity level?",
        "How urgent is this problem?",
        "Is this a serious condition?",
        "What is the severity score?",
    ],
    "treatment": [
        "What fungicide should I use?",
        "How do I treat this?",
        "What spray is recommended?",
        "What medicine should I apply?",
        "How can I cure this disease?",
        "What chemicals should I use?",
        "What is the treatment for this?",
        "How to fix this problem?",
        "What should I spray on this?",
        "What dosage of fungicide is needed?",
        "Recommend a treatment plan.",
        "What bactericide should I apply?",
    ],
    "prevention": [
        "How can I prevent this?",
        "What precautions should I take?",
        "How to stop it from spreading?",
        "How do I protect my plants?",
        "What preventive measures are recommended?",
        "How to avoid this disease in future?",
        "What should I do to prevent recurrence?",
        "How can I keep my plants safe?",
        "What steps prevent this disease?",
        "How to protect the orchard?",
    ],
    "pathogen": [
        "What organism causes this?",
        "Is this a fungal or bacterial disease?",
        "What is the pathogen?",
        "What causes this disease?",
        "What microorganism is responsible?",
        "Tell me about the pathogen.",
        "Is this caused by a fungus?",
        "What is the causal agent?",
    ],
}

# Flatten all templates for vocabulary building
ALL_QUESTION_TEXTS = [q for templates in QUESTION_TEMPLATES.values() for q in templates]


def build_default_tokenizer() -> VQATokenizer:
    """Build and return a tokenizer fitted on all question templates."""
    return VQATokenizer.build_from_texts(ALL_QUESTION_TEXTS)


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok = build_default_tokenizer()
    print(tok)
    print(f"Vocabulary size: {tok.vocab_size}")

    test_questions = [
        "What disease is this?",
        "How do I treat this?",
        "Is this rot severe?",
        "What fungicide should I use for anthracnose?",
    ]
    for q in test_questions:
        ids = tok.encode(q, max_len=16)
        decoded = tok.decode(ids)
        print(f"  Q: {q!r}")
        print(f"  IDs: {ids}")
        print(f"  Decoded: {decoded!r}")
        print()
