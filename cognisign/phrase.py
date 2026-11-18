"""Sentence builder: chain gestures into a full sentence, then speak it.

Single gestures cover quick replies ("yes", "no", "thank you"). For anything
longer, gestures can add words to a buffer that's shown on screen and spoken
in one go, the same way AAC (augmentative and alternative communication)
boards work.
"""

from __future__ import annotations


class PhraseBuffer:
    def __init__(self, max_words: int = 40):
        self.max_words = max_words
        self._words: list[str] = []

    @property
    def words(self) -> list[str]:
        return list(self._words)

    @property
    def text(self) -> str:
        if not self._words:
            return ""
        sentence = " ".join(self._words)
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".?!":
            sentence += "."
        return sentence

    def add(self, word: str) -> None:
        word = word.strip()
        if not word:
            return
        if len(self._words) >= self.max_words:
            self._words.pop(0)
        self._words.append(word)

    def undo(self) -> str | None:
        return self._words.pop() if self._words else None

    def clear(self) -> None:
        self._words.clear()

    def take(self) -> str:
        """Return the finished sentence and empty the buffer."""
        text = self.text
        self.clear()
        return text

    def __len__(self) -> int:
        return len(self._words)
