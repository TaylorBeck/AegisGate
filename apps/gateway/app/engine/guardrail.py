"""Single-pass secret and PII scrubber.

One compiled alternation covers provider keys, AWS access keys, GitHub tokens,
JWTs, SSNs, and card numbers. Card matches are kept only when they pass Luhn.
"""

from __future__ import annotations

import re

from app.engine.coerce import object_dict, object_list
from app.schemas.openai import ChatMessage, ContentPart

_SCRUB = re.compile(
    r"(?P<ANTHROPIC>sk-ant-[A-Za-z0-9_\-]{20,})"
    r"|(?P<OPENAI>sk-[A-Za-z0-9_\-]{20,})"
    r"|(?P<AWS>AKIA[0-9A-Z]{16})"
    r"|(?P<GITHUB>github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"
    r"|(?P<JWT>eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})"
    r"|(?P<SSN>\b\d{3}-\d{2}-\d{4}\b)"
    r"|(?P<CARD>\b\d{4}(?:[ -]\d{4}){3}\b|\b\d{13,19}\b)",
    re.ASCII,
)
# Cheap locator so the alternation above runs only at a plausible start.
_CANDIDATE = re.compile(r"sk-|AKIA|github_pat_|gh[pousr]_|eyJ|\d", re.ASCII)


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = ord(char) - 48
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _replace(match: re.Match[str]) -> str:
    kind = match.lastgroup
    if kind is None:
        return match.group(0)
    if kind == "CARD":
        digits = "".join(char for char in match.group(0) if char.isdigit())
        if not _luhn_valid(digits):
            return match.group(0)
    return f"[REDACTED_{kind}]"


def _scrub_text(text: str) -> str:
    found = _CANDIDATE.search(text)
    if found is None:
        return text
    pieces: list[str] = []
    cursor = 0
    while found is not None:
        start = found.start()
        if start < cursor:
            start = cursor
        pieces.append(text[cursor:start])
        match = _SCRUB.match(text, start)
        if match is None or match.end() <= start:
            pieces.append(text[start : start + 1])
            cursor = start + 1
        else:
            pieces.append(_replace(match))
            cursor = match.end()
        found = _CANDIDATE.search(text, cursor)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _scrub_tree(value: object) -> object:
    if isinstance(value, str):
        return _scrub_text(value)
    values = object_list(value)
    if values is not None:
        return [_scrub_tree(item) for item in values]
    mapped = object_dict(value)
    if mapped is not None:
        return {key: _scrub_tree(item) for key, item in mapped.items()}
    return value


class Guardrail:
    """Scrubs message text that is hashed, logged, and forwarded upstream."""

    __slots__ = ()

    def scrub(self, text: str) -> str:
        return _scrub_text(text)

    def scrub_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        return [self.scrub_message(message) for message in messages]

    def scrub_message(self, message: ChatMessage) -> ChatMessage:
        if self._is_simple(message):
            return self._scrub_simple(message)
        dumped = message.model_dump(mode="json")
        scrubbed = _scrub_tree(dumped)
        return ChatMessage.model_validate(scrubbed)

    def _is_simple(self, message: ChatMessage) -> bool:
        if message.__pydantic_extra__:
            return False
        content = message.content
        if isinstance(content, list):
            return all(not part.__pydantic_extra__ for part in content)
        return True

    def _scrub_simple(self, message: ChatMessage) -> ChatMessage:
        content = message.content
        if isinstance(content, str):
            scrubbed = self.scrub(content)
            if scrubbed is content:
                return message
            return message.model_copy(update={"content": scrubbed})
        if isinstance(content, list):
            return self._scrub_parts(message, content)
        return message

    def _scrub_parts(
        self, message: ChatMessage, parts: list[ContentPart]
    ) -> ChatMessage:
        changed = False
        updated: list[ContentPart] = []
        for part in parts:
            if part.text is None:
                updated.append(part)
                continue
            scrubbed = self.scrub(part.text)
            if scrubbed is part.text:
                updated.append(part)
                continue
            changed = True
            updated.append(part.model_copy(update={"text": scrubbed}))
        if not changed:
            return message
        return message.model_copy(update={"content": updated})
