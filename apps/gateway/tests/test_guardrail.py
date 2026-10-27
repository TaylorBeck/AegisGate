"""Secret and PII scrubber coverage, including the 8KB latency budget."""

from __future__ import annotations

import time

from app.engine.guardrail import Guardrail
from app.schemas.openai import ChatMessage, ContentPart

_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)


def test_each_pattern_is_redacted() -> None:
    guard = Guardrail()
    samples = {
        "OPENAI": "sk-" + ("a" * 32),
        "ANTHROPIC": "sk-ant-" + ("b" * 24),
        "AWS": "AKIAIOSFODNN7EXAMPLE",
        "GITHUB": "ghp_" + ("c" * 36),
        "JWT": _JWT,
        "SSN": "219-09-9999",
        "CARD": "4111111111111111",
    }
    assert len(samples["AWS"]) == 20
    for kind, secret in samples.items():
        scrubbed = guard.scrub(f"before {secret} after")
        assert scrubbed == f"before [REDACTED_{kind}] after"
        assert secret not in scrubbed


def test_grouped_card_and_github_fine_grained_token() -> None:
    guard = Guardrail()
    assert guard.scrub("pay 4111-1111-1111-1111 now") == "pay [REDACTED_CARD] now"
    token = "github_pat_" + ("d" * 22)
    assert guard.scrub(token) == "[REDACTED_GITHUB]"


def test_luhn_false_positive_is_preserved() -> None:
    guard = Guardrail()
    plain = "reference 4111111111111112 is not a card"
    grouped = "reference 4111-1111-1111-1112 is not a card"
    assert guard.scrub(plain) == plain
    assert guard.scrub(grouped) == grouped


def test_scrub_is_idempotent_and_does_not_mutate_messages() -> None:
    guard = Guardrail()
    original = "ssn 219-09-9999 and sk-" + ("a" * 32)
    message = ChatMessage(role="user", content=original)
    scrubbed = guard.scrub_messages([message])
    assert message.content == original
    assert isinstance(scrubbed[0].content, str)
    assert "219-09-9999" not in scrubbed[0].content
    assert guard.scrub(scrubbed[0].content) == scrubbed[0].content


def test_content_parts_and_extra_fields_are_scrubbed() -> None:
    guard = Guardrail()
    message = ChatMessage(
        role="user",
        content=[ContentPart(type="text", text="token sk-" + ("e" * 32))],
    )
    scrubbed = guard.scrub_messages([message])
    content = scrubbed[0].content
    assert isinstance(content, list)
    assert content[0].text == "token [REDACTED_OPENAI]"

    extra = ChatMessage.model_validate(
        {"role": "user", "content": "hello", "note": "ssn 219-09-9999"}
    )
    dumped = guard.scrub_messages([extra])[0].model_dump()
    assert dumped["note"] == "ssn [REDACTED_SSN]"
    assert dumped["role"] == "user"


def test_eight_kilobyte_prompt_stays_under_budget() -> None:
    guard = Guardrail()
    secret = " sk-" + ("a" * 32) + " "
    filler = "Quarterly revenue increased across the enterprise segment. "
    body = filler * (1 + 8192 // len(filler))
    prompt = (secret + body)[:8192]
    assert len(prompt) == 8192
    assert "[REDACTED_OPENAI]" in guard.scrub(prompt)

    for _ in range(5):
        guard.scrub(prompt)
    samples: list[float] = []
    for _ in range(30):
        started = time.perf_counter_ns()
        guard.scrub(prompt)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    samples.sort()
    median_ms = samples[len(samples) // 2]
    assert median_ms < 0.4, f"guardrail median {median_ms:.3f}ms"
