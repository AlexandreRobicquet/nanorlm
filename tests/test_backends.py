from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import build_parser as build_bench_parser
from bench import resolve_provider_choice
from examples.run_verifiers import build_parser as build_verifiers_parser
from nanorlm import (
    AnthropicMessagesBackend,
    ContextBlock,
    HeuristicBackend,
    MemoryItem,
    OpenAICompatibleBackend,
    RLM,
    RLMConfig,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def openai_payload(content: str, prompt_tokens: int = 3, completion_tokens: int = 2) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def anthropic_payload(content: str, input_tokens: int = 4, output_tokens: int = 3) -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": content}],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def memory_item(summary: str = "alpha summary") -> MemoryItem:
    return MemoryItem(
        summary=summary,
        provenance="alpha.txt",
        raw_pointer="root.0",
        tokens=8,
        depth=1,
        timestamp=1.0,
    )


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_provider_selection(self) -> None:
        self.assertIsInstance(RLM(RLMConfig(model="demo/heuristic", provider="heuristic")).backend, HeuristicBackend)
        self.assertIsInstance(RLM(RLMConfig(model="gpt-4.1-mini", provider="openai_compatible")).backend, OpenAICompatibleBackend)
        self.assertIsInstance(RLM(RLMConfig(model="claude-3-5-sonnet", provider="anthropic")).backend, AnthropicMessagesBackend)

    def test_auto_provider_selection(self) -> None:
        self.assertIsInstance(RLM(RLMConfig(model="demo/heuristic")).backend, HeuristicBackend)
        self.assertIsInstance(RLM(RLMConfig(model="claude-3-5-sonnet")).backend, AnthropicMessagesBackend)
        self.assertIsInstance(
            RLM(RLMConfig(model="qwen3:14b", base_url="http://localhost:11434/v1")).backend,
            OpenAICompatibleBackend,
        )
        self.assertIsInstance(
            RLM(RLMConfig(model="claude-3-5-sonnet", base_url="http://localhost:11434/v1")).backend,
            OpenAICompatibleBackend,
        )


class BackendTransportTests(unittest.TestCase):
    def _capture_requests(self, responses: list[dict[str, object]]) -> tuple[list[object], object]:
        requests: list[object] = []

        def fake_urlopen(request, timeout: int = 120):  # type: ignore[no-untyped-def]
            requests.append(request)
            return FakeHTTPResponse(responses[len(requests) - 1])

        return requests, fake_urlopen

    def test_openai_compatible_contracts(self) -> None:
        backend = OpenAICompatibleBackend(
            RLMConfig(
                model="gpt-4.1-mini",
                provider="openai_compatible",
                base_url="https://api.openai.com/v1",
                api_key="test-openai-key",
            )
        )
        requests, fake_urlopen = self._capture_requests(
            [
                openai_payload('{"summary":"branch","evidence":["alpha"],"answer_candidate":"beta","confidence":0.7}'),
                openai_payload("alpha.txt: beta"),
                openai_payload('{"score": 6.5}'),
                openai_payload('{"winner": "right"}'),
            ]
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            inspect = backend.inspect("What happened?", [ContextBlock(name="alpha.txt", text="alpha")], depth=1, branch="root.0")
            answer = backend.answer("What happened?", [memory_item()])
            score = backend.score_candidate("What happened?", memory_item())
            winner = backend.compare_candidates("What happened?", memory_item("left"), memory_item("right"))

        self.assertEqual(inspect.summary, "branch")
        self.assertEqual(answer.answer, "alpha.txt: beta")
        self.assertEqual(score, 6.5)
        self.assertEqual(winner, -1)
        self.assertEqual(len(requests), 4)
        first = requests[0]
        first_body = json.loads(first.data.decode("utf-8"))
        self.assertEqual(first.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(first.get_header("Authorization"), "Bearer test-openai-key")
        self.assertEqual(first_body["messages"][0]["role"], "system")
        self.assertEqual(first_body["messages"][1]["role"], "user")

    def test_anthropic_contracts(self) -> None:
        backend = AnthropicMessagesBackend(
            RLMConfig(
                model="claude-3-5-sonnet",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                api_key="test-anthropic-key",
            )
        )
        requests, fake_urlopen = self._capture_requests(
            [
                anthropic_payload('{"summary":"branch","evidence":["alpha"],"answer_candidate":"beta","confidence":0.7}'),
                anthropic_payload("alpha.txt: beta"),
                anthropic_payload('{"score": 6.5}'),
                anthropic_payload('{"winner": "left"}'),
            ]
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            inspect = backend.inspect("What happened?", [ContextBlock(name="alpha.txt", text="alpha")], depth=1, branch="root.0")
            answer = backend.answer("What happened?", [memory_item()])
            score = backend.score_candidate("What happened?", memory_item())
            winner = backend.compare_candidates("What happened?", memory_item("left"), memory_item("right"))

        self.assertEqual(inspect.summary, "branch")
        self.assertEqual(answer.answer, "alpha.txt: beta")
        self.assertEqual(score, 6.5)
        self.assertEqual(winner, 1)
        self.assertEqual(len(requests), 4)
        first = requests[0]
        first_body = json.loads(first.data.decode("utf-8"))
        self.assertEqual(first.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(first.get_header("X-api-key"), "test-anthropic-key")
        self.assertEqual(first.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(first_body["system"].split(".")[0], "You are a recursive language model worker")
        self.assertEqual(first_body["messages"][0]["role"], "user")

    def test_anthropic_base_url_with_v1_suffix_does_not_duplicate_version_path(self) -> None:
        backend = AnthropicMessagesBackend(
            RLMConfig(
                model="claude-3-5-sonnet",
                provider="anthropic",
                base_url="https://api.anthropic.com/v1",
                api_key="test-anthropic-key",
            )
        )
        requests, fake_urlopen = self._capture_requests([anthropic_payload("alpha.txt: beta")])
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = backend.answer("What happened?", [memory_item()])

        self.assertEqual(answer.answer, "alpha.txt: beta")
        self.assertEqual(requests[0].full_url, "https://api.anthropic.com/v1/messages")

    def test_json_repair_recovers_once(self) -> None:
        backend = OpenAICompatibleBackend(
            RLMConfig(
                model="qwen3:14b",
                provider="openai_compatible",
                base_url="http://localhost:11434/v1",
            )
        )
        requests, fake_urlopen = self._capture_requests(
            [
                openai_payload("```json\n{\"summary\": \"branch\"\n```", prompt_tokens=5, completion_tokens=2),
                openai_payload(
                    '{"summary":"branch","evidence":["alpha"],"answer_candidate":"beta","confidence":0.8}',
                    prompt_tokens=2,
                    completion_tokens=1,
                ),
            ]
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = backend.inspect("What happened?", [ContextBlock(name="alpha.txt", text="alpha")], depth=1, branch="root.0")

        self.assertEqual(result.summary, "branch")
        self.assertEqual(result.usage.prompt_tokens, 7)
        repair_body = json.loads(requests[1].data.decode("utf-8"))
        self.assertIn("Previous response", repair_body["messages"][1]["content"])

    def test_json_repair_raises_after_second_failure(self) -> None:
        backend = AnthropicMessagesBackend(
            RLMConfig(
                model="claude-3-5-sonnet",
                provider="anthropic",
                base_url="https://api.anthropic.com",
                api_key="test-anthropic-key",
            )
        )
        requests, fake_urlopen = self._capture_requests(
            [
                anthropic_payload("not json"),
                anthropic_payload("still not json"),
            ]
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as error:
                backend.score_candidate("What happened?", memory_item())

        self.assertIn("anthropic returned invalid JSON for score_candidate", str(error.exception))
        self.assertEqual(len(requests), 2)

    def test_localhost_openai_compatible_does_not_require_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "should-not-be-used"}, clear=True):
            engine = RLM(
                RLMConfig(
                    model="qwen3:14b",
                    provider="openai_compatible",
                    base_url="http://localhost:11434/v1",
                )
            )
            self.assertIsInstance(engine.backend, OpenAICompatibleBackend)
            requests, fake_urlopen = self._capture_requests([openai_payload("alpha.txt: beta")])
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = engine.backend.answer("What happened?", [memory_item()])

        self.assertEqual(result.answer, "alpha.txt: beta")
        self.assertIsNone(requests[0].get_header("Authorization"))


class CliTests(unittest.TestCase):
    def test_bench_provider_flag_and_alias(self) -> None:
        parser = build_bench_parser()
        anth = parser.parse_args(["--provider", "anthropic"])
        aliased = parser.parse_args(["--openai"])
        self.assertEqual(resolve_provider_choice(anth.provider, anth.openai), "anthropic")
        self.assertEqual(resolve_provider_choice(aliased.provider, aliased.openai), "openai_compatible")

    def test_verifiers_provider_flag_and_alias(self) -> None:
        parser = build_verifiers_parser()
        openai = parser.parse_args(["--provider", "openai-compatible"])
        aliased = parser.parse_args(["--openai"])
        self.assertEqual(resolve_provider_choice(openai.provider, openai.openai), "openai_compatible")
        self.assertEqual(resolve_provider_choice(aliased.provider, aliased.openai), "openai_compatible")


if __name__ == "__main__":
    unittest.main()
