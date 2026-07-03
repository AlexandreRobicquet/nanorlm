from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from nanorlm import (
    ContextBlock,
    RLM,
    RLMConfig,
    RLMResult,
    item_source_paths,
    load_text_blocks,
    normalize_text,
    supports_cost_estimate,
    write_trace,
)


ROOT = Path(__file__).resolve().parent
CLI_PROVIDER_CHOICES = ["heuristic", "openai-compatible", "anthropic"]
DATASET_CHOICES = [
    "pairbench",
    "needlepairs",
    "dossierbench",
    "ruler_synthetic",
    "babilong_synthetic",
    "verifiers_30",
    "verifiers_smoke",
    "external_jsonl",
]
DEFAULT_POLICIES = [
    "direct_full_context",
    "keep_recent",
    "summary_only",
    "single_critic_topk",
    "pairwise_tournament",
    "learned_retention",
]


@dataclass(slots=True)
class BenchmarkExample:
    name: str
    query: str
    context: list[ContextBlock]
    answer: str
    must_contain: list[str]
    expected_provenance: list[str] = field(default_factory=list)
    task_class: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_anchor_blocks(path: str | Path, anchors: Sequence[str], window: int = 6) -> list[ContextBlock]:
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    if not lines:
        return [ContextBlock(name=file_path.name, text="", metadata={"path": str(file_path)})]
    blocks: list[ContextBlock] = []
    seen_ranges: set[tuple[int, int]] = set()
    lower_lines = [line.lower() for line in lines]
    for anchor in anchors:
        anchor_lower = anchor.lower()
        for index, line in enumerate(lower_lines):
            if anchor_lower in line:
                start = max(0, index - window)
                end = min(len(lines), index + window + 1)
                key = (start, end)
                if key in seen_ranges:
                    break
                seen_ranges.add(key)
                text = "\n".join(lines[start:end])
                blocks.append(
                    ContextBlock(
                        name=f"{file_path.name}:{start + 1}-{end}",
                        text=text,
                        metadata={"path": str(file_path)},
                    )
                )
                break
    if not blocks:
        return load_text_blocks(file_path, chunk_size_lines=24)[:1]
    return blocks


def score_answer(answer: str, must_contain: Sequence[str]) -> float:
    normalized = normalize_text(answer)
    return 1.0 if all(normalize_text(fragment) in normalized for fragment in must_contain) else 0.0


def score_provenance(result: RLMResult, expected_provenance: Sequence[str]) -> tuple[float, list[str]]:
    if not expected_provenance:
        return 0.0, []
    actual_paths: set[str] = set()
    for item in result.kept_items:
        actual_paths.update(item_source_paths(item))
        actual_paths.add(item.provenance)
    hits: list[str] = []
    for expected in expected_provenance:
        expected_lower = expected.lower()
        basename = Path(expected).name.lower()
        if any(expected_lower in path.lower() or basename in Path(path).name.lower() or basename in path.lower() for path in actual_paths):
            hits.append(expected)
    return round(len(hits) / len(expected_provenance), 3), hits


def compactness_score(retained_tokens: int, budget: int) -> float:
    if budget <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (retained_tokens / budget)), 3)


def retention_reward_score(
    *,
    answer_accuracy: float,
    provenance_score: float,
    compactness: float,
    latency_ms: float,
    cost_estimate: float,
) -> float:
    latency_penalty = min(0.12, latency_ms / 10000.0)
    cost_penalty = min(0.12, cost_estimate * 2.0)
    reward = (
        0.62 * answer_accuracy
        + 0.18 * provenance_score
        + 0.12 * compactness
        - latency_penalty
        - cost_penalty
    )
    return round(max(0.0, reward), 3)


def pair_words() -> list[str]:
    return [
        "amber",
        "comet",
        "frost",
        "lattice",
        "mango",
        "orbit",
        "quartz",
        "raven",
        "signal",
        "topaz",
        "vector",
        "willow",
        "yonder",
        "zephyr",
    ]


def build_pairbench(n: int = 100, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    words = pair_words()
    examples: list[BenchmarkExample] = []
    distractor_space = max(128, n + 64)
    for index in range(n):
        pair_id = f"pair-{index:03d}"
        left_value = words[index % len(words)]
        right_value = words[(index * 3 + 5) % len(words)]
        docs: list[ContextBlock] = []
        for distractor in range(18):
            distractor_pair = f"pair-{(index + distractor + 7) % distractor_space:03d}"
            if distractor_pair == pair_id:
                distractor_pair = f"pair-{(index + distractor + 37) % distractor_space:03d}"
            distractor_kind = "left" if distractor % 2 == 0 else "right"
            distractor_value = words[(index + distractor * 2) % len(words)]
            docs.append(
                ContextBlock(
                    name=f"notes/{distractor_pair}-{distractor_kind}-{distractor}.md",
                    text=(
                        f"PAIR_ID: {distractor_pair}\n"
                        f"FACT_KIND: {distractor_kind}\n"
                        f"FACT_VALUE: {distractor_value}\n"
                        "SLOT: memo\n"
                        f"{distractor_pair} {distractor_kind} token is {distractor_value}; belongs to another pair.\n"
                    ),
                )
            )
        docs.extend(
            [
                ContextBlock(
                    name=f"vault/{pair_id}-left.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: durable\n"
                        f"{pair_id} left token is {left_value}; combine with right token.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-right.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: right\n"
                        f"FACT_VALUE: {right_value}\n"
                        "SLOT: durable\n"
                        f"{pair_id} right token is {right_value}; combine with left token.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-left-duplicate.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: duplicate\n"
                        f"Duplicate: {pair_id} left token is {left_value}; tempts single-item ranking.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-left-archive.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: archive\n"
                        f"Archive: {pair_id} left token is {left_value}; competes for memory.\n"
                    ),
                ),
            ]
        )
        rng.shuffle(docs)
        examples.append(
            BenchmarkExample(
                name=pair_id,
                query=f"For {pair_id}, what is the full code? Combine the left token and the right token.",
                context=docs,
                answer=f"{left_value} {right_value}",
                must_contain=[left_value, right_value],
                task_class="complementary-facts",
            )
        )
    return examples


def build_needlepairs(n: int = 50, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    base = build_pairbench(n=n, seed=seed)
    examples: list[BenchmarkExample] = []
    filler = "Noise block. " * 50
    for example in base:
        padded: list[ContextBlock] = []
        for index in range(96):
            padded.append(ContextBlock(name=f"haystack/noise-{index:03d}.txt", text=filler + f" slot {index}"))
        padded.extend(example.context)
        rng.shuffle(padded)
        examples.append(
            BenchmarkExample(
                name=f"needle-{example.name}",
                query=example.query,
                context=padded,
                answer=example.answer,
                must_contain=example.must_contain,
                task_class="needle-haystack",
            )
        )
    return examples


def build_dossierbench(n: int = 24, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    services = ["api-gateway", "rollout-router", "eval-orchestrator", "browser-runner", "sandbox-manager", "prompt-builder"]
    root_causes = [
        "stale endpoint registry cache",
        "missing retry budget on env worker",
        "incompatible BrowserEnv extra install",
        "resume metadata mismatch",
        "non-increasing chat template regression",
        "sandbox teardown timeout leak",
    ]
    fixes = [
        "invalidate the endpoint cache on reload",
        "thread max_retries through the worker config",
        "move browser extras behind a dedicated optional path",
        "validate resume metadata before replay",
        "normalize the chat template before rollout",
        "tighten sandbox shutdown and retry classification",
    ]
    files = [
        "verifiers/clients/config.py",
        "verifiers/cli/commands/eval.py",
        "verifiers/envs/browser_env.py",
        "verifiers/utils/save_utils.py",
        "verifiers/utils/chat_template.py",
        "verifiers/envs/sandbox_env.py",
    ]
    owners = ["will", "sebastian", "infra", "envs", "evals", "clients"]
    types = ["incident", "migration", "release"]
    examples: list[BenchmarkExample] = []
    distractor_space = max(128, n + 64)

    for index in range(n):
        case_type = types[index % len(types)]
        case_id = f"{case_type}-{index:03d}"
        service = services[index % len(services)]
        root_cause = root_causes[index % len(root_causes)]
        fix = fixes[index % len(fixes)]
        file_path = files[index % len(files)]
        owner = owners[index % len(owners)]
        docs: list[ContextBlock] = []
        for distractor in range(24):
            distractor_id = f"{types[(index + distractor + 1) % len(types)]}-{(index + distractor + 9) % distractor_space:03d}"
            distractor_kind = "root_cause" if distractor % 2 == 0 else "file"
            distractor_value = root_causes[(index + distractor) % len(root_causes)] if distractor % 2 == 0 else files[(index + distractor) % len(files)]
            distractor_label = "root cause" if distractor_kind == "root_cause" else "patch file"
            docs.append(
                ContextBlock(
                    name=f"dossiers/{distractor_id}-memo-{distractor}.md",
                    text=(
                        f"CASE_ID: {distractor_id}\n"
                        f"FACT_KIND: {distractor_kind}\n"
                        f"FACT_VALUE: {distractor_value}\n"
                        "SLOT: distractor\n"
                        f"{distractor_id} {distractor_label}: {distractor_value}; belongs to another investigation.\n"
                    ),
                )
            )
        docs.extend(
            [
                ContextBlock(
                    name=f"dossiers/{case_id}-service.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: service\n"
                        f"FACT_VALUE: {service}\n"
                        "SLOT: durable\n"
                        f"{case_id} active service: {service}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"dossiers/{case_id}-root-cause.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: root_cause\n"
                        f"FACT_VALUE: {root_cause}\n"
                        "SLOT: durable\n"
                        f"{case_id} root cause and blocker: {root_cause}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"dossiers/{case_id}-fix.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: fix\n"
                        f"FACT_VALUE: {fix}\n"
                        "SLOT: durable\n"
                        f"{case_id} first change and fix: {fix}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"dossiers/{case_id}-file.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: file\n"
                        f"FACT_VALUE: {file_path}\n"
                        "SLOT: durable\n"
                        f"{case_id} patch file and minimal fix site: {file_path}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"dossiers/{case_id}-owner.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: owner\n"
                        f"FACT_VALUE: {owner}\n"
                        "SLOT: archive\n"
                        f"{case_id} patch owner: {owner}; {owner} owns review and rollout.\n"
                    ),
                ),
                ContextBlock(
                    name=f"dossiers/{case_id}-duplicate.md",
                    text=(
                        f"CASE_ID: {case_id}\n"
                        "FACT_KIND: root_cause\n"
                        f"FACT_VALUE: {root_cause}\n"
                        "SLOT: duplicate\n"
                        f"Duplicate: {case_id} root cause is {root_cause}; tempts single-score ranking.\n"
                    ),
                ),
            ]
        )
        rng.shuffle(docs)
        if case_type == "incident":
            query = f"For {case_id}, what is the root cause and what file should receive the minimal fix?"
            must_contain = [root_cause, file_path]
        elif case_type == "migration":
            query = f"For {case_id}, what blocks the migration and what change should be made first?"
            must_contain = [root_cause, fix]
        else:
            query = f"For {case_id}, what is the release blocker, who owns the patch, and which file should change?"
            must_contain = [root_cause, owner, file_path]
        examples.append(
            BenchmarkExample(
                name=case_id,
                query=query,
                context=docs,
                answer=" | ".join(must_contain),
                must_contain=must_contain,
                expected_provenance=[
                    f"dossiers/{case_id}-root-cause.md",
                    f"dossiers/{case_id}-fix.md",
                    f"dossiers/{case_id}-file.md",
                ],
                task_class=case_type,
                metadata={"service": service, "root_cause": root_cause, "fix": fix, "file": file_path, "owner": owner},
            )
        )
    return examples


def build_ruler_synthetic(n: int = 24, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    codes = ["orchid", "quartz", "ember", "lattice", "topaz", "willow", "raven", "comet"]
    mids = ["alpha-node", "bravo-node", "cedar-node", "delta-node", "ember-node", "frost-node"]
    regions = ["north", "south", "east", "west"]
    examples: list[BenchmarkExample] = []
    distractor_space = max(128, n + 64)

    for index in range(n):
        ruler_id = f"ruler-{index:03d}"
        mode = index % 3
        code = codes[(index * 2) % len(codes)]
        mid = mids[(index * 3) % len(mids)]
        docs: list[ContextBlock] = []
        for distractor in range(30):
            other_id = f"ruler-{(index + distractor + 17) % distractor_space:03d}"
            other_code = codes[(index + distractor) % len(codes)]
            docs.append(
                ContextBlock(
                    name=f"ruler/noise/{other_id}-{distractor:02d}.txt",
                    text=(
                        f"RULER_ID: {other_id}\n"
                        "FACT_KIND: distractor\n"
                        f"FACT_VALUE: {other_code}\n"
                        "SLOT: distractor\n"
                        f"{other_id} unrelated trace stores code {other_code}; belongs to another synthetic RULER task. "
                        f"Background filler {regions[distractor % len(regions)]} ledger text repeats neutral tokens.\n"
                    ),
                )
            )

        if mode == 0:
            docs.extend(
                [
                    ContextBlock(
                        name=f"ruler/{ruler_id}-first-hop.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: first_hop\n"
                            f"FACT_VALUE: {mid}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} first hop sends START_NODE root to MID_NODE {mid}.\n"
                        ),
                    ),
                    ContextBlock(
                        name=f"ruler/{ruler_id}-final-code.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: final_code\n"
                            f"FACT_VALUE: {code}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} MID_NODE {mid} resolves to FINAL_CODE {code}.\n"
                        ),
                    ),
                    ContextBlock(
                        name=f"ruler/{ruler_id}-duplicate-hop.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: first_hop\n"
                            f"FACT_VALUE: {mid}\n"
                            "SLOT: duplicate\n"
                            f"Duplicate: {ruler_id} root reaches {mid}, but it does not state the final code.\n"
                        ),
                    ),
                ]
            )
            query = f"For {ruler_id}, follow START_NODE root through MID_NODE and report the MID_NODE plus FINAL_CODE."
            must_contain = [mid, code]
            expected = [f"ruler/{ruler_id}-first-hop.txt", f"ruler/{ruler_id}-final-code.txt"]
            task_class = "ruler/variable_tracking"
        elif mode == 1:
            first = index + 11
            second = index + 7
            total = first + second
            docs.extend(
                [
                    ContextBlock(
                        name=f"ruler/{ruler_id}-north-shard.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: aggregate_shard\n"
                            f"FACT_VALUE: north={first}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} north shard contributes {first} units to the aggregate total.\n"
                        ),
                    ),
                    ContextBlock(
                        name=f"ruler/{ruler_id}-south-shard.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: aggregate_shard\n"
                            f"FACT_VALUE: south={second}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} south shard contributes {second} units to the aggregate total.\n"
                        ),
                    ),
                    ContextBlock(
                        name=f"ruler/{ruler_id}-total.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: aggregate_total\n"
                            f"FACT_VALUE: {total}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} aggregate total is {total} after combining north and south shards.\n"
                        ),
                    ),
                ]
            )
            query = f"For {ruler_id}, what aggregate total follows from the north and south shards?"
            must_contain = [str(total), str(first), str(second)]
            expected = [
                f"ruler/{ruler_id}-north-shard.txt",
                f"ruler/{ruler_id}-south-shard.txt",
                f"ruler/{ruler_id}-total.txt",
            ]
            task_class = "ruler/aggregation"
        else:
            key = f"key-{codes[index % len(codes)]}"
            value = f"value-{codes[(index + 3) % len(codes)]}"
            docs.extend(
                [
                    ContextBlock(
                        name=f"ruler/{ruler_id}-lookup-key.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: lookup_key\n"
                            f"FACT_VALUE: {key}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} requested lookup key is {key}.\n"
                        ),
                    ),
                    ContextBlock(
                        name=f"ruler/{ruler_id}-lookup-value.txt",
                        text=(
                            f"RULER_ID: {ruler_id}\n"
                            "FACT_KIND: lookup_value\n"
                            f"FACT_VALUE: {value}\n"
                            "SLOT: durable\n"
                            f"{ruler_id} {key} maps to retained value {value}.\n"
                        ),
                    ),
                ]
            )
            query = f"For {ruler_id}, which lookup key is requested and what retained value does it map to?"
            must_contain = [key, value]
            expected = [f"ruler/{ruler_id}-lookup-key.txt", f"ruler/{ruler_id}-lookup-value.txt"]
            task_class = "ruler/key_value"

        rng.shuffle(docs)
        examples.append(
            BenchmarkExample(
                name=ruler_id,
                query=query,
                context=docs,
                answer=" | ".join(must_contain),
                must_contain=must_contain,
                expected_provenance=expected,
                task_class=task_class,
                metadata={"benchmark_shape": "RULER synthetic", "mode": task_class},
            )
        )
    return examples


def build_babilong_synthetic(n: int = 24, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    people = ["Mara", "Jon", "Iris", "Tao", "Nia", "Omar"]
    objects = ["blue key", "silver map", "green badge", "red ledger", "amber token", "white pass"]
    rooms = ["archive", "observatory", "gallery", "workshop", "vault", "library"]
    codes = ["opal", "cedar", "onyx", "saffron", "pearl", "basalt"]
    examples: list[BenchmarkExample] = []
    distractor_space = max(128, n + 64)

    for index in range(n):
        story_id = f"babi-{index:03d}"
        person = people[index % len(people)]
        carried = objects[(index * 2) % len(objects)]
        room = rooms[(index * 3) % len(rooms)]
        code = codes[(index * 5) % len(codes)]
        docs: list[ContextBlock] = []
        for distractor in range(32):
            other_id = f"babi-{(index + distractor + 23) % distractor_space:03d}"
            other_person = people[(index + distractor) % len(people)]
            other_object = objects[(index + distractor + 1) % len(objects)]
            docs.append(
                ContextBlock(
                    name=f"babilong/noise/{other_id}-{distractor:02d}.txt",
                    text=(
                        f"BABILONG_ID: {other_id}\n"
                        "FACT_KIND: distractor_story\n"
                        f"FACT_VALUE: {other_person} carried {other_object}\n"
                        "SLOT: distractor\n"
                        f"{other_person} carried {other_object} in {other_id}; belongs to another story thread. "
                        "The corridor log adds neutral sentences to lengthen the episode.\n"
                    ),
                )
            )

        docs.extend(
            [
                ContextBlock(
                    name=f"babilong/{story_id}-person-object.txt",
                    text=(
                        f"BABILONG_ID: {story_id}\n"
                        "FACT_KIND: person_object\n"
                        f"FACT_VALUE: {person} carried {carried}\n"
                        "SLOT: durable\n"
                        f"In {story_id}, {person} carried the {carried} after leaving the hallway.\n"
                    ),
                ),
                ContextBlock(
                    name=f"babilong/{story_id}-object-room.txt",
                    text=(
                        f"BABILONG_ID: {story_id}\n"
                        "FACT_KIND: object_room\n"
                        f"FACT_VALUE: {carried} opens {room}\n"
                        "SLOT: durable\n"
                        f"In {story_id}, the {carried} opens the {room}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"babilong/{story_id}-room-code.txt",
                    text=(
                        f"BABILONG_ID: {story_id}\n"
                        "FACT_KIND: room_code\n"
                        f"FACT_VALUE: {room} contains {code}\n"
                        "SLOT: durable\n"
                        f"In {story_id}, the {room} contains code {code}.\n"
                    ),
                ),
                ContextBlock(
                    name=f"babilong/{story_id}-duplicate-object.txt",
                    text=(
                        f"BABILONG_ID: {story_id}\n"
                        "FACT_KIND: person_object\n"
                        f"FACT_VALUE: {person} carried {carried}\n"
                        "SLOT: duplicate\n"
                        f"Duplicate: {person} carried the {carried}, but this note omits the destination and code.\n"
                    ),
                ),
            ]
        )
        rng.shuffle(docs)
        examples.append(
            BenchmarkExample(
                name=story_id,
                query=f"In {story_id}, what object did {person} carry, which room did it open, and what code was there?",
                context=docs,
                answer=f"{carried} | {room} | {code}",
                must_contain=[carried, room, code],
                expected_provenance=[
                    f"babilong/{story_id}-person-object.txt",
                    f"babilong/{story_id}-object-room.txt",
                    f"babilong/{story_id}-room-code.txt",
                ],
                task_class="babilong/multi_hop_story",
                metadata={"benchmark_shape": "BABILong synthetic"},
            )
        )
    return examples


def load_curated_dataset(
    repo_root: str | Path,
    dataset_path: str | Path,
    *,
    distractors: int = 4,
    seed: int = 0,
) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    repo_root = Path(repo_root)
    dataset_path = Path(dataset_path)
    rows = json.loads(dataset_path.read_text())
    pool = sorted(path for path in repo_root.rglob("*") if path.is_file() and ".git" not in path.parts)
    examples: list[BenchmarkExample] = []
    for row in rows:
        provenance_paths = [repo_root / path for path in row["provenance"]]
        context: list[ContextBlock] = []
        for path in provenance_paths:
            context.extend(extract_anchor_blocks(path, row["must_contain"], window=8))
        distractor_pool = [path for path in pool if path not in provenance_paths and path.suffix in {".md", ".toml", ".py"}]
        rng.shuffle(distractor_pool)
        for path in distractor_pool[:distractors]:
            context.extend(load_text_blocks(path, chunk_size_lines=24)[:1])
        examples.append(
            BenchmarkExample(
                name=row["name"],
                query=row["query"],
                context=context,
                answer=row["answer"],
                must_contain=list(row["must_contain"]),
                expected_provenance=list(row.get("provenance", [])),
                task_class=str(row.get("task_class", "repo-qa")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return examples


def load_verifiers_30(repo_root: str | Path, dataset_path: str | Path | None = None, distractors: int = 4, seed: int = 0) -> list[BenchmarkExample]:
    return load_curated_dataset(
        repo_root=repo_root,
        dataset_path=dataset_path or ROOT / "examples" / "verifiers_30.json",
        distractors=distractors,
        seed=seed,
    )


def load_verifiers_smoke(repo_root: str | Path, dataset_path: str | Path | None = None, distractors: int = 2, seed: int = 0) -> list[BenchmarkExample]:
    return load_curated_dataset(
        repo_root=repo_root,
        dataset_path=dataset_path or ROOT / "tests" / "fixtures" / "verifiers_smoke.json",
        distractors=distractors,
        seed=seed,
    )


EXTERNAL_JSONL_MAPPED_KEYS = {
    "name",
    "query",
    "context",
    "input",
    "answer",
    "expected",
    "output",
    "outputs",
    "must_contain",
    "expected_provenance",
    "provenance",
    "task_class",
    "metadata",
}


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _external_answer_payload(row: dict[str, Any], line_number: int) -> Any:
    for key in ["answer", "expected", "output"]:
        if key in row:
            return row[key]
    outputs = row.get("outputs")
    if isinstance(outputs, list):
        if outputs:
            return outputs[0]
        raise ValueError(f"external_jsonl line {line_number} has empty outputs")
    if outputs is not None:
        return outputs
    raise ValueError(f"external_jsonl line {line_number} is missing answer, expected, output, or outputs")


def _external_context_blocks(example_name: str, payload: Any, line_number: int) -> list[ContextBlock]:
    if isinstance(payload, str):
        return [
            ContextBlock(
                name=f"{example_name}/context.txt",
                text=payload,
                metadata={"source": "external_jsonl", "line": line_number},
            )
        ]
    if isinstance(payload, list):
        blocks: list[ContextBlock] = []
        for index, item in enumerate(payload, start=1):
            fallback_name = f"{example_name}/context-{index}.txt"
            if isinstance(item, str):
                blocks.append(
                    ContextBlock(
                        name=fallback_name,
                        text=item,
                        metadata={"source": "external_jsonl", "line": line_number},
                    )
                )
            elif isinstance(item, dict):
                metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), dict) else {}
                metadata.update({"source": "external_jsonl", "line": line_number})
                blocks.append(
                    ContextBlock(
                        name=str(item.get("name", fallback_name)),
                        text=str(item.get("text", "")),
                        metadata=metadata,
                    )
                )
            else:
                raise ValueError(f"external_jsonl line {line_number} context item {index} must be a string or object")
        if blocks:
            return blocks
    raise ValueError(f"external_jsonl line {line_number} context must be a string or non-empty list")


def _external_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {}
    extras = {key: value for key, value in row.items() if key not in EXTERNAL_JSONL_MAPPED_KEYS}
    if extras:
        existing_source_row = metadata.get("source_row")
        if isinstance(existing_source_row, dict):
            metadata["source_row"] = {**existing_source_row, **extras}
        else:
            metadata["source_row"] = extras
    return metadata


def load_external_jsonl(path: str | Path) -> list[BenchmarkExample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ValueError(f"external_jsonl dataset path does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"external_jsonl dataset path is not a file: {dataset_path}")
    examples: list[BenchmarkExample] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"external_jsonl line {line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"external_jsonl line {line_number} must be a JSON object")
            query = row.get("query")
            if query is None:
                raise ValueError(f"external_jsonl line {line_number} is missing query")
            answer_payload = _external_answer_payload(row, line_number)
            answer_parts = [part.strip() for part in _as_string_list(answer_payload) if part.strip()]
            if not answer_parts:
                raise ValueError(f"external_jsonl line {line_number} has empty answer payload")
            answer = " | ".join(answer_parts)
            context_payload = row.get("context", row.get("input"))
            if context_payload is None:
                raise ValueError(f"external_jsonl line {line_number} is missing context or input")
            name = str(row.get("name", f"external-{line_number}"))
            must_contain = _as_string_list(row.get("must_contain")) or answer_parts
            expected_provenance = _as_string_list(row.get("expected_provenance", row.get("provenance")))
            examples.append(
                BenchmarkExample(
                    name=name,
                    query=str(query),
                    context=_external_context_blocks(name, context_payload, line_number),
                    answer=answer,
                    must_contain=must_contain,
                    expected_provenance=expected_provenance,
                    task_class=str(row.get("task_class", "external")),
                    metadata=_external_metadata(row),
                )
            )
    return examples


def resolve_provider_arg(provider: str, use_openai_backend: bool | None) -> str:
    if use_openai_backend is None:
        return provider
    return "openai_compatible" if use_openai_backend else provider


def validate_benchmark_cost_support(provider: str, model: str, base_url: str | None) -> None:
    if supports_cost_estimate(provider, model, base_url):
        return
    raise ValueError(
        f"benchmark cost estimates are not supported for provider={provider} model={model}; "
        "use a priced OpenAI-compatible model from the built-in table"
    )


def run_policy_case(
    example: BenchmarkExample,
    policy: str,
    *,
    budget: int,
    max_depth: int,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    cache_dir: str | None,
    max_output_tokens: int,
    learned_retention_model: str | None,
    seed: int,
) -> RLMResult:
    config = RLMConfig(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        cache_dir=cache_dir,
        max_output_tokens=max_output_tokens,
        max_depth=max_depth,
        max_steps=256,
        memory_budget_tokens=budget,
        retention_policy="keep_recent" if policy == "direct_full_context" else policy,
        retention_model_path=learned_retention_model if policy == "learned_retention" else None,
        seed=seed,
    )
    engine = RLM(config=config)
    if policy == "direct_full_context":
        return engine.direct_completion(example.query, example.context)
    return engine.completion(example.query, example.context)


def run_dataset(
    examples: Sequence[BenchmarkExample],
    policy: str,
    *,
    budget: int = 120,
    max_depth: int = 2,
    provider: str = "heuristic",
    model: str = "demo/heuristic",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    max_output_tokens: int = 1024,
    learned_retention_model: str | Path | None = None,
    max_estimated_cost: float | None = None,
    initial_cost_estimate: float = 0.0,
    output_dir: str | Path | None = None,
    use_openai_backend: bool | None = None,
    seed: int = 0,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    provider = resolve_provider_arg(provider, use_openai_backend)
    validate_benchmark_cost_support(provider, model, base_url)
    results: list[dict[str, Any]] = []
    stop_reason: str | None = None
    cumulative_cost = round(initial_cost_estimate, 6)
    trace_root: Path | None = None
    if output_dir is not None:
        trace_root = Path(output_dir) / "trace_examples" / policy
        trace_root.mkdir(parents=True, exist_ok=True)
    if max_estimated_cost is not None and cumulative_cost >= max_estimated_cost and examples:
        stop_reason = "cost_cap"
    for example in examples:
        if max_estimated_cost is not None and cumulative_cost >= max_estimated_cost:
            stop_reason = "cost_cap"
            break
        started = time.perf_counter()
        result = run_policy_case(
            example,
            policy,
            budget=budget,
            max_depth=max_depth,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            max_output_tokens=max_output_tokens,
            learned_retention_model=str(learned_retention_model) if learned_retention_model else None,
            seed=seed,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        cumulative_cost = round(cumulative_cost + result.cost_estimate, 6)
        answer_accuracy = score_answer(result.answer, example.must_contain)
        provenance_score, provenance_hits = score_provenance(result, example.expected_provenance)
        retained_tokens = sum(item.tokens for item in result.kept_items)
        compactness = compactness_score(retained_tokens, budget)
        reward_score = retention_reward_score(
            answer_accuracy=answer_accuracy,
            provenance_score=provenance_score,
            compactness=compactness,
            latency_ms=elapsed_ms,
            cost_estimate=result.cost_estimate,
        )
        row = {
            "dataset": dataset_name,
            "seed": seed,
            "name": example.name,
            "task_class": example.task_class,
            "policy": policy,
            "query": example.query,
            "answer": result.answer,
            "expected": example.answer,
            "must_contain": list(example.must_contain),
            "expected_provenance": list(example.expected_provenance),
            "answer_accuracy": answer_accuracy,
            "provenance_score": provenance_score,
            "provenance_hits": provenance_hits,
            "compactness": compactness,
            "reward_score": reward_score,
            "retained_items": len(result.kept_items),
            "retained_tokens": retained_tokens,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "calls": result.usage.calls,
            },
            "cost_estimate": result.cost_estimate,
            "cumulative_cost_estimate": cumulative_cost,
            "latency_ms": elapsed_ms,
            "retention_stats": result.retention_stats,
            "drop_reasons": result.drop_reasons,
            "per_step_budget": result.per_step_budget,
            "retained_summaries": [item.summary for item in result.kept_items],
        }
        if trace_root is not None:
            write_trace(result, trace_root / f"{example.name}.jsonl")
            result.trace.write_tree(trace_root / f"{example.name}.tree.txt")
        results.append(row)

    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in results), 3) if results else 0.0

    summary = {
        "dataset": dataset_name,
        "policy": policy,
        "examples": len(results),
        "requested_examples": len(examples),
        "accuracy": mean("answer_accuracy"),
        "answer_accuracy": mean("answer_accuracy"),
        "provenance_score": mean("provenance_score"),
        "compactness": mean("compactness"),
        "reward_score": mean("reward_score"),
        "avg_retained_tokens": mean("retained_tokens"),
        "avg_latency_ms": mean("latency_ms"),
        "avg_cost_estimate": round(statistics.fmean(float(row["cost_estimate"]) for row in results), 6) if results else 0.0,
        "total_cost_estimate": round(sum(float(row["cost_estimate"]) for row in results), 6),
        "initial_cost_estimate": round(initial_cost_estimate, 6),
        "final_cumulative_cost_estimate": cumulative_cost,
        "max_estimated_cost": max_estimated_cost,
        "completed": stop_reason is None,
        "stop_reason": stop_reason,
        "last_completed_case": results[-1]["name"] if results else None,
        "results": results,
    }
    return summary


def policy_sweep(
    examples: Sequence[BenchmarkExample],
    policies: Sequence[str],
    *,
    budget: int,
    max_depth: int,
    output_dir: str | Path | None = None,
    provider: str = "heuristic",
    model: str = "demo/heuristic",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    max_output_tokens: int = 1024,
    learned_retention_model: str | Path | None = None,
    max_estimated_cost: float | None = None,
    use_openai_backend: bool | None = None,
    seed: int = 0,
    dataset_name: str = "dataset",
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    for policy in policies:
        summary = run_dataset(
            examples,
            policy,
            budget=budget,
            max_depth=max_depth,
            output_dir=output_dir,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            cache_dir=cache_dir,
            max_output_tokens=max_output_tokens,
            learned_retention_model=learned_retention_model,
            max_estimated_cost=max_estimated_cost,
            initial_cost_estimate=cumulative_cost,
            use_openai_backend=use_openai_backend,
            seed=seed,
            dataset_name=dataset_name,
        )
        summaries.append(summary)
        cumulative_cost = float(summary["final_cumulative_cost_estimate"])
    return summaries


def generate_curves(
    dataset_name: str,
    example_factory: Callable[[int], Sequence[BenchmarkExample]],
    *,
    policies: Sequence[str],
    budgets: Sequence[int],
    depths: Sequence[int],
    seeds: Sequence[int],
    provider: str = "heuristic",
    model: str = "demo/heuristic",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    max_output_tokens: int = 1024,
    learned_retention_model: str | Path | None = None,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for seed in seeds:
        examples = list(example_factory(seed))
        for depth in depths:
            for budget in budgets:
                summaries = policy_sweep(
                    examples,
                    policies,
                    budget=budget,
                    max_depth=depth,
                    output_dir=None,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    cache_dir=cache_dir,
                    max_output_tokens=max_output_tokens,
                    learned_retention_model=learned_retention_model,
                    seed=seed,
                    dataset_name=dataset_name,
                )
                for summary in summaries:
                    points.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "depth": depth,
                            "budget": budget,
                            "policy": summary["policy"],
                            "answer_accuracy": summary["answer_accuracy"],
                            "provenance_score": summary["provenance_score"],
                            "compactness": summary["compactness"],
                            "reward_score": summary["reward_score"],
                            "avg_retained_tokens": summary["avg_retained_tokens"],
                            "avg_latency_ms": summary["avg_latency_ms"],
                            "avg_cost_estimate": summary["avg_cost_estimate"],
                        }
                    )
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault((point["policy"], point["budget"], point["depth"]), []).append(point)
    aggregates = []
    for (policy, budget, depth), rows in grouped.items():
        aggregates.append(
            {
                "policy": policy,
                "budget": budget,
                "depth": depth,
                "answer_accuracy": round(statistics.fmean(row["answer_accuracy"] for row in rows), 3),
                "provenance_score": round(statistics.fmean(row["provenance_score"] for row in rows), 3),
                "compactness": round(statistics.fmean(row["compactness"] for row in rows), 3),
                "reward_score": round(statistics.fmean(row["reward_score"] for row in rows), 3),
                "avg_retained_tokens": round(statistics.fmean(row["avg_retained_tokens"] for row in rows), 3),
                "avg_latency_ms": round(statistics.fmean(row["avg_latency_ms"] for row in rows), 3),
                "avg_cost_estimate": round(statistics.fmean(row["avg_cost_estimate"] for row in rows), 6),
                "seeds": len(rows),
            }
        )
    return {
        "dataset": dataset_name,
        "budgets": list(budgets),
        "depths": list(depths),
        "seeds": list(seeds),
        "points": points,
        "aggregates": sorted(aggregates, key=lambda row: (row["depth"], row["budget"], row["policy"])),
    }


def curves_from_summaries(
    dataset_name: str,
    summaries: Sequence[dict[str, Any]],
    *,
    budget: int,
    depth: int,
    seed: int = 0,
) -> dict[str, Any]:
    points = [
        {
            "dataset": dataset_name,
            "seed": seed,
            "depth": depth,
            "budget": budget,
            "policy": summary["policy"],
            "answer_accuracy": summary["answer_accuracy"],
            "provenance_score": summary["provenance_score"],
            "compactness": summary["compactness"],
            "reward_score": summary.get("reward_score", 0.0),
            "avg_retained_tokens": summary["avg_retained_tokens"],
            "avg_latency_ms": summary["avg_latency_ms"],
            "avg_cost_estimate": summary["avg_cost_estimate"],
            "total_cost_estimate": summary["total_cost_estimate"],
            "completed": summary["completed"],
            "stop_reason": summary["stop_reason"],
        }
        for summary in summaries
    ]
    aggregates = [
        {
            "policy": point["policy"],
            "budget": budget,
            "depth": depth,
            "answer_accuracy": point["answer_accuracy"],
            "provenance_score": point["provenance_score"],
            "compactness": point["compactness"],
            "reward_score": point["reward_score"],
            "avg_retained_tokens": point["avg_retained_tokens"],
            "avg_latency_ms": point["avg_latency_ms"],
            "avg_cost_estimate": point["avg_cost_estimate"],
            "total_cost_estimate": point["total_cost_estimate"],
            "completed": point["completed"],
            "stop_reason": point["stop_reason"],
            "seeds": 1,
        }
        for point in points
    ]
    return {
        "dataset": dataset_name,
        "budgets": [budget],
        "depths": [depth],
        "seeds": [seed],
        "points": points,
        "aggregates": sorted(aggregates, key=lambda row: (row["depth"], row["budget"], row["policy"])),
    }


def write_report_bundle(
    output_dir: str | Path,
    *,
    dataset_name: str,
    summaries: Sequence[dict[str, Any]],
    curves: dict[str, Any],
    command: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    insights = build_experiment_insights(dataset_name, summaries)
    summary_payload = {
        "dataset": dataset_name,
        "generated_by": "bench.py",
        "command": command,
        "policies": [summary["policy"] for summary in summaries],
        "insights": insights,
        "summaries": list(summaries),
    }
    (output_path / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    with (output_path / "per_case.jsonl").open("w") as handle:
        for summary in summaries:
            for row in summary["results"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_path / "curves.json").write_text(json.dumps(curves, indent=2))
    (output_path / "experiment_report.md").write_text(
        format_experiment_report(
            dataset_name=dataset_name,
            summaries=summaries,
            insights=insights,
            command=command,
        )
    )


def build_dataset(
    dataset_name: str,
    *,
    limit: int,
    seed: int,
    repo_root: str,
    dataset_path: str | Path | None = None,
    start_index: int = 0,
) -> list[BenchmarkExample]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    requested = limit + start_index
    def window(examples: Sequence[BenchmarkExample]) -> list[BenchmarkExample]:
        return list(examples[start_index : start_index + limit])

    if dataset_name == "pairbench":
        return window(build_pairbench(n=requested, seed=seed))
    if dataset_name == "needlepairs":
        return window(build_needlepairs(n=requested, seed=seed))
    if dataset_name == "dossierbench":
        return window(build_dossierbench(n=requested, seed=seed))
    if dataset_name == "ruler_synthetic":
        return window(build_ruler_synthetic(n=requested, seed=seed))
    if dataset_name == "babilong_synthetic":
        return window(build_babilong_synthetic(n=requested, seed=seed))
    if dataset_name == "verifiers_30":
        return window(load_verifiers_30(repo_root, seed=seed))
    if dataset_name == "verifiers_smoke":
        return window(load_verifiers_smoke(repo_root, seed=seed))
    if dataset_name == "external_jsonl":
        if dataset_path is None:
            raise ValueError("--dataset-path is required when --dataset external_jsonl")
        return window(load_external_jsonl(dataset_path))
    raise ValueError(f"unknown dataset: {dataset_name}")


def format_table(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        "| policy | examples | answer | prov | compact | reward | avg toks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['examples']} | {row['answer_accuracy']:.3f} | {row['provenance_score']:.3f} | "
            f"{row['compactness']:.3f} | {row.get('reward_score', 0.0):.3f} | {row['avg_retained_tokens']:.1f} |"
        )
    return "\n".join(lines)


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value is not None else 0.0


def _mean_metric(rows: Sequence[dict[str, Any]], key: str) -> float:
    return round(statistics.fmean(_metric(row, key) for row in rows), 3) if rows else 0.0


def _policy_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        -_metric(row, "reward_score"),
        -_metric(row, "answer_accuracy"),
        -_metric(row, "provenance_score"),
        -_metric(row, "compactness"),
        _metric(row, "avg_latency_ms"),
        str(row.get("policy", "")),
    )


def _failure_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if _metric(row, "answer_accuracy") < 1.0:
        tags.append("answer_miss")
    if row.get("expected_provenance") and _metric(row, "provenance_score") < 1.0:
        tags.append("provenance_miss")
    if _metric(row, "compactness") <= 0.05:
        tags.append("budget_saturated")
    if row.get("drop_reasons"):
        tags.append("retention_dropped_items")
    return tags


def build_experiment_insights(
    dataset_name: str,
    summaries: Sequence[dict[str, Any]],
    *,
    baseline_policy: str = "direct_full_context",
) -> dict[str, Any]:
    ranking = [
        {
            "rank": index,
            "policy": summary["policy"],
            "examples": summary["examples"],
            "answer_accuracy": summary["answer_accuracy"],
            "provenance_score": summary["provenance_score"],
            "compactness": summary["compactness"],
            "reward_score": summary.get("reward_score", 0.0),
            "avg_retained_tokens": summary["avg_retained_tokens"],
            "avg_latency_ms": summary["avg_latency_ms"],
            "total_cost_estimate": summary["total_cost_estimate"],
            "completed": summary["completed"],
            "stop_reason": summary["stop_reason"],
        }
        for index, summary in enumerate(sorted(summaries, key=_policy_rank_key), start=1)
    ]
    baseline = next((summary for summary in summaries if summary["policy"] == baseline_policy), None)
    deltas = []
    if baseline is not None:
        for summary in summaries:
            deltas.append(
                {
                    "policy": summary["policy"],
                    "answer_accuracy_delta": round(summary["answer_accuracy"] - baseline["answer_accuracy"], 3),
                    "provenance_score_delta": round(summary["provenance_score"] - baseline["provenance_score"], 3),
                    "compactness_delta": round(summary["compactness"] - baseline["compactness"], 3),
                    "reward_score_delta": round(summary.get("reward_score", 0.0) - baseline.get("reward_score", 0.0), 3),
                    "avg_retained_tokens_delta": round(summary["avg_retained_tokens"] - baseline["avg_retained_tokens"], 3),
                }
            )

    all_rows = [row for summary in summaries for row in summary["results"]]
    task_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    failure_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        policy = str(row.get("policy", ""))
        task_class = str(row.get("task_class", "general"))
        task_groups.setdefault((policy, task_class), []).append(row)
        tags = _failure_tags(row)
        if tags:
            failure_groups.setdefault((policy, task_class, ",".join(tags)), []).append(row)

    task_breakdown = [
        {
            "policy": policy,
            "task_class": task_class,
            "examples": len(rows),
            "answer_accuracy": _mean_metric(rows, "answer_accuracy"),
            "provenance_score": _mean_metric(rows, "provenance_score"),
            "compactness": _mean_metric(rows, "compactness"),
            "reward_score": _mean_metric(rows, "reward_score"),
            "answer_misses": sum(1 for row in rows if _metric(row, "answer_accuracy") < 1.0),
            "provenance_misses": sum(1 for row in rows if row.get("expected_provenance") and _metric(row, "provenance_score") < 1.0),
        }
        for (policy, task_class), rows in sorted(task_groups.items())
    ]
    failure_clusters = [
        {
            "policy": policy,
            "task_class": task_class,
            "tags": tags.split(","),
            "cases": len(rows),
            "examples": [str(row.get("name", "")) for row in rows[:3]],
        }
        for (policy, task_class, tags), rows in sorted(failure_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    completed = [summary["policy"] for summary in summaries if summary["completed"]]
    partial = [
        {"policy": summary["policy"], "stop_reason": summary["stop_reason"]}
        for summary in summaries
        if not summary["completed"]
    ]
    return {
        "dataset": dataset_name,
        "baseline_policy": baseline_policy if baseline is not None else None,
        "policy_ranking": ranking,
        "policy_deltas": deltas,
        "task_breakdown": task_breakdown,
        "failure_clusters": failure_clusters,
        "coverage": {
            "policies": len(summaries),
            "completed_policies": completed,
            "partial_policies": partial,
            "case_rows": len(all_rows),
        },
    }


def _status_label(row: dict[str, Any]) -> str:
    if row.get("completed", True):
        return "complete"
    return f"partial:{row.get('stop_reason') or 'unknown'}"


def format_experiment_report(
    *,
    dataset_name: str,
    summaries: Sequence[dict[str, Any]],
    insights: dict[str, Any],
    command: str,
) -> str:
    lines = [
        "# Experiment Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Command: `{command}`",
        f"- Case rows: {insights['coverage']['case_rows']}",
        "",
        "## Policy Ranking",
        "",
        "| rank | policy | answer | prov | compact | reward | avg toks | cost | status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in insights["policy_ranking"]:
        lines.append(
            f"| {row['rank']} | `{row['policy']}` | {row['answer_accuracy']:.3f} | {row['provenance_score']:.3f} | "
            f"{row['compactness']:.3f} | {row.get('reward_score', 0.0):.3f} | {row['avg_retained_tokens']:.1f} | "
            f"{row['total_cost_estimate']:.6f} | {_status_label(row)} |"
        )

    if insights["baseline_policy"] and insights["policy_deltas"]:
        lines.extend(
            [
                "",
                f"## Deltas Vs `{insights['baseline_policy']}`",
                "",
                "| policy | answer delta | prov delta | compact delta | reward delta | avg toks delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in insights["policy_deltas"]:
            lines.append(
                f"| `{row['policy']}` | {row['answer_accuracy_delta']:+.3f} | {row['provenance_score_delta']:+.3f} | "
                f"{row['compactness_delta']:+.3f} | {row.get('reward_score_delta', 0.0):+.3f} | "
                f"{row['avg_retained_tokens_delta']:+.1f} |"
            )

    lines.extend(["", "## Failure Clusters", ""])
    if insights["failure_clusters"]:
        lines.extend(["| policy | task | tags | cases | examples |", "| --- | --- | --- | ---: | --- |"])
        for row in insights["failure_clusters"][:12]:
            examples = ", ".join(f"`{name}`" for name in row["examples"] if name)
            tags = ", ".join(f"`{tag}`" for tag in row["tags"])
            lines.append(f"| `{row['policy']}` | `{row['task_class']}` | {tags} | {row['cases']} | {examples} |")
    else:
        lines.append("No failures were tagged in this run.")

    lines.extend(
        [
            "",
            "## Task Breakdown",
            "",
            "| policy | task | examples | answer | prov | compact | reward | answer misses | prov misses |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in insights["task_breakdown"]:
        lines.append(
            f"| `{row['policy']}` | `{row['task_class']}` | {row['examples']} | {row['answer_accuracy']:.3f} | "
            f"{row['provenance_score']:.3f} | {row['compactness']:.3f} | {row.get('reward_score', 0.0):.3f} | "
            f"{row['answer_misses']} | {row['provenance_misses']} |"
        )

    lines.extend(
        [
            "",
            "## Bundle",
            "",
            "- `summary.json`: machine-readable summary plus this run's insights",
            "- `per_case.jsonl`: one scored row per policy and case",
            "- `curves.json`: sweep points and aggregates",
            "- `trace_examples/`: retained recursive traces when `--output-dir` is set",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_provider_choice(provider: str, use_openai_alias: bool) -> str:
    normalized = provider.strip().lower()
    if use_openai_alias and normalized == "heuristic":
        return "openai_compatible"
    return normalized.replace("-", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nanoRLM synthetic or repo-backed benchmarks.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="pairbench",
    )
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--repo-root", type=str, default="/tmp/nanorlm-verifiers")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--curve-budgets", type=str, default="")
    parser.add_argument("--curve-depths", type=str, default="")
    parser.add_argument("--curve-seeds", type=str, default="")
    parser.add_argument("--model", type=str, default="demo/heuristic")
    parser.add_argument("--provider", choices=CLI_PROVIDER_CHOICES, default="heuristic")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--learned-retention-model", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-estimated-cost",
        type=float,
        default=None,
        help="Global run-level cap for supported remote model cost estimates.",
    )
    parser.add_argument("--openai", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    provider = resolve_provider_choice(args.provider, args.openai)
    policies = parse_csv_strings(args.policies)
    cache_dir = None if args.no_cache else args.cache_dir or None
    try:
        validate_benchmark_cost_support(provider, args.model, args.base_url or None)
        examples = build_dataset(
            args.dataset,
            limit=args.limit,
            seed=args.seed,
            repo_root=args.repo_root,
            dataset_path=args.dataset_path or None,
            start_index=args.start_index,
        )
    except ValueError as exc:
        parser.error(str(exc))
    summaries = policy_sweep(
        examples,
        policies,
        budget=args.budget,
        max_depth=args.depth,
        output_dir=args.output_dir or None,
        provider=provider,
        model=args.model,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        cache_dir=cache_dir,
        max_output_tokens=args.max_output_tokens,
        learned_retention_model=args.learned_retention_model or None,
        max_estimated_cost=args.max_estimated_cost,
        seed=args.seed,
        dataset_name=args.dataset,
    )
    print(format_table(summaries))

    if provider == "heuristic":
        curve_budgets = parse_csv_ints(args.curve_budgets) if args.curve_budgets else [args.budget]
        curve_depths = parse_csv_ints(args.curve_depths) if args.curve_depths else [args.depth]
        curve_seeds = parse_csv_ints(args.curve_seeds) if args.curve_seeds else [0]
        curves = generate_curves(
            args.dataset,
            lambda seed: build_dataset(
                args.dataset,
                limit=args.limit,
                seed=seed,
                repo_root=args.repo_root,
                dataset_path=args.dataset_path or None,
                start_index=args.start_index,
            ),
            policies=policies,
            budgets=curve_budgets,
            depths=curve_depths,
            seeds=curve_seeds,
            provider=provider,
            model=args.model,
            base_url=args.base_url or None,
            api_key=args.api_key or None,
            cache_dir=cache_dir,
            max_output_tokens=args.max_output_tokens,
            learned_retention_model=args.learned_retention_model or None,
        )
    else:
        curves = curves_from_summaries(args.dataset, summaries, budget=args.budget, depth=args.depth)
    if args.output_dir:
        write_report_bundle(
            args.output_dir,
            dataset_name=args.dataset,
            summaries=summaries,
            curves=curves,
            command=" ".join(["python", "bench.py", *filter(None, [
                f"--dataset {args.dataset}",
                f"--limit {args.limit}",
                f"--start-index {args.start_index}",
                f"--seed {args.seed}",
                f"--budget {args.budget}",
                f"--depth {args.depth}",
                f"--provider {args.provider}",
                f"--dataset-path {args.dataset_path}" if args.dataset_path else "",
                f"--base-url {args.base_url}" if args.base_url else "",
                f"--cache-dir {args.cache_dir}" if cache_dir else "",
                f"--learned-retention-model {args.learned_retention_model}" if args.learned_retention_model else "",
                f"--max-output-tokens {args.max_output_tokens}",
                f"--max-estimated-cost {args.max_estimated_cost}" if args.max_estimated_cost is not None else "",
            ])]),
        )


if __name__ == "__main__":
    main()
