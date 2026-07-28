from __future__ import annotations

import html
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DEFINITION_RE = re.compile(
    r"^\s{0,3}\[[^\]\n]+\]:\s*(?P<destination><[^>\n]+>|\S+)"
)
FENCE_RE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)(?P<heading>.*)$")
SETEXT_HEADING_RE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
HTML_ANCHOR_RE = re.compile(
    r"""<(?:a|[A-Za-z][A-Za-z0-9-]*)\b[^>]*\b(?:id|name)\s*=\s*
        (?:"(?P<double>[^"]+)"|'(?P<single>[^']+)')""",
    re.IGNORECASE | re.VERBOSE,
)
MARKDOWN_ESCAPE_RE = re.compile(r"""\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])""")


@dataclass(frozen=True, slots=True)
class MarkdownLink:
    source: Path
    line: int
    destination: str


@dataclass(frozen=True, slots=True)
class LinkIssue:
    link: MarkdownLink
    reason: str
    resolved: str


@dataclass(frozen=True, slots=True)
class LinkCheckResult:
    documents: int
    local_checked: int
    external_skipped: int
    issues: tuple[LinkIssue, ...]


def tracked_repository_files(root: Path = ROOT) -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"could not run git to enumerate tracked repository files: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip() or "git ls-files failed"
        raise RuntimeError(f"could not enumerate tracked repository files: {detail}")

    return sorted(
        [
            root / entry.decode(errors="surrogateescape")
            for entry in completed.stdout.split(b"\0")
            if entry
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )


def tracked_markdown_files(
    root: Path = ROOT,
    tracked_files: Iterable[Path] | None = None,
) -> list[Path]:
    repository_files = (
        list(tracked_files)
        if tracked_files is not None
        else tracked_repository_files(root)
    )
    return [path for path in repository_files if path.suffix.lower() == ".md"]


def strip_inline_code(line: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(line):
        if line[index] != "`":
            output.append(line[index])
            index += 1
            continue

        marker_end = index
        while marker_end < len(line) and line[marker_end] == "`":
            marker_end += 1
        marker = line[index:marker_end]
        closing = line.find(marker, marker_end)
        if closing == -1:
            output.append(line[index])
            index += 1
            continue

        output.append(" " * (closing + len(marker) - index))
        index = closing + len(marker)
    return "".join(output)


def _parse_title_and_close(line: str, index: int) -> bool:
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line):
        return False
    if line[index] == ")":
        return True

    delimiter = line[index]
    if delimiter not in {'"', "'", "("}:
        return False
    closing = ")" if delimiter == "(" else delimiter
    index += 1
    depth = 1
    while index < len(line):
        character = line[index]
        if character == "\\" and index + 1 < len(line):
            index += 2
            continue
        if delimiter == "(" and character == "(":
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                index += 1
                break
        index += 1
    else:
        return False

    while index < len(line) and line[index].isspace():
        index += 1
    return index < len(line) and line[index] == ")"


def _inline_destination(line: str, index: int) -> str | None:
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line):
        return None

    if line[index] == "<":
        index += 1
        destination: list[str] = []
        while index < len(line):
            character = line[index]
            if character == "\\" and index + 1 < len(line):
                destination.extend(line[index : index + 2])
                index += 2
                continue
            if character == ">":
                index += 1
                return (
                    "".join(destination)
                    if _parse_title_and_close(line, index)
                    else None
                )
            destination.append(character)
            index += 1
        return None

    destination = []
    parenthesis_depth = 0
    while index < len(line):
        character = line[index]
        if character == "\\" and index + 1 < len(line):
            destination.extend(line[index : index + 2])
            index += 2
            continue
        if character == "(":
            parenthesis_depth += 1
            destination.append(character)
            index += 1
            continue
        if character == ")":
            if parenthesis_depth == 0:
                return "".join(destination)
            parenthesis_depth -= 1
            destination.append(character)
            index += 1
            continue
        if character.isspace() and parenthesis_depth == 0:
            return (
                "".join(destination)
                if _parse_title_and_close(line, index)
                else None
            )
        destination.append(character)
        index += 1
    return None


def _is_escaped(line: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and line[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def inline_destinations(line: str) -> Iterator[str]:
    for label_start, character in enumerate(line):
        if character != "[" or _is_escaped(line, label_start):
            continue

        bracket_depth = 1
        index = label_start + 1
        while index < len(line) and bracket_depth:
            character = line[index]
            if character == "\\" and index + 1 < len(line):
                index += 2
                continue
            if character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth -= 1
            index += 1

        if bracket_depth or index >= len(line) or line[index] != "(":
            continue

        destination = _inline_destination(line, index + 1)
        if destination is not None:
            yield destination


def links_in_document(source: Path) -> Iterator[MarkdownLink]:
    fence_character = ""
    fence_length = 0

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fence_match = FENCE_RE.match(raw_line)
        if fence_character:
            if fence_match:
                marker = fence_match.group("fence")
                if marker[0] == fence_character and len(marker) >= fence_length:
                    fence_character = ""
                    fence_length = 0
            continue
        if fence_match:
            marker = fence_match.group("fence")
            fence_character = marker[0]
            fence_length = len(marker)
            continue

        line = strip_inline_code(raw_line)
        for destination in inline_destinations(line):
            yield MarkdownLink(
                source=source,
                line=line_number,
                destination=destination,
            )
        definition = REFERENCE_DEFINITION_RE.match(line)
        if definition:
            yield MarkdownLink(
                source=source,
                line=line_number,
                destination=definition.group("destination"),
            )


def _heading_slug(heading: str) -> str:
    heading = html.unescape(heading)
    heading = re.sub(r"(`+)(.*?)\1", r"\2", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"[*~]", "", heading)
    heading = heading.strip().rstrip("#").strip().casefold()
    heading = "".join(
        character
        for character in heading
        if character.isalnum() or character in {" ", "-", "_"}
    )
    return re.sub(r"\s+", "-", heading)


def anchors_in_document(source: Path) -> set[str]:
    anchors: set[str] = set()
    generated_slugs: set[str] = set()
    fence_character = ""
    fence_length = 0
    previous_line: str | None = None

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        fence_match = FENCE_RE.match(raw_line)
        if fence_character:
            if fence_match:
                marker = fence_match.group("fence")
                if marker[0] == fence_character and len(marker) >= fence_length:
                    fence_character = ""
                    fence_length = 0
            previous_line = None
            continue
        if fence_match:
            marker = fence_match.group("fence")
            fence_character = marker[0]
            fence_length = len(marker)
            previous_line = None
            continue

        for match in HTML_ANCHOR_RE.finditer(raw_line):
            anchors.add(html.unescape(match.group("double") or match.group("single")))

        heading: str | None = None
        atx_match = ATX_HEADING_RE.match(raw_line)
        if atx_match:
            heading = atx_match.group("heading")
        elif previous_line is not None and SETEXT_HEADING_RE.match(raw_line):
            heading = previous_line.strip()

        if heading is not None:
            base_slug = _heading_slug(heading)
            if base_slug:
                slug = base_slug
                suffix = 1
                while slug in generated_slugs:
                    slug = f"{base_slug}-{suffix}"
                    suffix += 1
                generated_slugs.add(slug)
                anchors.add(slug)

        previous_line = raw_line

    return anchors


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def check_markdown_links(
    root: Path,
    documents: Iterable[Path],
    tracked_files: Iterable[Path] | None = None,
) -> LinkCheckResult:
    resolved_root = root.resolve()
    tracked_candidates = (
        tuple(path.resolve() for path in tracked_files)
        if tracked_files is not None
        else None
    )
    document_list = sorted(
        (path.resolve() for path in documents),
        key=lambda path: _display_path(path, resolved_root),
    )
    issues: list[LinkIssue] = []
    local_checked = 0
    external_skipped = 0
    anchor_cache: dict[Path, set[str]] = {}

    for source in document_list:
        for link in links_in_document(source):
            destination = html.unescape(link.destination.strip())
            if destination.startswith("<") and destination.endswith(">"):
                destination = destination[1:-1].strip()
            destination = MARKDOWN_ESCAPE_RE.sub(r"\1", destination)

            try:
                parsed = urlsplit(destination)
            except ValueError:
                local_checked += 1
                issues.append(
                    LinkIssue(link=link, reason="invalid destination", resolved=destination)
                )
                continue

            if parsed.scheme or parsed.netloc or destination.startswith("//"):
                external_skipped += 1
                continue

            local_checked += 1
            if parsed.path:
                decoded_path = unquote(parsed.path)
                local_path = Path(decoded_path)
                if local_path.is_absolute():
                    issues.append(
                        LinkIssue(
                            link=link,
                            reason="absolute local path",
                            resolved=decoded_path,
                        )
                    )
                    continue

                candidate = (source.parent / local_path).resolve()
                try:
                    candidate.relative_to(resolved_root)
                except ValueError:
                    issues.append(
                        LinkIssue(
                            link=link,
                            reason="outside repository",
                            resolved=str(candidate),
                        )
                    )
                    continue
            else:
                candidate = source

            if not candidate.exists():
                issues.append(
                    LinkIssue(
                        link=link,
                        reason="missing",
                        resolved=_display_path(candidate, resolved_root),
                    )
                )
                continue

            if tracked_candidates is not None and not any(
                tracked == candidate
                or (candidate.is_dir() and tracked.is_relative_to(candidate))
                for tracked in tracked_candidates
            ):
                issues.append(
                    LinkIssue(
                        link=link,
                        reason="untracked or generated",
                        resolved=_display_path(candidate, resolved_root),
                    )
                )
                continue

            fragment = unquote(parsed.fragment)
            if (
                fragment
                and candidate.is_file()
                and candidate.suffix.lower() == ".md"
            ):
                anchors = anchor_cache.setdefault(
                    candidate,
                    anchors_in_document(candidate),
                )
                if fragment not in anchors:
                    resolved = f"{_display_path(candidate, resolved_root)}#{fragment}"
                    issues.append(
                        LinkIssue(
                            link=link,
                            reason="missing anchor",
                            resolved=resolved,
                        )
                    )

    return LinkCheckResult(
        documents=len(document_list),
        local_checked=local_checked,
        external_skipped=external_skipped,
        issues=tuple(
            sorted(
                issues,
                key=lambda issue: (
                    _display_path(issue.link.source, resolved_root),
                    issue.link.line,
                    issue.link.destination,
                ),
            )
        ),
    )


def main() -> int:
    try:
        tracked_files = tracked_repository_files(ROOT)
        documents = tracked_markdown_files(ROOT, tracked_files)
        result = check_markdown_links(ROOT, documents, tracked_files)
    except (OSError, RuntimeError, UnicodeError) as exc:
        print(f"Markdown link check could not run: {exc}", file=sys.stderr)
        return 2

    if result.issues:
        print(
            "Markdown link check failed: "
            f"{len(result.issues)} local target(s) are missing or invalid:",
            file=sys.stderr,
        )
        for issue in result.issues:
            source = _display_path(issue.link.source, ROOT)
            print(
                f"  {source}:{issue.link.line}: {issue.link.destination!r} -> "
                f"{issue.resolved} ({issue.reason})",
                file=sys.stderr,
            )
        print("Fix or remove every target listed above.", file=sys.stderr)
        return 1

    print(
        "Markdown link check passed: "
        f"{result.documents} tracked Markdown files; "
        f"{result.local_checked} local targets checked; "
        f"{result.external_skipped} external URLs skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
