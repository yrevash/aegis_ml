"""Fail if anything under ``src/`` is a placeholder rather than working code.

The user's requirement, in their words: *"for mock keep things separate not in code so it
can be really trusted."* This script is the enforcement. It is deliberately blunt — a
grep-level check that cannot be argued with — because the failure it prevents is a demo
where a number on screen came from a constant instead of a fit.

Three classes of finding:

``forbidden_token``
    A word that means "not finished" appearing in shipped source. Test doubles are
    legitimate, but they belong in ``tests/fixtures/``, never in ``src/``.

``empty_body``
    A function or method whose entire body is ``pass``, ``...`` or a bare docstring.
    A signature with no implementation type-checks, imports cleanly, and does nothing.

``swallowed_import``
    ``except ImportError`` with a body that does not raise. Aegis's most important rule is
    that a control which cannot run fails closed and says so; a swallowed import turns
    "AutoGluon is not installed" into "AutoGluon found nothing better", and the leaderboard
    cannot tell them apart.

**The opt-out is deliberately noisy.** A line may carry ``# audit-ok: <reason>`` to be
skipped, and the reason is mandatory — an empty one does not suppress. That keeps the rare
legitimate case (a capability probe that reports availability rather than choosing a code
path) reviewable in the diff, instead of pushing it into a config file nobody reads. Every
opt-out in this repo should be greppable in one command: ``grep -rn "audit-ok:" src/``.

Run: ``python scripts/audit_no_mocks.py`` — exit code 0 clean, 1 with findings.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

#: Words that mean "not real code". Matched case-insensitively on word boundaries.
#: ``sample``/``example`` are deliberately absent — they are legitimate in this domain
#: (``max_samples``, ``sample_frame``), and a check that cries wolf gets switched off.
FORBIDDEN = (
    "mock",
    "fake",
    "stub",
    "dummy",
    "placeholder",
    "todo",
    "fixme",
    "xxx",
    "hack",
    "not implemented",
    "notimplemented",
    "for now",
    "in a real",
    "in production you would",
)

#: Lines exempt from the token scan: this file names the words in order to ban them, and a
#: module may legitimately document that it refuses to do something.
EXEMPT_PATH_PARTS = ("tests", "fixtures", "templates")

_WORD = re.compile(r"|".join(rf"\b{re.escape(w)}\b" for w in FORBIDDEN), re.IGNORECASE)

#: An explicit, reason-carrying opt-out. The reason is mandatory: ``# audit-ok:`` with
#: nothing after it does not suppress anything.
_AUDIT_OK = re.compile(r"#\s*audit-ok:\s*(?P<reason>\S.*)$")


def _exempt_lines(text: str) -> set[int]:
    """Return the 1-indexed lines a reason-bearing ``# audit-ok:`` marker covers.

    A marker covers its own line **and the next one**, because the natural place to write
    the justification for an ``except`` clause is the line above it — and an AST handler
    node reports the ``except`` line, not the comment. Requiring the marker to sit inline
    on a long ``except (ImportError, ValueError, ModuleNotFoundError):`` would push it past
    the 100-column limit and get it reformatted away.
    """
    covered: set[int] = set()
    for n, line in enumerate(text.splitlines(), start=1):
        if _AUDIT_OK.search(line):
            covered.add(n)
            covered.add(n + 1)
    return covered


def _is_empty_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function's body is a docstring and/or ``pass``/``...`` only.

    A Protocol method or an ``@abstractmethod`` is allowed to be empty — that IS its
    implementation — so those are filtered by the caller, not here.
    """
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        return True
    return all(
        isinstance(stmt, ast.Pass)
        or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis)
        for stmt in body
    )


def _protocol_or_abstract(node: ast.AST, protocol_bodies: set[int]) -> bool:
    """Return whether an empty body is legitimate (Protocol member or abstract method)."""
    if id(node) in protocol_bodies:
        return True
    for dec in getattr(node, "decorator_list", []):
        name = dec.id if isinstance(dec, ast.Name) else getattr(dec, "attr", "")
        if "abstract" in str(name).lower() or str(name) == "overload":
            return True
    return False


def _collect_protocol_members(tree: ast.Module) -> set[int]:
    """Return ids of function nodes that are members of a ``Protocol`` class."""
    members: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases
        }
        if "Protocol" not in bases:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                members.add(id(child))
    return members


def _swallowed_imports(tree: ast.Module) -> list[int]:
    """Return line numbers of ``except ImportError`` handlers that do not raise."""
    bad: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc = ast.unparse(node.type) if node.type else ""
        if "ImportError" not in exc and "ModuleNotFoundError" not in exc:
            continue
        if not any(isinstance(s, ast.Raise) for s in ast.walk(node)):
            bad.append(node.lineno)
    return bad


def audit(src: Path = SRC) -> list[tuple[str, str, int, str]]:
    """Scan every Python file under ``src`` and return findings.

    Returns:
        ``(kind, relative_path, lineno, detail)`` tuples, empty when the tree is clean.
    """
    findings: list[tuple[str, str, int, str]] = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel.split("/") for part in EXEMPT_PATH_PARTS):
            continue
        text = path.read_text(encoding="utf-8")
        exempt = _exempt_lines(text)

        for lineno, line in enumerate(text.splitlines(), start=1):
            if lineno in exempt:
                continue
            match = _WORD.search(line)
            if match:
                findings.append(("forbidden_token", rel, lineno, match.group(0)))

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # a file that will not parse is the worst finding
            findings.append(("syntax_error", rel, exc.lineno or 0, str(exc.msg)))
            continue

        protocol_members = _collect_protocol_members(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_empty_body(node):
                if node.lineno not in exempt and not _protocol_or_abstract(node, protocol_members):
                    findings.append(("empty_body", rel, node.lineno, node.name))

        for lineno in _swallowed_imports(tree):
            if lineno in exempt:
                continue
            findings.append(("swallowed_import", rel, lineno, "except ImportError without raise"))

    return findings


def main() -> int:
    """Print findings and return a shell exit code."""
    findings = audit()
    if not findings:
        files = list(SRC.rglob("*.py"))
        optouts = sum(
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if _AUDIT_OK.search(ln))
            for f in files
        )
        print(f"PASS  no mocks, stubs, empty bodies or swallowed imports in {len(files)} source files")
        print(f"      {optouts} reviewed opt-out(s); list them with: grep -rn 'audit-ok:' src/")
        return 0
    print(f"FAIL  {len(findings)} finding(s) under src/\n")
    for kind, rel, lineno, detail in findings:
        print(f"  {kind:18} {rel}:{lineno}  {detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
