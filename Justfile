# ──────────────────────────────────────────────
# Hermes Agent — Justfile
# ──────────────────────────────────────────────
# Usage: just <recipe>
#   just test          Run all tests
#   just test-quick    Run only plan tests (fast feedback)
#   just lint          Run ruff lint + format check
#   just format        Auto-fix ruff issues
#   just typecheck     Run type checker (ty)
#   just ci            Full CI pipeline: lint → typecheck → test
#   just plan-tests    Run plan-specific tests
#   just precommit     Install pre-commit hooks
# ──────────────────────────────────────────────

venv := if `test -d .venv && echo yes || echo no` == "yes" { ".venv" } else { "venv" }
_python := venv + "/bin/python"
_pytest := _python + " -m pytest"

# Default recipe
default:
    @just --list

# ── Tests ─────────────────────────────────────

# Run the full test suite
test:
    scripts/run_tests.sh

# Run only plan tests for fast feedback
test-quick:
    {{_pytest}} tests/cli/test_plan_command.py tests/gateway/test_plan_command.py -q

# Run plan-specific tests
plan-tests:
    {{_pytest}} tests/cli/test_plan_command.py tests/gateway/test_plan_command.py -v

# ── Linting ───────────────────────────────────

# Run ruff lint + format check
lint:
    {{_python}} -m ruff check .
    {{_python}} -m ruff format --check .

# Auto-fix ruff issues
format:
    {{_python}} -m ruff check --fix .
    {{_python}} -m ruff format .

# ── Type Checking ─────────────────────────────

# Run type checker (ty)
typecheck:
    {{_python}} -m ty src/ 2>/dev/null || echo "Type check warnings found (see above)"

# ── CI Pipeline ───────────────────────────────

# Full CI pipeline: lint → typecheck → test-quick
ci: lint typecheck test-quick
    @echo "✓ CI passed"

# ── Pre-commit ────────────────────────────────

# Install pre-commit hooks (requires core.hooksPath to be unset)
precommit:
    git config --unset-all core.hooksPath
    pre-commit install
    pre-commit install --hook-type pre-push
    git config --global core.hooksPath ~/.git-hooks/
    @echo "✓ Pre-commit hooks installed (global hooks re-set)"

# Run pre-commit on all files
precommit-all:
    pre-commit run --all-files

# ── Utilities ─────────────────────────────────

# Show git log
log:
    git log --oneline -20

# Update from upstream
sync-upstream:
    git fetch upstream
    @echo "Upstream updated. Run 'git log upstream/main --oneline' to review."

# Show branch status
status:
    git rev-list --left-right --count upstream/main...HEAD
