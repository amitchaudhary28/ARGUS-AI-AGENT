"""Regression tests for ARGUS's approval-gated, exact-text Python patches.

Run from backend/: python -m pytest -q
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import main as api


class FakeRequest:
    def __init__(self, user_id: str = "test-user") -> None:
        self.state = SimpleNamespace(argus_user={"id": user_id})


@pytest.mark.asyncio
async def test_large_python_file_uses_small_patch_without_full_file_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    source = "# header\n" + "value = 1\n" * 20_000 + "# footer\n"
    target = tmp_path / "large_module.py"
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(api, "ask_gemini", lambda **kwargs: _fake_patch(kwargs))

    proposed, _ = await api.generate_ai_fix(
        "large_module.py", source, "Correct value assignment", 2
    )
    assert len(proposed) > 100_000
    assert proposed.count("value = 2\n") == 1
    assert target.read_text(encoding="utf-8") == source


async def _fake_patch(kwargs):
    return ('{"old_text":"# header\\nvalue = 1\\n", "new_text":"# header\\nvalue = 2\\n"}', "gemini-test")


@pytest.mark.asyncio
async def test_patch_must_match_exactly_one_location(monkeypatch: pytest.MonkeyPatch) -> None:
    async def duplicate_patch(**kwargs):
        return ('{"old_text":"x = 1\\n", "new_text":"x = 2\\n"}', "gemini-test")

    monkeypatch.setattr(api, "ask_gemini", duplicate_patch)
    with pytest.raises(HTTPException) as exc:
        await api.generate_ai_fix("x.py", "x = 1\nx = 1\n", "fix x", 1)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_suggestion_does_not_write_until_approval(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = tmp_path / "sample.py"
    original = "def answer():\n    return 1\n"
    target.write_text(original, encoding="utf-8")

    async def make_patch(**kwargs):
        return (original.replace("return 1", "return 2"), "gemini-test")

    monkeypatch.setattr(api, "generate_ai_fix", make_patch)
    response = await api.suggest_repository_fix(
        api.SuggestFixRequest(
            repo_path=str(tmp_path), file_path="sample.py",
            finding_description="Return value is incorrect", line=2,
        ),
        FakeRequest(),
    )
    assert response["requires_approval"] is True
    assert response["applied"] is False
    assert target.read_text(encoding="utf-8") == original

    applied = api.apply_repository_fix(
        api.ApplyFixRequest(suggestion_id=response["suggestion_id"]), FakeRequest()
    )
    assert applied["success"] is True
    assert target.read_text(encoding="utf-8") == "def answer():\n    return 2\n"


@pytest.mark.asyncio
async def test_stale_file_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = tmp_path / "sample.py"
    original = "x = 1\n"
    target.write_text(original, encoding="utf-8")

    async def make_patch(**kwargs):
        return ("x = 2\n", "gemini-test")

    monkeypatch.setattr(api, "generate_ai_fix", make_patch)
    suggestion = await api.suggest_repository_fix(
        api.SuggestFixRequest(repo_path=str(tmp_path), file_path="sample.py", finding_description="Fix x value", line=1),
        FakeRequest(),
    )
    target.write_text("x = 99\n", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        api.apply_repository_fix(api.ApplyFixRequest(suggestion_id=suggestion["suggestion_id"]), FakeRequest())
    assert exc.value.status_code == 409
    assert target.read_text(encoding="utf-8") == "x = 99\n"


def test_validation_failure_after_replace_rolls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = tmp_path / "rollback.py"
    original = "x = 1\n"
    target.write_text(original, encoding="utf-8")
    suggestion_id = "rollback-test"
    api.pending_fix_suggestions[suggestion_id] = {
        "repo_path": str(tmp_path), "file_path": str(target),
        "original_hash": api.calculate_file_hash(original),
        "original_code": original, "proposed_code": "x = 2\n",
        "finding_description": "test", "line": 1,
        "created_at": api.time.time(), "owner_id": "test-user",
    }
    real_validate = api.validate_proposed_fix
    calls = 0

    def fail_after_write(path, code):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_validate(path, code)
        return {"status": "failed", "valid": False, "message": "forced verification failure"}

    monkeypatch.setattr(api, "validate_proposed_fix", fail_after_write)
    with pytest.raises(HTTPException) as exc:
        api.apply_repository_fix(api.ApplyFixRequest(suggestion_id=suggestion_id), FakeRequest())
    assert exc.value.status_code == 500
    assert target.read_text(encoding="utf-8") == original


def test_account_conversations_are_isolated_and_restore_after_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(api, "AUTH_DB_PATH", tmp_path / "accounts.sqlite3")
    api.init_auth_db()
    user_a = FakeRequest("account-a")
    user_b = FakeRequest("account-b")
    payload = api.ConversationSyncRequest(sessions=[{
        "id": "chat-a", "title": "A private chat", "mode": "agent",
        "messages": [{"role": "user", "content": "private"}],
        "steps": [], "updatedAt": 1,
    }])
    api.put_account_conversations(user_a, payload)
    assert api.get_account_conversations(user_a)["sessions"][0]["id"] == "chat-a"
    assert api.get_account_conversations(user_b)["sessions"] == []
    api.put_account_conversations(user_a, api.ConversationSyncRequest(sessions=[]))
    assert api.get_account_conversations(user_a)["sessions"] == []
