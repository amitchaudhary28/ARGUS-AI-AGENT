
from pathlib import Path

import pytest

from app.rag import RepositoryRAG


def test_tokenize_returns_lowercase_tokens():
    tokens = RepositoryRAG._tokenize("Repository SearchFunction")

    assert "repository" in tokens
    assert "searchfunction" in tokens


def test_tokenize_ignores_stop_words():
    tokens = RepositoryRAG._tokenize("this is my repository")

    assert "this" not in tokens
    assert "is" not in tokens
    assert "repository" in tokens


def test_repo_id_is_stable(tmp_path: Path):
    first = RepositoryRAG._repo_id(tmp_path)
    second = RepositoryRAG._repo_id(tmp_path)

    assert first == second


def test_repository_files_rejects_missing_directory(tmp_path: Path):
    missing = tmp_path / "missing"
    rag = RepositoryRAG.__new__(RepositoryRAG)

    with pytest.raises(
        ValueError,
        match="Repository path is not a directory",
    ):
        rag._repository_files(str(missing))
