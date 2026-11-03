
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from app.llm import get_chat_model
from app.rag import RepositoryRAG

rag = RepositoryRAG()


# ============================================================
# Repository Configuration
# ============================================================

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".chroma",
    "dist",
    "build",
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".c",
    ".h", ".cs", ".sql", ".html", ".css", ".md",
}

MAX_FILE_SIZE = 100_000
MAX_FILES = 100


# ============================================================
# Repository Utilities
# ============================================================

def validate_repository(repo_path: str) -> Path:
    """Resolve and validate a repository directory."""

    root = Path(repo_path).resolve()

    if not root.is_dir():
        raise ValueError(
            f"Repository directory does not exist: {root}"
        )

    return root


def is_ignored(path: Path) -> bool:
    """Return True when a path contains an excluded directory."""

    return any(part in IGNORED_DIRS for part in path.parts)


# ============================================================
# Tool 1: List Repository Files
# ============================================================

@tool
def list_repository_files(repo_path: str) -> str:
    """List supported source files in the selected repository.

    Use this tool to discover project structure and locate modules.
    """

    root = validate_repository(repo_path)
    results = []

    for path in root.rglob("*"):
        relative = path.relative_to(root)

        if is_ignored(relative):
            continue

        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            results.append(relative.as_posix())

    results.sort()

    if not results:
        return "No supported source files were found."

    shown = results[:MAX_FILES]

    output = "\n".join(shown)

    if len(results) > MAX_FILES:
        output += (
            f"\n\nShowing {MAX_FILES} of {len(results)} files. "
            "Search for a specific module if more files are needed."
        )

    return output


# ============================================================
# Tool 2: Search Repository Using RAG
# ============================================================

@tool
def search_repository(repo_path: str, question: str) -> str:
    """Search the selected repository for relevant source code.

    Use this tool to locate implementations, functions, classes,
    dependencies, API endpoints, and related code.
    """

    validate_repository(repo_path)

    if not question.strip():
        return "Search question cannot be empty."

    # Refresh the selected repository's index before searching.
    rag.index_repository(repo_path)

    matches = rag.search(
        repo_path,
        question,
        k=4,
    )

    if not matches:
        return (
            "No matching indexed code was found. "
            "Try a more specific search query."
        )

    sections = []

    for match in matches:
        source = match.get("source", "Unknown source")
        content = match.get("content", "")

        sections.append(
            f"Source: {source}\n"
            f"Retrieved code excerpt:\n{content}"
        )

    return "\n\n---\n\n".join(sections)


# ============================================================
# Tool 3: Read Repository File
# ============================================================

@tool
def read_repository_file(
    repo_path: str,
    relative_path: str,
) -> str:
    """Read a supported source file from the selected repository.

    Use this tool to verify exact implementation details before
    making code-specific claims.

    Always provide a repository-relative file path.
    Never read secrets or environment files.
    """

    root = validate_repository(repo_path)

    target = (root / relative_path).resolve()

    # Prevent path traversal outside the repository.
    if not target.is_relative_to(root):
        return "Access denied: path is outside the repository."

    relative = target.relative_to(root)

    if is_ignored(relative):
        return "Access denied: file is inside an excluded directory."

    if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return "Access denied: unsupported file type."

    if not target.is_file():
        return "File not found."

    if target.stat().st_size > MAX_FILE_SIZE:
        return (
            f"File is larger than {MAX_FILE_SIZE} bytes. "
            "Use search_repository to locate relevant sections."
        )

    try:
        content = target.read_text(encoding="utf-8")

    except UnicodeDecodeError:
        return "Unable to read this file as UTF-8 text."

    except OSError as exc:
        return f"Unable to read file: {exc}"

    numbered_lines = [
        f"{line_number:4}: {line}"
        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        )
    ]

    return (
        f"File: {relative.as_posix()}\n"
        f"Total lines: {len(numbered_lines)}\n"
        "Content with line numbers:\n"
        + "\n".join(numbered_lines)
    )


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are ARGUS, an AI software engineering agent created by Amit Chaudhary.

IDENTITY:
- If asked who built, created, or developed you, say you were created by Amit Chaudhary.
- Describe yourself as an AI software engineering assistant for developers.
- Your purpose is to help understand codebases, investigate architecture, review code, identify bugs, and suggest practical fixes.
- Do not invent personal or professional details about your creator.
- Always identify yourself as ARGUS, never DevSage or DevSage AI.

RESPONSE LENGTH AND STYLE:
- Match the answer length to the question.
- Simple definitions and greetings should be concise (usually 2-5 sentences).
- Repository questions should include enough verified evidence to be useful, but avoid unrelated details.
- Provide deep, step-by-step explanations only when requested or needed to explain the verified implementation.
- Use clear Markdown structure when it improves readability.
- Finish complete sentences and do not output fragments.

Your responsibility is to investigate, understand, explain,
and debug the user's actual software repository.

You must base technical answers on verified repository evidence.

============================================================
REPOSITORY INVESTIGATION RULES
============================================================

1. Always use the repository root supplied in the user's message.

2. Never assume a fixed repository path.

3. For repository questions, inspect the actual source code.

4. Listing files is only the first investigation step.
   It is not sufficient for implementation questions.

5. Use read_repository_file to verify exact implementation details.

6. Use search_repository to locate relevant functions, classes,
   API endpoints, imports, and code relationships.

7. Use list_repository_files to discover the project structure.

8. Do not invent files, functions, classes, frameworks,
   API endpoints, or implementation behavior.

9. If evidence is insufficient, clearly explain what
   could not be verified.

10. Do not stop investigating when multiple files are
    necessary to answer the question.

============================================================
TOOL CALL RULES
============================================================

Pass only the arguments defined by each tool.

Never include function-call wrappers such as:
- type
- function
- name
- arguments

inside tool arguments.

Use these exact argument names:

list_repository_files:
{
    "repo_path": "<repository root supplied by the user>"
}

search_repository:
{
    "repo_path": "<repository root supplied by the user>",
    "question": "<specific search query>"
}

read_repository_file:
{
    "repo_path": "<repository root supplied by the user>",
    "relative_path": "<repository-relative file path>"
}

IMPORTANT:

- Always use relative_path for read_repository_file.
- Never use file_path.
- Never invent the repository root.
- Use the repository root provided in the current request.

============================================================
ARCHITECTURE INVESTIGATION WORKFLOW
============================================================

When the user asks about application architecture,
entry points, components, or execution flow:

STEP 1:
Call list_repository_files to discover actual project files.

STEP 2:
Inspect relevant frontend files, including:

- frontend/src/main.tsx
- frontend/src/App.tsx

STEP 3:
Inspect relevant backend files, including:

- backend/app/main.py
- backend/app/agent.py
- backend/app/rag.py

STEP 4:
Use search_repository when you need to locate specific
functions, endpoints, classes, or relationships.

STEP 5:
Read the relevant files using read_repository_file.

STEP 6:
Trace how the frontend, backend, AI agent, repository
tools, RAG system, and Gemini model interact.

STEP 7:
Only provide conclusions supported by inspected code.

If a file is missing, identify its actual location
using the repository file listing.

Do not guess based on filenames.

============================================================
ARCHITECTURE ANSWER FORMAT
============================================================

For complete architecture questions, organize the answer
using these sections:

1. Frontend Entry Point
2. Backend Entry Point
3. AI Agent Implementation
4. RAG Implementation
5. Complete Request and Execution Flow
6. How the Components Work Together

Include actual file paths, function names, classes,
and API endpoints when verified.

============================================================
GENERAL ANSWERING RULES
============================================================

- Answer the user's actual question directly.
- Use actual repository-relative file paths.
- Explain technical concepts clearly.
- Prefer concise and practical explanations.
- Use Markdown headings and code blocks when useful.
- Distinguish verified facts from assumptions.
- Never claim that code was inspected if it was not.
- Do not describe unrelated features unless requested.

You are a read-only repository investigation agent.
Do not modify repository files.
"""


# ============================================================
# LangChain / LangGraph Agent
# ============================================================

def create_repository_agent():
    """Build the Gemini-backed LangGraph agent on demand."""
    return create_agent(
        model=get_chat_model(max_output_tokens=2048),
        tools=[
            list_repository_files,
            search_repository,
            read_repository_file,
        ],
        system_prompt=SYSTEM_PROMPT,
    )
