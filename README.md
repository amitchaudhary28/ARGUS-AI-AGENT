# ARGUS | AI Software Engineering Agent

### Where Code Meets Intelligence.

ARGUS is an AI-assisted software engineering platform that helps developers understand, analyze, debug and improve existing codebases. It combines repository-aware AI assistance, automated code analysis, RAG and controlled code editing in one unified workspace.

## Overview

ARGUS bridges the gap between general-purpose AI chatbots and practical software engineering workflows by connecting AI assistance with actual project repositories.

Developers can explore codebases, identify potential issues, request code improvements, review proposed changes and run development checks — all from a single interface.

## Key Features

- **AI Assistant:** Conversational programming assistance, technical explanations and coding support.
- **Repository Agent:** Understand and interact with project code using repository-aware AI assistance.
- **Repository Import:** Import projects through GitHub URLs, ZIP uploads or folder uploads.
- **Repository Scanner:** Combine static checks and AI-assisted analysis to identify potential bugs and code-quality issues.
- **AI Code Editor:** Generate focused code changes with diff previews, validation and approval controls.
- **Code Review:** Analyze source code and receive actionable improvement suggestions.
- **Run Tests:** Execute predefined testing, linting, build and Git status checks.
- **RAG-Based Retrieval:** Retrieve relevant repository context using embeddings and vector search.
- **Conversation Management:** Maintain chat history and switch between dedicated workspaces.

## Tech Stack

| Category | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, CSS |
| Backend | Python, FastAPI |
| AI Integration | Google Gemini, LangChain |
| Agent Framework | LangGraph |
| Vector Database | ChromaDB |
| Application Database | SQLite |
| Deployment | Vercel, Render |
| Version Control | Git, GitHub |

## Architecture

```text
             React + TypeScript
                     |
                     v
                FastAPI
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Gemini    LangGraph   Repository
          |       + LangChain  Tools
          |          |
          +----------+
                     |
                     v
                  ChromaDB
               (Vector Search)
```

## How It Works

1. Import a repository using GitHub, ZIP or folder upload.
2. Explore project files and access repository-aware assistance.
3. Ask questions or run repository analysis.
4. Retrieve relevant code context through the RAG pipeline.
5. Generate proposed improvements and inspect code diffs.
6. Approve changes and execute relevant validation tasks.

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js and npm
- Git
- Google Gemini API key

### Clone the Repository

```bash
git clone https://github.com/amitchaudhary28/ARGUS-AI-AGENT.git
cd ARGUS-AI-AGENT
```

### Backend Setup

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `backend/.env` and configure your Gemini credentials:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=your_model_name
GEMINI_EMBEDDING_MODEL=your_embedding_model
```

Start the backend:

```bash
python -m uvicorn app.main:app --reload
```

### Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

## Live Demo

- **Application:** https://argus-ai-agent-beige.vercel.app
- **Backend API:** https://argus-ai-agent.onrender.com
- **API Documentation:** https://argus-ai-agent.onrender.com/docs

## Important Notes

- AI-generated findings and code suggestions should be reviewed before being accepted.
- Repository scanning is subject to configured file-type, size and file-count limits.
- Gemini functionality requires valid API credentials and available quota.
- Render Free uses ephemeral storage; imported repositories and locally stored database or vector data may not persist across restarts.

## Future Scope

- Incremental repository indexing.
- Multi-file code modifications.
- Git-aware code editing and rollback.
- Advanced code-review reports.
- Smarter test selection.
- Expanded language support.

---

**ARGUS — Where Code Meets Intelligence.**