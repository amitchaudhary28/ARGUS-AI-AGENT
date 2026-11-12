import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import "./App.css";

type Mode = "chat" | "agent";


type ScanFinding = {
  title: string;
  description: string;
  line?: number;
  verified: boolean;
};

type FixSuggestion = {
  suggestion_id: string;
  file: string;
  finding: string;
  line?: number;
  explanation: string;
  diff: string;
  original_code: string;
  proposed_code: string;
  validation: {
    status: string;
    valid: boolean;
    message: string;
  };
  requires_approval: boolean;
  applied: boolean;
};

type ScanReport = {
  file: string;
  status: "completed" | "error" | "skipped" | string;
  review?: string;
  error?: string;
  reason?: string;
  static_analysis?: {
    tool?: string;
    status?: string;
    count?: number;
    findings?: unknown[];
  };
  language_validation?: {
    status?: string;
    count?: number;
    findings?: unknown[];
  };
};

type FindingSummary = {
  ruff_findings?: number;
  deterministic_findings?: number;
  ai_findings?: number;
  unverified_findings?: number;
};

type RepositoryScanSummary = {
  files_considered: number;
  files_completed: number;
  files_failed: number;
  files_skipped: number;
  files_skipped_large: number;
  static_findings_count: number;
  typescript_findings_count: number;
  finding_summary?: Partial<FindingSummary>;
};

type RepositoryScanResult = {
  repository?: string;
  files_scanned?: number;
  files_considered?: number;
  files_failed?: number;
  files_skipped_large?: number;
  max_files?: number;
  static_findings_count?: number;
  typescript_validation?: {
    count?: number;
    status?: string;
  };
  summary?: Partial<RepositoryScanSummary>;
  reports?: ScanReport[];
  message?: string;
  note?: string;
  model?: string;
  read_only?: boolean;
};


type Message = {
  role: "user" | "assistant";
  content: string;
};

type Session = {
  id: string;
  title: string;
  mode: Mode;
  messages: Message[];
  steps: string[];
  updatedAt: number;
};

const API_URL = import.meta.env.VITE_API_URL || `${typeof window !== "undefined" ? window.location.protocol : "http:"}//${typeof window !== "undefined" ? window.location.hostname : "127.0.0.1"}:8000`;

function getFixRequestError(error: unknown, action: string): string {
  const message = error instanceof Error ? error.message : String(error);
  const normalized = message.toLowerCase();

  if (
    error instanceof TypeError ||
    normalized.includes("failed to fetch") ||
    normalized.includes("networkerror") ||
    normalized.includes("network request failed")
  ) {
    return (
      `Cannot reach the ARGUS backend at ${API_URL}. ` +
      `Make sure the FastAPI server is running, the API URL is correct, ` +
      `and CORS allows this frontend. Then retry the ${action}.`
    );
  }

  return message || `Unable to ${action}.`;
}

const LEGACY_STORAGE_KEY = "devsage.sessions.v2";

function loadSessions(storageKey: string): Session[] {
  try {
    const saved = localStorage.getItem(storageKey);
    if (!saved) return [];

    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .filter(
        (item: unknown): item is Record<string, unknown> =>
          typeof item === "object" &&
          item !== null &&
          typeof (item as Record<string, unknown>).id === "string" &&
          typeof (item as Record<string, unknown>).title === "string" &&
          Array.isArray((item as Record<string, unknown>).messages)
      )
      .map((item): Session => ({
        id: item.id as string,
        title: item.title as string,
        mode: item.mode === "chat" ? "chat" : "agent",
        messages: (item.messages as Message[]).filter(
          (message) =>
            message &&
            (message.role === "user" || message.role === "assistant") &&
            typeof message.content === "string"
        ),
        steps: Array.isArray(item.steps)
          ? item.steps.filter(
              (step): step is string => typeof step === "string"
            )
          : [],
        updatedAt:
          typeof item.updatedAt === "number" ? item.updatedAt : Date.now(),
      }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

function makeSession(
  mode: Mode,
  question = "New conversation"
): Session {
  return {
    id: crypto.randomUUID(),
    title:
      question.length > 42
        ? `${question.slice(0, 42)}…`
        : question,
    mode,
    messages: [],
    steps: [],
    updatedAt: Date.now(),
  };
}

function CodeBlock({
  code,
  language,
}: {
  code: string;
  language: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="code-block">
      <div className="code-header">
        <span>{language || "code"}</span>
        <button
          type="button"
          className="copy-code"
          onClick={copyCode}
        >
          {copied ? "Copied ✓" : "Copy code"}
        </button>
      </div>

      <div className="code-scroll">
        <SyntaxHighlighter
          language={language || "text"}
          style={oneDark}
          showLineNumbers
          wrapLongLines={false}
          customStyle={{
            margin: 0,
            padding: "16px",
            background: "#090b10",
            fontSize: "12px",
            lineHeight: 1.7,
            whiteSpace: "pre",
            overflowWrap: "normal",
            wordBreak: "normal",
          }}
          codeTagProps={{
            style: {
              fontFamily: 'Consolas, "Courier New", monospace',
              whiteSpace: "pre",
              overflowWrap: "normal",
              wordBreak: "normal",
            },
          }}
        >
          {code.replace(/\n$/, "")}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre({ children }) {
            return <>{children}</>;
          },
          code({ className, children, ...props }) {
            const match = /language-([\w+#.-]+)/.exec(className || "");
            const code = String(children).replace(/\n$/, "");
            const isBlock = Boolean(match) || code.includes("\n");

            if (!isBlock) {
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }

            return (
              <CodeBlock
                code={code}
                language={match?.[1] || "text"}
              />
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ErrorAlert({ message, title = "Error", onClose, style }: { message: string; title?: string; onClose?: () => void; style?: React.CSSProperties }) {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setVisible(false);
      onClose?.();
    }, 5000);
    return () => window.clearTimeout(timeout);
  }, [message, onClose]);
  if (!visible || !message) return null;
  return (
    <div className="error-message" role="alert" style={style}>
      <span><strong>{title}:</strong> {message}</span>
      <button type="button" className="error-dismiss" aria-label="Dismiss error" onClick={() => { setVisible(false); onClose?.(); }}>×</button>
    </div>
  );
}

function Workspace({ user, onSignOut, onSignIn, isGuest = false }: { user: ArgusUser; onSignOut: () => void; onSignIn: () => void; isGuest?: boolean }) {
  const sessionStorageKey = isGuest ? "argus.sessions.guest" : `argus.sessions.${user.id}`;
  const [sessions, setSessions] = useState<Session[]>(() => loadSessions(sessionStorageKey));
  const [accountSyncReady, setAccountSyncReady] = useState(false);
  const [syncStatus, setSyncStatus] = useState<"syncing" | "synced" | "error">(isGuest ? "synced" : "syncing");
  const storageErrorLoggedRef = useRef(false);
  const [syncAttempt, setSyncAttempt] = useState(0);
  const [accountPanel, setAccountPanel] = useState<"profile" | "preferences" | null>(null);
  const [enterToSend, setEnterToSend] = useState(() => localStorage.getItem(`argus.enterToSend.${user.id}`) !== "false");

  const [mode, setMode] = useState<Mode>(() => {
    const initial = loadSessions(sessionStorageKey);
    const preferred = localStorage.getItem(`argus.defaultMode.${user.id}`);
    return initial[0]?.mode ?? (preferred === "chat" ? "chat" : "agent");
  });

  const [activeId, setActiveId] = useState<string | null>(() => {
    const initial = loadSessions(sessionStorageKey);
    return initial[0]?.id ?? null;
  });

  const [input, setInput] = useState("");
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [voiceNotice, setVoiceNotice] = useState("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any>(null);
  const [attachments, setAttachments] = useState<File[]>([]);
  const attachmentInputRef = useRef<HTMLInputElement>(null);
  const [workspaceMenuOpen, setWorkspaceMenuOpen] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [repoImportUrl, setRepoImportUrl] = useState("");
  const [repoImportToken, setRepoImportToken] = useState("");
  const [repoImportBusy, setRepoImportBusy] = useState(false);
  const [repoImportMessage, setRepoImportMessage] = useState("");
  const [repoImportError, setRepoImportError] = useState("");
  const zipInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [repoPath, setRepoPath] = useState(() => {
    const savedPath = localStorage.getItem(`argus.repoPath.${user.id}`) || "";
    return /^C:\\Users\\amitc\\ARGUS$/i.test(savedPath) ? "" : savedPath;
  });
  const [repoName, setRepoName] = useState(() => {
    const savedName = localStorage.getItem(`argus.repoName.${user.id}`) || "";
    // Older upload responses exposed the temporary workspace ID as the name.
    return /^[a-f0-9]{24,}$/i.test(savedName) ? "Imported project" : savedName;
  });
  const [repoSource, setRepoSource] = useState(() => localStorage.getItem(`argus.repoSource.${user.id}`) || "");
  const [repoFiles, setRepoFiles] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(`argus.repoFiles.${user.id}`) || "[]");
      return Array.isArray(saved) ? saved.filter((item): item is string => typeof item === "string") : [];
    } catch {
      return [];
    }
  });

  const editableRepositoryFiles = repoFiles.filter((file) =>
    /\.(py|js|jsx|ts|tsx)$/i.test(file) &&
    !/(^|\/)\.env(?:\.|$)/i.test(file)
  );

  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
    folderInputRef.current?.setAttribute("directory", "");
  }, []);

  useEffect(() => {
    if (repoPath.trim()) localStorage.setItem(`argus.repoPath.${user.id}`, repoPath.trim());
    else localStorage.removeItem(`argus.repoPath.${user.id}`);
  }, [repoPath, user.id]);

  useEffect(() => {
    const key = `argus.repoName.${user.id}`;
    if (repoName) localStorage.setItem(key, repoName);
    else localStorage.removeItem(key);
  }, [repoName, user.id]);

  useEffect(() => {
    const key = `argus.repoSource.${user.id}`;
    if (repoSource) localStorage.setItem(key, repoSource);
    else localStorage.removeItem(key);
  }, [repoSource, user.id]);

  useEffect(() => {
    const key = `argus.repoFiles.${user.id}`;
    if (repoFiles.length) localStorage.setItem(key, JSON.stringify(repoFiles));
    else localStorage.removeItem(key);
  }, [repoFiles, user.id]);

  useEffect(() => {
    if (!repoPath.trim()) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(
          `${API_URL}/api/repository/files?repo_path=${encodeURIComponent(repoPath)}`,
          { credentials: "include" }
        );
        if (!response.ok) throw new Error("Imported workspace is no longer available.");
        const listing = await response.json() as { files?: string[] };
        if (!cancelled) {
          const availableFiles = Array.isArray(listing.files) ? listing.files : [];
          setRepoFiles(availableFiles);
          setEditFilePath((current) =>
            current && availableFiles.includes(current)
              ? current
              : availableFiles.find((file) => /\.(py|js|jsx|ts|tsx)$/i.test(file)) || ""
          );
        }
      } catch {
        if (!cancelled) {
          setRepoPath("");
          setRepoName("");
          setRepoSource("");
          setRepoFiles([]);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [repoPath]);


  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const textarea = promptRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }, [input]);


  const [scanOpen, setScanOpen] = useState(false);
  const [scanLoading, setScanLoading] = useState(false);
  const [scanMaxFiles, setScanMaxFiles] = useState(5);
  const [scanResult, setScanResult] =
    useState<RepositoryScanResult | null>(null);
  const [scanError, setScanError] = useState("");

  const [engineeringOpen, setEngineeringOpen] = useState(false);
  const [codeEditorOpen, setCodeEditorOpen] = useState(false);
  const [engineeringAction, setEngineeringAction] = useState("run_tests");
  const [engineeringLoading, setEngineeringLoading] = useState(false);
  const engineeringRunLockRef = useRef(false);
  const [engineeringError, setEngineeringError] = useState("");
  const [editFilePath, setEditFilePath] = useState("");
  const [editInstruction, setEditInstruction] = useState("");

  const [fixLoading, setFixLoading] = useState(false);
  const [fixActionLoading, setFixActionLoading] = useState(false);
  const [fixError, setFixError] = useState("");
  const [fixMessage, setFixMessage] = useState("");
  const [fixSuggestion, setFixSuggestion] =
    useState<FixSuggestion | null>(null);

  const activeSession =
    sessions.find((session) => session.id === activeId) ?? null;

  const messages = activeSession?.messages ?? [];
  const steps = activeSession?.steps ?? [];

  const modeSessions = sessions.filter(
    (session) => session.mode === mode
  );

  useEffect(() => {
    if (isGuest) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${API_URL}/api/account/conversations`, { credentials: "include" });
        if (!response.ok) throw new Error("Unable to restore account conversations.");
        const data = await response.json();
        if (cancelled) return;
        const remoteSessions = Array.isArray(data.sessions) ? data.sessions as Session[] : [];
        if (remoteSessions.length) {
          setSessions(remoteSessions);
          setActiveId(remoteSessions[0]?.id ?? null);
          if (remoteSessions[0]) setMode(remoteSessions[0].mode);
        }
        // If this account has no server history yet, retain its locally migrated
        // history and let the following sync effect upload it exactly once.
        setSyncStatus("synced");
        setError("");
        setAccountSyncReady(true);
      } catch (syncError) {
        if (!cancelled) {
          setSyncStatus("error");
          setError(syncError instanceof Error ? syncError.message : "Account history could not be restored. Local history remains available.");
          setAccountSyncReady(false); // Do not overwrite remote history until it has been restored successfully.
        }
      }
    })();
    return () => { cancelled = true; };
  }, [user.id, syncAttempt, isGuest]);

  useEffect(() => {
    try {
      localStorage.setItem(sessionStorageKey, JSON.stringify(sessions));
      storageErrorLoggedRef.current = false;
    } catch (storageError) {
      if (!storageErrorLoggedRef.current) {
        console.error("Unable to save chat history in browser storage.", storageError);
        storageErrorLoggedRef.current = true;
      }
    }
    if (isGuest || !accountSyncReady) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(`${API_URL}/api/account/conversations`, {
          method: "PUT", credentials: "include", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessions }),
        });
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || `Conversation sync failed (HTTP ${response.status}).`);
        }
        if (!cancelled) setSyncStatus("synced");
      } catch (syncError) {
        if (!cancelled) {
          setSyncStatus("error");
          setError(syncError instanceof Error ? `${syncError.message} Your local history is preserved.` : "Account sync is unavailable. Your local history is preserved.");
        }
      }
    })();
    return () => { cancelled = true; };
  }, [sessions, accountSyncReady, isGuest, sessionStorageKey]);

  function updateSession(
    id: string,
    update: (session: Session) => Session
  ) {
    setSessions((previous) =>
      previous.map((session) =>
        session.id === id ? update(session) : session
      )
    );
  }

  function createChat(nextMode: Mode = mode) {
    if (loading) return;

    setScanOpen(false);
    setEngineeringOpen(false);
    setCodeEditorOpen(false);
    setScanResult(null);
    setScanError("");
    setFixSuggestion(null);
    setFixError("");
    setFixMessage("");

    const session = makeSession(nextMode);
    setSessions((previous) => [session, ...previous]);
    setActiveId(session.id);
    setMode(nextMode);
    setInput("");
    setError("");
  }

  function switchMode(nextMode: Mode) {
    if (loading) return;

    setScanOpen(false);
    setEngineeringOpen(false);
    setCodeEditorOpen(false);
    setScanResult(null);
    setScanError("");
    setMode(nextMode);
    setInput("");
    setError("");

    const existing = sessions.find(
      (session) => session.mode === nextMode
    );

    if (existing) {
      setActiveId(existing.id);
    } else {
      const session = makeSession(nextMode);
      setSessions((previous) => [session, ...previous]);
      setActiveId(session.id);
    }
  }

  function selectSession(session: Session) {
    if (loading) return;

    setScanOpen(false);
    setEngineeringOpen(false);
    setCodeEditorOpen(false);
    setScanResult(null);
    setScanError("");
    setFixSuggestion(null);
    setActiveId(session.id);
    setMode(session.mode);
    setInput("");
    setError("");
  }

  function deleteSession(id: string) {
    if (loading && id === activeId) return;

    const remaining = sessions.filter(
      (session) => session.id !== id
    );

    setSessions(remaining);

    if (id === activeId) {
      const replacement = remaining.find(
        (session) => session.mode === mode
      );

      if (replacement) {
        setActiveId(replacement.id);
      } else {
        const fresh = makeSession(mode);
        setSessions((previous) => [fresh, ...previous]);
        setActiveId(fresh.id);
      }
    }
  }

function toggleVoiceInput() {
  const SpeechRecognitionCtor =
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition;

  if (!SpeechRecognitionCtor) {
    setVoiceNotice("Voice input is not supported. Try Chrome or Edge.");
    return;
  }

  // Stop recording
  if (isListening) {
    try {
      recognitionRef.current?.stop();
    } catch {
      recognitionRef.current?.abort?.();
    }

    setIsListening(false);
    setVoiceNotice("");
    return;
  }

  try {
    const recognition = new SpeechRecognitionCtor();

    recognition.lang = "en-IN";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setIsListening(true);
      setVoiceNotice("Listening… speak now.");
    };

    recognition.onresult = (event: any) => {
      let transcript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          transcript += event.results[i][0].transcript;
        }
      }

      if (transcript.trim()) {
        setInput((previous) =>
          `${previous}${previous.trim() ? " " : ""}${transcript.trim()}`
        );
      }
    };

    recognition.onerror = (event: any) => {
      setVoiceNotice(
        event.error === "not-allowed"
          ? "Microphone permission denied."
          : `Voice input error: ${event.error}`
      );
      setIsListening(false);
      recognitionRef.current = null;
    };

    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
  } catch {
    setIsListening(false);
    setVoiceNotice("Unable to start voice input.");
    recognitionRef.current = null;
  }
}

  useEffect(() => () => {
    recognitionRef.current?.stop?.();
  }, []);

  function handleAttachmentSelection(event: React.ChangeEvent<HTMLInputElement>) {
    setError("");
    const selected = Array.from(event.target.files ?? []);
    const textAllowed = /\.(txt|md|json|csv|log|py|js|jsx|ts|tsx|html|css|java|c|cpp|h|sql|yaml|yml|xml|toml|ini|sh|bat|ps1)$/i;
    const richAllowed = /\.(pdf|png|jpe?g|webp)$/i;
    const valid = selected.filter((file) => {
      const isRichAttachment = richAllowed.test(file.name);
      if (!textAllowed.test(file.name) && !(mode === "chat" && isRichAttachment)) { setError(`${file.name}: unsupported file type. AI Assistant supports PDF, PNG, JPG and WEBP images, plus text/code files.`); return false; }
      const maxBytes = isRichAttachment ? 5 * 1024 * 1024 : 2 * 1024 * 1024;
      if (file.size > maxBytes) { setError(`${file.name}: file exceeds the ${isRichAttachment ? "5 MB" : "2 MB"} per-file limit.`); return false; }
      return true;
    });
    setAttachments((previous) => [...previous, ...valid].filter((file, index, all) => all.findIndex((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified) === index));
    event.target.value = "";
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const question = input.trim();
    if ((!question && attachments.length === 0) || loading) return;

    if (mode === "agent" && !repoPath.trim()) {
      setError("Please enter your repository folder path.");
      return;
    }

    // Stop microphone when submitting a message
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        recognitionRef.current.abort?.();
      }

      recognitionRef.current = null;
    }

    setIsListening(false);
    setVoiceNotice("");

    let attachmentContext = "";
    const multimodalAttachments: { name: string; mime_type: string; content_base64: string }[] = [];
    if (attachments.length) {
      const fileParts: string[] = [];
      let remainingChars = mode === "agent" ? 18000 : 40000;
      for (const file of attachments) {
        const isPdfOrImage = /\.(pdf|png|jpe?g|webp)$/i.test(file.name);
        if (isPdfOrImage) {
          if (mode !== "chat") continue;
          const buffer = await file.arrayBuffer();
          const bytes = new Uint8Array(buffer);
          let binary = "";
          for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
          const mimeType = file.type || (file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : file.name.toLowerCase().endsWith(".png") ? "image/png" : file.name.toLowerCase().endsWith(".webp") ? "image/webp" : "image/jpeg");
          multimodalAttachments.push({ name: file.name, mime_type: mimeType, content_base64: btoa(binary) });
          continue;
        }
        if (remainingChars <= 0) break;
        const content = await file.text();
        const excerpt = content.slice(0, remainingChars);
        fileParts.push(`\n\n--- Attached file: ${file.name} ---\n${excerpt}${content.length > excerpt.length ? "\n[File content truncated to fit the request limit.]" : ""}\n--- End of ${file.name} ---`);
        remainingChars -= excerpt.length;
      }
      attachmentContext = fileParts.join("");
    }
    const effectiveQuestion = `${question || "Please review and explain the attached file(s)."}${attachmentContext}`;

    let session = activeSession;

    if (!session || session.mode !== mode) {
      session = makeSession(mode, question || attachments.map((file) => file.name).join(", "));
      setSessions((previous) => [session!, ...previous]);
      setActiveId(session.id);
    }

    const sessionId = session.id;

    updateSession(sessionId, (current) => ({
      ...current,
      title:
        current.messages.length === 0
          ? (question || attachments.map((file) => file.name).join(", ")).slice(0, 42)
          : current.title,
      messages: [
        ...current.messages,
        { role: "user", content: question || `Please review the attached file(s): ${attachments.map((file) => file.name).join(", ")}` },
      ],
      steps: [],
      updatedAt: Date.now(),
    }));

    setInput("");
    setError("");
    setLoading(true);

    try {
      const endpoint =
        mode === "agent"
          ? `${API_URL}/api/agent`
          : `${API_URL}/api/chat`;

      const payload =
        mode === "agent"
          ? { repo_path: repoPath.trim(), question: effectiveQuestion }
          : { message: effectiveQuestion, attachments: multimodalAttachments };

      const response = await fetch(endpoint, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? `HTTP ${response.status}`)
        );
      }

      const answer =
        mode === "chat"
          ? data.reply ?? data.answer
          : data.answer ?? data.reply;

      if (typeof answer !== "string" || !answer.trim()) {
        throw new Error("The backend returned an empty answer.");
      }

      setAttachments([]);

      const activity = Array.isArray(data.steps)
        ? data.steps.map((step: unknown) => {
            if (typeof step === "string") return step;

            if (step && typeof step === "object") {
              const item = step as Record<string, unknown>;
              const name = String(item.tool ?? item.name ?? "Tool");
              const result = String(item.result ?? item.content ?? "");
              return result ? `${name}: ${result}` : name;
            }

            return String(step);
          })
        : [];

      updateSession(sessionId, (current) => ({
        ...current,
        messages: [
          ...current.messages,
          { role: "assistant", content: answer },
        ],
        steps: activity,
        updatedAt: Date.now(),
      }));

      setSessions((previous) =>
        [...previous].sort((a, b) => b.updatedAt - a.updatedAt)
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unexpected request error."
      );
    } finally {
      setLoading(false);
    }
  }

  async function importRepository(source: "github" | "zip" | "folder", selectedFiles?: FileList | null) {
    // Capture the source name before awaiting the upload; the input is reset immediately after this call.
    const selectedFileNames = Array.from(selectedFiles ?? []).map((file) => ({
      name: file.name,
      relativePath: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
    }));
    let sourceDisplayName = "";
    if (source === "github") {
      sourceDisplayName = repoImportUrl.trim().replace(/\.git$/i, "").split("/").filter(Boolean).pop() || "";
    } else if (source === "zip" && selectedFileNames[0]) {
      sourceDisplayName = selectedFileNames[0].name.replace(/\.zip$/i, "");
    } else if (source === "folder" && selectedFileNames[0]) {
      sourceDisplayName = selectedFileNames[0].relativePath.split(/[\\/]/).filter(Boolean)[0] || "";
    }

    setRepoImportBusy(true);
    setRepoImportError("");
    setRepoImportMessage("");
    try {
      let response: Response;
      if (source === "github") {
        if (!repoImportUrl.trim()) throw new Error("Enter a GitHub repository URL.");
        const body = new FormData();
        body.append("repository_url", repoImportUrl.trim());
        if (repoImportToken.trim()) body.append("github_token", repoImportToken.trim());
        response = await fetch(`${API_URL}/api/repository/import/github`, {
          method: "POST", credentials: "include", body,
        });
      } else {
        const chosen = Array.from(selectedFiles ?? []);
        if (!chosen.length) throw new Error("Select a ZIP file or a project folder.");
        const body = new FormData();
        for (const file of chosen) {
          body.append("files", file, file.name);
          body.append("relative_paths", source === "folder" ? (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name : file.name);
        }
        response = await fetch(`${API_URL}/api/repository/import/upload`, {
          method: "POST", credentials: "include", body,
        });
      }
      const data = await response.json() as { repo_path?: string; repository?: string; detail?: string };
      if (!response.ok || !data.repo_path) throw new Error(data.detail || `Repository import failed (${response.status}).`);
      setRepoPath(data.repo_path);

      // Never display the backend's temporary workspace ID as the repository name.
      const backendName = typeof data.repository === "string" ? data.repository.trim() : "";
      const safeBackendName = backendName && !/^[a-f0-9]{24,}$/i.test(backendName)
        ? backendName
        : "Imported project";
      setRepoName(sourceDisplayName || safeBackendName);
      setRepoSource(source === "github" ? "GitHub" : source === "zip" ? "ZIP upload" : "Folder upload");
      const listingResponse = await fetch(
        `${API_URL}/api/repository/files?repo_path=${encodeURIComponent(data.repo_path)}`,
        { credentials: "include" }
      );
      if (!listingResponse.ok) {
        const listingError = await listingResponse.json().catch(() => ({}));
        throw new Error(listingError.detail || "Repository imported, but its file list could not be loaded.");
      }
      const listing = await listingResponse.json() as { files?: string[]; file_count?: number };
      const importedFiles = Array.isArray(listing.files) ? listing.files : [];
      setRepoFiles(importedFiles);
      const firstEditableFile = importedFiles.find((file) => /\.(py|js|jsx|ts|tsx)$/i.test(file));
      setEditFilePath(firstEditableFile || "");
      setRepoImportMessage(`Repository ready · ${listing.file_count ?? importedFiles.length} files`);
      setRepoImportUrl("");
      setRepoImportToken("");
    } catch (err) {
      setRepoImportError(err instanceof Error ? err.message : "Unable to import repository.");
    } finally {
      setRepoImportBusy(false);
    }
  }

  async function runRepositoryScan(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!repoPath.trim()) {
      setScanError("Please enter your repository folder path.");
      return;
    }

    setScanLoading(true);
    setScanError("");
    setScanResult(null);

    try {
      const response = await fetch(`${API_URL}/api/repository/scan`, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_path: repoPath.trim(),
          max_files: scanMaxFiles,
        }),
      });

      const data: unknown = await response.json();
      const result = data as RepositoryScanResult & {
        detail?: unknown;
      };

      if (!response.ok) {
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : JSON.stringify(result.detail ?? `HTTP ${response.status}`)
        );
      }

      setScanResult(result);
      const sessionId = activeId;
      const reports = result.reports ?? [];
      const filesConsidered = result.files_considered ?? reports.length;
      const aiReviewsCompleted = result.files_scanned ?? reports.filter((report) => report.status === "completed" && Boolean(report.review)).length;
      const filesFailed = result.files_failed ?? reports.filter((report) => report.status === "error").length;
      const filesSkipped = result.summary?.files_skipped ?? reports.filter((report) => report.status === "skipped").length;
      const staticFindingCount = result.static_findings_count ?? result.summary?.static_findings_count ?? 0;
      const findingSummary = result.summary?.finding_summary ?? {};
      const aiFindingCount = findingSummary.ai_findings;
      const ruffFindingCount = findingSummary.ruff_findings;
      const deterministicFindingCount = findingSummary.deterministic_findings;

      const staticFindingLines = reports.flatMap((report) => {
        const groups = [report.static_analysis?.findings, report.language_validation?.findings];
        return groups.flatMap((items) => Array.isArray(items) ? items.map((item) => {
          if (!item || typeof item !== "object") return null;
          const finding = item as Record<string, unknown>;
          const title = String(finding.message ?? finding.title ?? finding.code ?? "Static-analysis finding");
          const line = typeof finding.line === "number" ? ` (line ${finding.line})` : "";
          return `- **${report.file}${line}:** ${title}`;
        }).filter((line): line is string => Boolean(line)) : []);
      });

      const reportLines = reports.map((report) => {
        const statusLabel = report.status === "completed" ? "Review completed" : report.status === "skipped" ? "Skipped" : "Review failed";
        const staticCount = (report.static_analysis?.count ?? 0) + (report.language_validation?.count ?? 0);
        const details = [`- **${report.file}** — ${statusLabel}; ${staticCount} static finding(s).`];
        if (report.error) details.push(`  - Review error: ${report.error}`);
        else if (report.reason) details.push(`  - Reason: ${report.reason}`);
        else if (report.review && report.status === "completed") details.push("  - AI review is available in the file report above.");
        return details.join("\n");
      });

      const scanSummary = [
        "## Repository analysis complete",
        `ARGUS analyzed **${filesConsidered} file(s)** in **${result.repository ?? repoPath.trim()}**. ${filesFailed ? `${filesFailed} file(s) could not be fully reviewed.` : "All selected files were processed without file-level failures."}`,
        "",
        "### Results at a glance",
        `- **AI reviews completed:** ${aiReviewsCompleted} of ${filesConsidered}`,
        `- **Static-analysis findings:** ${staticFindingCount}`,
        ...(typeof ruffFindingCount === "number" ? [`- **Ruff findings:** ${ruffFindingCount}`] : []),
        ...(typeof deterministicFindingCount === "number" ? [`- **Deterministic findings:** ${deterministicFindingCount}`] : []),
        ...(typeof aiFindingCount === "number" ? [`- **AI-reported findings:** ${aiFindingCount}`] : []),
        `- **Files needing review:** ${filesFailed}`,
        ...(filesSkipped ? [`- **Files skipped:** ${filesSkipped}`] : []),
        "",
        ...(staticFindingLines.length ? ["### Findings to review", ...staticFindingLines.slice(0, 10), ...(staticFindingLines.length > 10 ? [`- ${staticFindingLines.length - 10} additional finding(s) are listed in the dashboard.`] : [])] : ["### Findings to review", "No individual static findings were returned by the available checks."]),
        "",
        "### File-by-file outcome",
        ...reportLines,
        "",
        ...(typeof aiFindingCount === "number" && aiFindingCount === 0 ? ["The AI review completed without reporting additional AI-detected findings. Static-analysis findings listed above are separate and should still be reviewed."] : []),
        ...(filesFailed ? ["Some files could not be fully reviewed. Check their file reports for the reason before treating the scan as complete."] : []),
        "**Recommended next step:** Review the listed findings, make changes through your normal development workflow, then run the relevant tests and scan again to verify the result.",
        result.note ? `\n${result.note}` : "",
      ].filter(Boolean).join("\n");
      if (sessionId) {
        updateSession(sessionId, (current) => ({
          ...current,
          title: current.title === "New conversation" ? "Repository Scan" : current.title,
          messages: [
            ...current.messages,
            { role: "user", content: `Analyze repository for bugs and code-quality issues: ${repoPath.trim()}` },
            { role: "assistant", content: scanSummary },
          ],
          updatedAt: Date.now(),
        }));
        setSessions((previous) => [...previous].sort((a, b) => b.updatedAt - a.updatedAt));
      }
    } catch (err) {
      setScanError(
        err instanceof Error
          ? err.message
          : "Unexpected repository scan error."
      );
    } finally {
      setScanLoading(false);
    }
  }

  async function requestManualCodeEdit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!repoPath.trim()) { setFixError("Please enter your repository folder path."); return; }
    if (!editFilePath.trim() || !editInstruction.trim()) { setFixError("Choose a source file from the imported repository and describe the change you want."); return; }
    if (!/\.(py|js|jsx|ts|tsx)$/i.test(editFilePath.trim())) { setFixError("Choose a supported source file (.py, .js, .jsx, .ts or .tsx) from the imported repository."); return; }
    setFixLoading(true); setFixError(""); setFixMessage(""); setFixSuggestion(null);
    try {
      const response = await fetch(`${API_URL}/api/repository/fix/suggest`, {
        credentials: "include", method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_path: repoPath.trim(), file_path: editFilePath.trim(), finding_description: editInstruction.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? `HTTP ${response.status}`));
      setFixSuggestion(data as FixSuggestion);
    } catch (err) {
      setFixError(err instanceof Error ? err.message : "Unable to generate the proposed edit.");
    } finally { setFixLoading(false); }
  }

  async function runEngineeringAction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!repoPath.trim()) {
      setEngineeringError("Please enter your repository folder path.");
      return;
    }
    if (engineeringRunLockRef.current) return;
    engineeringRunLockRef.current = true;
    setEngineeringLoading(true);
    setEngineeringError("");
    try {
      const response = await fetch(`${API_URL}/api/repository/execute`, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_path: repoPath.trim(), action: engineeringAction }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? `HTTP ${response.status}`));
      }
      const sessionId = activeId;
      if (sessionId) {
        const output = [
          `## ${data.label ?? "Repository task"}`,
          `**Status:** ${data.timed_out ? "Timed out" : data.success ? "Passed" : `Failed (exit ${data.exit_code ?? "unknown"})`}`,
          `**Repository:** ${repoPath.trim()}`,
          "",
          "```text",
          String(data.output || data.error || "No output."),
          "```",
          data.note ?? "",
        ].filter(Boolean).join("\n");
        updateSession(sessionId, (current) => ({
          ...current,
          title: current.title === "New conversation" ? String(data.label ?? "Repository Task") : current.title,
          messages: [
            ...current.messages,
            { role: "user", content: `Run repository task: ${engineeringAction}` },
            { role: "assistant", content: output },
          ],
          updatedAt: Date.now(),
        }));
        setSessions((previous) => [...previous].sort((a, b) => b.updatedAt - a.updatedAt));
      }
    } catch (err) {
      setEngineeringError(err instanceof Error ? err.message : "Repository task failed.");
    } finally {
      engineeringRunLockRef.current = false;
      setEngineeringLoading(false);
    }
  }

  function extractReviewFindings(review: string): ScanFinding[] {
    const blocks = review.split(
      /(?=^\s*#{1,6}\s+(?:Critical|High|Medium|Low|Info(?:rmational)?)\s*:)/im
    );

    return blocks.flatMap((block) => {
      const heading = block.match(
        /^\s*#{1,6}\s+((?:Critical|High|Medium|Low|Info(?:rmational)?)\s*:\s*[^\n]+)/im
      );
      if (!heading) return [];

      // Only deterministic findings explicitly marked by the backend can
      // request a fix. AI-only findings and unverified evidence stay read-only.
      const deterministicVerification =
        /^\s*[-*]\s*Verification\s*:\s*deterministic_static_check\s*$/im.test(
          block
        );
      const hasUnverifiedEvidence = /\bUNVERIFIED\b|\bINVALID\s*:/i.test(
        block
      );

      if (!deterministicVerification || hasUnverifiedEvidence) {
        return [];
      }

      const lineMatch = block.match(
        /(?:Exact source line number|Source line|Line number)\s*:\s*(\d+)/i
      );
      const line = lineMatch ? Number(lineMatch[1]) : undefined;

      // A deterministic finding must identify a concrete source line.
      if (!line || !Number.isInteger(line) || line < 1) {
        return [];
      }

      const description = block.trim();

      return [{
        title: heading[1].trim(),
        description: description.slice(0, 3900),
        line,
        verified: true,
      }];
    });
  }

  async function suggestCodeFix(filePath: string, finding: ScanFinding) {
    if (!finding.verified) {
      setFixError(
        "Fix suggestions are available only for deterministic findings with verified source evidence."
      );
      return;
    }

    if (!repoPath.trim()) {
      setFixError("Please enter your repository folder path.");
      return;
    }

    const selectedFilePath = filePath.trim();
    const pathSegments = selectedFilePath.split(/[\\/]+/);
    if (
      !/\.(py|js|jsx|ts|tsx)$/i.test(selectedFilePath) ||
      selectedFilePath.includes("://") ||
      selectedFilePath.startsWith("/") ||
      /^[A-Za-z]:/.test(selectedFilePath) ||
      pathSegments.some((segment) => !segment || segment === "." || segment === "..")
    ) {
      setFixError("The selected finding does not point to a supported, repository-relative source file.");
      return;
    }

    setFixLoading(true);
    setFixError("");
    setFixMessage("");
    setFixSuggestion(null);

    try {
      const response = await fetch(`${API_URL}/api/repository/fix/suggest`, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo_path: repoPath.trim(),
          file_path: selectedFilePath,
          finding_description: finding.description,
          line: finding.line,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? `HTTP ${response.status}`)
        );
      }

      setFixSuggestion(data as FixSuggestion);
    } catch (err) {
      setFixError(getFixRequestError(err, "generate the fix suggestion"));
    } finally {
      setFixLoading(false);
    }
  }

  async function applyCodeFix() {
    if (!fixSuggestion) return;

    const approved = window.confirm(
      `Apply the proposed fix to ${fixSuggestion.file}? The original file will be replaced only after backend validation.`
    );
    if (!approved) return;

    setFixActionLoading(true);
    setFixError("");
    setFixMessage("");

    try {
      const response = await fetch(`${API_URL}/api/repository/fix/apply`, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ suggestion_id: fixSuggestion.suggestion_id }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? `HTTP ${response.status}`)
        );
      }

      setFixMessage(data.message || "Approved fix applied successfully.");
      setFixSuggestion(null);
    } catch (err) {
      setFixError(getFixRequestError(err, "apply the approved fix"));
    } finally {
      setFixActionLoading(false);
    }
  }

  async function rejectCodeFix() {
    if (!fixSuggestion) return;

    setFixActionLoading(true);
    setFixError("");
    setFixMessage("");

    try {
      const response = await fetch(`${API_URL}/api/repository/fix/reject`, {
        credentials: "include",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ suggestion_id: fixSuggestion.suggestion_id }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? `HTTP ${response.status}`)
        );
      }

      setFixMessage(data.message || "Fix suggestion rejected. Original file remains unchanged.");
      setFixSuggestion(null);
    } catch (err) {
      setFixError(getFixRequestError(err, "reject the fix suggestion"));
    } finally {
      setFixActionLoading(false);
    }
  }


  return (
    <div className={`app ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${mode === "chat" ? "ai-assistant-mode" : ""}`}>
      <aside className="sidebar">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {sidebarCollapsed ? "»" : "«"}
        </button>
        <div className="brand">
          <img
            className="brand-logo"
            src="/argus-agent-logo.png"
            alt="ARGUS AI engineering agent"
          />
          <h1>ARGUS</h1>
          <p>Where Code Meets Intelligence.</p>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={() => createChat()}
          disabled={loading}
        >
          <span aria-hidden="true">＋</span><strong>New chat</strong>
        </button>

        <div className="history-section">
          <div className="history-heading">
            RECENT CONVERSATIONS
          </div>

          {modeSessions.length === 0 ? (
            <p className="history-empty">
              Your conversations will appear here.
            </p>
          ) : (
            modeSessions.map((session) => (
              <div
                className={`history-item ${
                  session.id === activeId ? "active" : ""
                }`}
                key={session.id}
              >
                <button
                  type="button"
                  className="history-select"
                  onClick={() => selectSession(session)}
                  title={session.title}
                >
                  <span className="history-icon">◷</span>
                  <span>{session.title}</span>
                </button>

                <button
                  type="button"
                  className="history-delete"
                  aria-label={`Delete ${session.title}`}
                  onClick={() => deleteSession(session.id)}
                  disabled={loading && session.id === activeId}
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-settings">
          <div className="field-label workspace-label">WORKSPACE MODE</div>
          <div className="workspace-mode-control">
            <button type="button" className="workspace-mode-trigger" aria-haspopup="menu" aria-expanded={workspaceMenuOpen} disabled={loading} onClick={() => setWorkspaceMenuOpen((open) => !open)}>
              <span className="workspace-mode-icon" aria-hidden="true">{mode === "agent" ? "⌘" : "✳"}</span>
              <span className="workspace-mode-copy">
                <strong>{mode === "agent" ? "Repository Agent" : "AI Assistant"}</strong>
              </span>
              <span className="workspace-mode-chevron" aria-hidden="true">⌄</span>
            </button>
            {workspaceMenuOpen && (
              <div className="workspace-mode-menu" role="menu">
                <button type="button" role="menuitem" className={`workspace-mode-option ${mode === "agent" ? "selected" : ""}`} onClick={() => { switchMode("agent"); setWorkspaceMenuOpen(false); }}>
                  <span className="workspace-mode-icon" aria-hidden="true">⌘</span>
                  <span><strong>Repository Agent</strong></span>
                  {mode === "agent" && <span className="mode-check">✓</span>}
                </button>
                <button type="button" role="menuitem" className={`workspace-mode-option ${mode === "chat" ? "selected" : ""}`} onClick={() => { switchMode("chat"); setWorkspaceMenuOpen(false); }}>
                  <span className="workspace-mode-icon" aria-hidden="true">✳</span>
                  <span><strong>AI Assistant</strong></span>
                  {mode === "chat" && <span className="mode-check">✓</span>}
                </button>
              </div>
            )}
          </div>

          {mode === "agent" && (
            <>
              <label htmlFor="repo-url" className="field-label">GITHUB REPOSITORY URL</label>
              <input
                id="repo-url"
                name="githubRepoUrl"
                type="url"
                className="field"
                value={repoImportUrl}
                onChange={(event) => setRepoImportUrl(event.target.value)}
                placeholder="https://github.com/owner/repository"
                autoComplete="off"
                disabled={repoImportBusy}
              />
              <input
                name="githubPrivateToken"
                className="field"
                type="password"
                value={repoImportToken}
                onChange={(event) => setRepoImportToken(event.target.value)}
                placeholder="Private repo token (optional)"
                autoComplete="new-password"
                disabled={repoImportBusy}
              />
              <button type="button" className="repo-import-button" disabled={repoImportBusy || !repoImportUrl.trim()} onClick={() => void importRepository("github")}>
                {repoImportBusy ? "Importing…" : "Import GitHub Repository"}
              </button>
              <div className="repo-upload-actions">
                <button type="button" className="repo-import-button secondary" disabled={repoImportBusy} onClick={() => zipInputRef.current?.click()}>Upload ZIP</button>
                <button type="button" className="repo-import-button secondary" disabled={repoImportBusy} onClick={() => folderInputRef.current?.click()}>Upload Folder</button>
              </div>
              <input ref={zipInputRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => { const selected = event.target.files; if (selected?.length) void importRepository("zip", selected); event.target.value = ""; }} />
              <input ref={folderInputRef} type="file" multiple hidden onChange={(event) => { const selected = event.target.files; if (selected?.length) void importRepository("folder", selected); event.target.value = ""; }} />
              {repoImportBusy && <p className="sidebar-help">Importing repository. Please keep this page open.</p>}
              {repoImportMessage && <p className="repo-import-success" role="status">{repoImportMessage}</p>}
              {repoImportError && <ErrorAlert title="Repository import failed" message={repoImportError} onClose={() => setRepoImportError("")} />}
              <section className="active-repository" aria-label="Active repository">
                <div className="active-repository-heading">
                  <span className="field-label">ACTIVE REPOSITORY</span>
                  {repoPath && <span className="repository-status"><i /> Ready</span>}
                </div>
                {repoPath ? (
                  <>
                    <div className="active-repository-name" title={repoName}>{repoName}</div>
                    <div className="active-repository-meta">
                      <span>{repoSource || "Imported project"}</span>
                      <span>{repoFiles.length} files</span>
                    </div>
                    <details className="repository-file-browser">
                      <summary>Browse project files</summary>
                      <div className="repository-file-list">
                        {repoFiles.length ? repoFiles.slice(0, 250).map((file) => (
                          <div className="repository-file-item" key={file} title={file}>
                            <span aria-hidden="true">{file.includes("/") ? "▱" : "▤"}</span>
                            <span>{file}</span>
                          </div>
                        )) : <p className="sidebar-help">No files found in this repository.</p>}
                        {repoFiles.length > 250 && <p className="sidebar-help">Showing the first 250 of {repoFiles.length} files.</p>}
                      </div>
                    </details>
                  </>
                ) : (
                  <p className="repository-empty">No repository selected. Import a GitHub repository, ZIP, or folder to get started.</p>
                )}
              </section>
            </>
          )}
        </div>

        <div className="sidebar-bottom">
          {isGuest ? (
            <button type="button" className="profile-trigger guest-signin-trigger" onClick={onSignIn}>
              <span className="profile-avatar">↗</span>
              <span className="profile-trigger-copy"><strong>Sign In</strong><small>Access your ARGUS workspace</small></span>
            </button>
          ) : (
            <div className="profile-menu-wrap">
              {profileMenuOpen && (
                <div className="profile-menu" role="menu">
                  <div className="profile-menu-heading"><span className="profile-avatar">{user.name.trim().charAt(0).toUpperCase()}</span><span><strong>{user.name}</strong></span></div>
                  <div className="profile-menu-divider" />
                  <button type="button" role="menuitem" onClick={() => { setProfileMenuOpen(false); setAccountPanel("profile"); }}><span>♙</span> My profile</button>
                  <button type="button" role="menuitem" onClick={() => { setProfileMenuOpen(false); setAccountPanel("preferences"); }}><span>⚙</span> Preferences</button>
                  <button type="button" role="menuitem" onClick={() => { setProfileMenuOpen(false); onSignOut(); }}><span>↗</span> Sign out</button>
                </div>
              )}
              <button type="button" className="profile-trigger" aria-haspopup="menu" aria-expanded={profileMenuOpen} onClick={() => setProfileMenuOpen((open) => !open)}>
                <span className="profile-avatar">{user.name.trim().charAt(0).toUpperCase()}</span>
                <span className="profile-trigger-copy"><strong>{user.name}</strong></span>
                <span className="profile-trigger-more">•••</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">ARGUS · AI SOFTWARE ENGINEERING AGENT</p>
            <h2>
              {mode === "agent"
                ? "Repository Engineering Agent"
                : "AI Assistant"}
            </h2>
            <p className="topbar-subtitle">
              {mode === "agent" ? "Analyze, understand and improve your codebase." : "Where Code Meets Intelligence."}
            </p>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              flexWrap: "wrap",
            }}
          >
            {mode === "agent" && (
              <button
                type="button"
                className="send-button"
                onClick={() => {
                  if (scanLoading || engineeringLoading || fixLoading) return;
                  const next = !scanOpen;
                  setScanOpen(next);
                  setCodeEditorOpen(false);
                  setEngineeringOpen(false);
                  setScanError("");
                  setEngineeringError("");
                }}
              >
                {scanOpen ? "Close Repository Scan" : "Scan Repository"}
              </button>
            )}

            {mode === "agent" && (
              <button
                type="button"
                className="send-button"
                aria-expanded={codeEditorOpen}
                onClick={() => {
                  if (scanLoading || engineeringLoading || fixLoading) return;
                  const next = !codeEditorOpen;
                  setCodeEditorOpen(next);
                  setEngineeringOpen(false);
                  setScanOpen(false);
                  setFixError("");
                  setFixMessage("");
                }}
              >
                {codeEditorOpen ? "Close AI Code Editor" : "AI Code Editor"}
              </button>
            )}

            {mode === "agent" && (
              <button
                type="button"
                className="send-button"
                aria-expanded={engineeringOpen}
                onClick={() => {
                  if (scanLoading || engineeringLoading || fixLoading) return;
                  const next = !engineeringOpen;
                  setEngineeringOpen(next);
                  setCodeEditorOpen(false);
                  setScanOpen(false);
                  setEngineeringError("");
                  setScanError("");
                }}
              >
                {engineeringOpen ? "Close Run Tests" : "Run Tests"}
              </button>
            )}

          </div>
        </header>

        <div className="workspace-scroll">
        {codeEditorOpen && mode === "agent" && (
          <section className="engineering-tools-shell code-editor-workspace" aria-label="AI Code Editor">
            <div className="engineering-tools-heading">
              <h3>AI Code Editor</h3>
              <p>Describe a code change, inspect the proposed diff, and approve it before your original file is modified.</p>
            </div>

            <section className="engineering-tool-card code-edit-card" aria-label="AI Code Editing">
              <form onSubmit={requestManualCodeEdit} className="code-editor-form">
                <select
                  aria-label="Choose source file to edit"
                  value={editFilePath}
                  onChange={(event) => setEditFilePath(event.target.value)}
                  disabled={fixLoading || !editableRepositoryFiles.length}
                  className="code-editor-file-input"
                >
                  <option value="">
                    {editableRepositoryFiles.length
                      ? "Choose a source file from the imported repository"
                      : "Import a repository containing .py, .js, .jsx, .ts or .tsx files"}
                  </option>
                  {editableRepositoryFiles.map((file) => (
                    <option value={file} key={file}>{file}</option>
                  ))}
                </select>

                <div className="code-editor-composer">
                  <div className="glow-wrapper">
                    <div className="input-box">
                      <textarea
                        className="prompt-area"
                        aria-label="Describe code change"
                        value={editInstruction}
                        onChange={(event) => setEditInstruction(event.target.value)}
                        placeholder="Describe the bug or code improvement you want ARGUS to implement..."
                        rows={2}
                        disabled={fixLoading}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && enterToSend && !event.shiftKey && !event.nativeEvent.isComposing) {
                            event.preventDefault();
                            event.currentTarget.form?.requestSubmit();
                          }
                        }}
                      />
                      <div className="actions">
                        <span className="code-editor-hint">Enter to generate · Shift + Enter for a new line</span>
                        <button
                          type="submit"
                          className="send-button code-editor-submit"
                          disabled={fixLoading || !repoPath.trim() || !editFilePath.trim() || !editInstruction.trim()}
                        >
                          {fixLoading ? "Generating fix…" : "Generate Code Fix ↑"}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </form>

              {fixError && <ErrorAlert title="Code edit failed" message={fixError} onClose={() => setFixError("")} />}
              {fixMessage && <p role="status" className="code-editor-success">{fixMessage}</p>}

              {fixSuggestion && (
                <section className="code-editor-preview" aria-label="AI code edit preview">
                  <h4>Proposed edit: {fixSuggestion.file}</h4>
                  <p>{fixSuggestion.explanation}</p>
                  <p><strong>Validation:</strong> {fixSuggestion.validation.message}</p>
                  <pre>{fixSuggestion.diff}</pre>
                  <p className="code-editor-preview-note">The original file has not been modified. Review the diff before approving.</p>
                  <div className="code-editor-preview-actions">
                    <button type="button" className="send-button" onClick={applyCodeFix} disabled={fixActionLoading || !fixSuggestion.validation.valid}>
                      {fixActionLoading ? "Applying…" : "Approve & Apply Fix"}
                    </button>
                    <button type="button" className="send-button" onClick={rejectCodeFix} disabled={fixActionLoading}>Reject Fix</button>
                  </div>
                </section>
              )}
            </section>
          </section>
        )}

        {engineeringOpen && mode === "agent" && (
          <section className="engineering-tools-shell validate-workspace" aria-label="Run and Validate">
            <div className="engineering-tools-heading">
              <h3>Run &amp; Validate</h3>
              <p>Run predefined repository checks. Arbitrary shell commands are not accepted.</p>
            </div>
            <section className="engineering-tool-card validate-card">
              <div className="engineering-card-heading">
                <h4>Repository Execution</h4>
                <p>Choose a supported validation task for the selected repository.</p>
              </div>
              <form onSubmit={runEngineeringAction} className="engineering-run-form">
                <select aria-label="Repository task" value={engineeringAction} onChange={(event) => setEngineeringAction(event.target.value)} disabled={engineeringLoading}>
                  <option value="run_tests">Run automated tests</option>
                  <option value="run_build">Run project build / syntax check</option>
                  <option value="run_lint">Run lint checks</option>
                  <option value="git_status">Check Git status</option>
                </select>
                <button type="submit" className="send-button" disabled={engineeringLoading || !repoPath.trim()}>
                  {engineeringLoading ? "Running…" : "Execute task"}
                </button>
              </form>
              {engineeringError && <p role="alert" className="engineering-error">{engineeringError}</p>}
            </section>
          </section>
        )}

        {scanOpen && mode === "agent" && (
          <section
            className="repository-scan-panel"
            aria-label="Repository-wide scan"
            style={{
              margin: "16px 24px 0",
              padding: "18px",
              border: "1px solid var(--border, #303642)",
              borderRadius: "12px",
              background: "var(--panel, #11151d)",
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: "8px" }}>
              Repository-Wide AI Bug Detection
            </h3>

            <p className="repository-scan-description">
              Identify bugs, reliability risks, and code-quality issues across your codebase with deterministic checks and AI-assisted review. ARGUS analyzes your source files and reports actionable findings without changing your code.
            </p>

            <form
              onSubmit={runRepositoryScan}
              style={{
                display: "flex",
                gap: "10px",
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              <label
                htmlFor="scan-max-files"
                style={{ fontSize: "13px" }}
              >
                Files to analyze
              </label>

              <select
                id="scan-max-files"
                value={scanMaxFiles}
                onChange={(event) =>
                  setScanMaxFiles(Number(event.target.value))
                }
                disabled={scanLoading}
                className="field"
                style={{ width: "110px" }}
              >
                <option value={3}>3 files</option>
                <option value={5}>5 files</option>
                <option value={10}>10 files</option>
                <option value={15}>15 files</option>
                <option value={25}>25 files</option>
              </select>

              <button
                type="submit"
                className="send-button"
                disabled={scanLoading || !repoPath.trim()}
              >
                {scanLoading ? "Scanning..." : "Start Repository Scan"}
              </button>
            </form>

            {scanError && <ErrorAlert title="Scan failed" message={scanError} onClose={() => setScanError("")} style={{ marginTop: "12px" }} />}

            {scanLoading && (
              <div role="status" style={{ marginTop: "14px" }}>
                <span className="status-dot" /> ARGUS is analyzing the selected files with static checks and AI-assisted review. This may take a few moments.
              </div>
            )}

            {scanResult && (
              <div style={{ marginTop: "18px" }}>
                <div
                  className="scan-metrics-grid"
                  style={{ gap: "10px", marginBottom: "14px" }}
                >
                  {[
                    {
                      label: "AI reviews completed",
                      value: scanResult.files_scanned ?? 0,
                    },
                    {
                      label: "Files considered",
                      value:
                        scanResult.files_considered ??
                        scanResult.reports?.length ??
                        0,
                    },
                    {
                      label: "Static findings",
                      value: scanResult.static_findings_count ?? 0,
                    },
                    {
                      label: "Files failed",
                      value: scanResult.files_failed ?? 0,
                    },
                    {
                      label: "Large files skipped",
                      value: scanResult.files_skipped_large ?? 0,
                    },
                    {
                      label: "Empty files skipped",
                      value:
                        scanResult.summary?.files_skipped ??
                        scanResult.reports?.filter(
                          (report) => report.status === "skipped"
                        ).length ??
                        0,
                    },
                    {
                      label: "TypeScript findings",
                      value:
                        scanResult.summary?.typescript_findings_count ??
                        scanResult.typescript_validation?.count ??
                        0,
                    },
                    {
                      label: "Ruff findings",
                      value:
                        scanResult.summary?.finding_summary?.ruff_findings ??
                        0,
                    },
                    {
                      label: "Deterministic findings",
                      value:
                        scanResult.summary?.finding_summary
                          ?.deterministic_findings ?? 0,
                    },
                    {
                      label: "AI findings",
                      value:
                        scanResult.summary?.finding_summary?.ai_findings ??
                        0,
                    },
                    {
                      label: "Unverified findings",
                      value:
                        scanResult.summary?.finding_summary
                          ?.unverified_findings ?? 0,
                    },
                  ].map((metric) => (
                    <div
                      className="scan-metric-card"
                      data-metric={metric.label}
                      key={metric.label}
                      style={{
                        padding: "12px",
                        border: "1px solid var(--border, #303642)",
                        borderRadius: "8px",
                      }}
                    >
                      <div style={{ fontSize: "12px", opacity: 0.75 }}>
                        {metric.label}
                      </div>
                      <strong
                        style={{
                          display: "block",
                          fontSize: "22px",
                          marginTop: "4px",
                        }}
                      >
                        {metric.value}
                      </strong>
                    </div>
                  ))}
                </div>

                {scanResult.message && (
                  <p style={{ opacity: 0.8 }}>{scanResult.message}</p>
                )}

                {scanResult.note && (
                  <p style={{ fontSize: "12px", opacity: 0.7 }}>
                    {scanResult.note}
                  </p>
                )}

                {scanResult.reports &&
                  scanResult.reports.length > 0 && (
                    <div style={{ display: "grid", gap: "12px" }}>
                      {scanResult.reports.map((report, index) => (
                        <details
                          className="scan-report-card"
                          data-status={report.status}
                          data-severity={/critical|high\s*:/i.test(report.review ?? "") ? "high" : /potential|warning|medium\s*:/i.test(report.review ?? "") ? "warning" : "normal"}
                          key={`${report.file}-${index}`}
                          open={index === 0}
                          style={{
                            border: "1px solid var(--border, #303642)",
                            borderRadius: "10px",
                            padding: "12px",
                          }}
                        >
                          <summary
                            style={{
                              cursor: "pointer",
                              display: "flex",
                              gap: "8px",
                              alignItems: "center",
                              flexWrap: "wrap",
                            }}
                          >
                            <strong
                              style={{ overflowWrap: "anywhere" }}
                            >
                              {report.file}
                            </strong>

                            <span
                              style={{
                                fontSize: "11px",
                                padding: "3px 7px",
                                borderRadius: "999px",
                                background:
                                  report.status === "completed"
                                    ? "rgba(34, 197, 94, 0.13)"
                                    : report.status === "error"
                                      ? "rgba(239, 68, 68, 0.13)"
                                      : "rgba(148, 163, 184, 0.13)",
                              }}
                            >
                              {report.status}
                            </span>

                            {typeof report.static_analysis?.count ===
                              "number" && (
                              <span
                                style={{
                                  fontSize: "12px",
                                  opacity: 0.75,
                                }}
                              >
                                Ruff findings: {report.static_analysis.count}
                              </span>
                            )}
                          </summary>

                          {report.error && <ErrorAlert title="File scan failed" message={report.error} style={{ marginTop: "12px" }} />}

                          {report.reason && (
                            <p style={{ marginTop: "12px" }}>
                              {report.reason}
                            </p>
                          )}

                          {report.review && (
                            <div style={{ marginTop: "12px" }}>
                              <MarkdownContent content={report.review} />

                              {report.status === "completed" &&
                                report.file.toLowerCase().endsWith(".py") &&
                                extractReviewFindings(report.review).length > 0 && (
                                  <div style={{ marginTop: "14px", display: "grid", gap: "8px" }}>
                                    <strong style={{ fontSize: "13px" }}>Verified deterministic fix suggestions</strong>
                                    {extractReviewFindings(report.review).map((finding, findingIndex) => (
                                      <div
                                        key={`${report.file}-${findingIndex}`}
                                        style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}
                                      >
                                        <span style={{ fontSize: "12px", opacity: 0.8, flex: "1 1 220px" }}>
                                          {finding.title}{finding.line ? ` · Line ${finding.line}` : ""}
                                        </span>
                                        <button
                                          type="button"
                                          className="send-button"
                                          disabled={fixLoading || fixActionLoading}
                                          onClick={() => suggestCodeFix(report.file, finding)}
                                        >
                                          {fixLoading ? "Generating fix..." : "Suggest AI Fix"}
                                        </button>
                                      </div>
                                    ))}
                                  </div>
                                )}
                            </div>
                          )}

                          {Array.isArray(
                            report.static_analysis?.findings
                          ) &&
                            report.static_analysis.findings.length > 0 && (
                              <details style={{ marginTop: "12px" }}>
                                <summary>
                                  View static-analysis details
                                </summary>
                                <pre
                                  style={{
                                    overflowX: "auto",
                                    whiteSpace: "pre-wrap",
                                    overflowWrap: "anywhere",
                                    fontSize: "12px",
                                  }}
                                >
                                  {JSON.stringify(
                                    report.static_analysis.findings,
                                    null,
                                    2
                                  )}
                                </pre>
                              </details>
                            )}
                        </details>
                      ))}
                    </div>
                  )}

                {fixError && <ErrorAlert title="Fix operation failed" message={fixError} onClose={() => setFixError("")} style={{ marginTop: "14px" }} />}

                {fixMessage && (
                  <p role="status" style={{ marginTop: "14px", color: "#22c55e" }}>
                    {fixMessage}
                  </p>
                )}

                {fixSuggestion && (
                  <section
                    aria-label="AI fix preview"
                    style={{ marginTop: "18px", padding: "16px", border: "1px solid var(--border, #303642)", borderRadius: "10px" }}
                  >
                    <h3 style={{ marginTop: 0 }}>AI Fix Preview</h3>
                    <p><strong>File:</strong> {fixSuggestion.file}</p>
                    <p>{fixSuggestion.explanation}</p>
                    <p style={{ fontSize: "13px" }}>
                      <strong>Validation:</strong> {fixSuggestion.validation.message}
                    </p>
                    <div style={{ marginTop: "12px" }}>
                      <strong>Proposed changes</strong>
                      <pre style={{ marginTop: "8px", padding: "14px", borderRadius: "8px", background: "#090b10", overflowX: "auto", whiteSpace: "pre", fontSize: "12px", lineHeight: 1.6 }}>
                        {fixSuggestion.diff || "No diff was returned."}
                      </pre>
                    </div>
                    <p style={{ fontSize: "12px", opacity: 0.75 }}>
                      The original file has not been modified. Review the diff before approving.
                    </p>
                    <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="send-button"
                        onClick={applyCodeFix}
                        disabled={fixActionLoading || !fixSuggestion.validation.valid}
                      >
                        {fixActionLoading ? "Applying..." : "Approve & Apply Fix"}
                      </button>
                      <button
                        type="button"
                        className="send-button"
                        onClick={rejectCodeFix}
                        disabled={fixActionLoading}
                      >
                        Reject Fix
                      </button>
                    </div>
                  </section>
                )}
              </div>
            )}
          </section>
        )}

        <section className="conversation">
          {messages.length === 0 && !input.trim() && !scanOpen && !scanResult && !scanLoading ? (
            <div className="welcome welcome-minimal" aria-live="polite">
              {isGuest ? (
                <h2>Hi, welcome to ARGUS.</h2>
              ) : (
                <h2>Hi, <span>{user.name.trim().split(/\s+/)[0] || "there"}.</span></h2>
              )}
              <p>{isGuest ? "Ask anything about your repository, find issues, get improvements and build better software with AI." : "What are we building today?"}</p>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article
                  className={`message ${
                    message.role === "user"
                      ? "user-message"
                      : "assistant-message"
                  }`}
                  key={`${activeId}-${index}`}
                >
                  <div className="message-role">
                    {message.role === "user" ? "YOU" : "ARGUS"}
                  </div>

                  {message.role === "assistant" ? (
                    <MarkdownContent content={message.content} />
                  ) : (
                    <div className="user-content">
                      {message.content}
                    </div>
                  )}
                </article>
              ))}

              {loading && (
                <div className="loading-indicator">
                  <span className="status-dot" />
                  ARGUS is working on your request…
                </div>
              )}

              {steps.length > 0 && (
                <details className="activity-panel">
                  <summary>
                    <span>⌘</span> Agent activity ({steps.length} events)
                  </summary>
                  <ol>
                    {steps.map((step, index) => (
                      <li key={`${index}-${step}`}>{step}</li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          )}

          {error && <ErrorAlert title="Request failed" message={error} onClose={() => setError("")} />}
        </section>
        </div>

        {!(mode === "agent" && (engineeringOpen || codeEditorOpen || scanOpen)) && <footer className="composer-area">
          <form className="composer" onSubmit={sendMessage}>
            <div className="glow-wrapper">
              <div className="input-box">
                      {attachments.length > 0 && <div className="attachment-chips">{attachments.map((file, index) => <span className="attachment-chip" key={`${file.name}-${file.lastModified}`}>{file.name}<button type="button" aria-label={`Remove ${file.name}`} onClick={() => setAttachments((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></span>)}</div>}
                <textarea
                  ref={promptRef}
                  className="prompt-area"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={mode === "agent" ? "Ask ARGUS about your repository..." : "Ask your coding question..."}
                  rows={1}
                  disabled={loading}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && enterToSend && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      event.currentTarget.form?.requestSubmit();
                    }
                  }}
                />
                <div className="actions">
                  <div className="composer-actions-right">
                    <input ref={attachmentInputRef} type="file" multiple hidden accept=".txt,.md,.json,.csv,.log,.py,.js,.jsx,.ts,.tsx,.html,.css,.java,.c,.cpp,.h,.sql,.yaml,.yml,.xml,.toml,.ini,.sh,.bat,.ps1,.pdf,.png,.jpg,.jpeg,.webp" onChange={handleAttachmentSelection} />
                    <button type="button" className="attach-button" onClick={() => attachmentInputRef.current?.click()} title="Attach files (PDF, images, text or code)" aria-label="Attach files">📎</button>
                    <button type="button" className="voice-button" onClick={toggleVoiceInput} aria-label={`Voice input for ${mode === "agent" ? "Repository Agent" : "AI Assistant"}`} title={isListening ? "Stop voice input" : "Start voice input"}>
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M12 3v10" /><path d="M8.5 6.5v6a3.5 3.5 0 0 0 7 0v-6" /><path d="M5 11.5v1a7 7 0 0 0 14 0v-1" /><path d="M12 19.5V22" /><path d="M8.5 22h7" />
                      </svg>
                    </button>
                    <button type="submit" className="submit-btn" aria-label="Send message" title={loading ? "Working..." : "Send message"} disabled={loading || (!input.trim() && attachments.length === 0)}>
                      {loading ? <span className="submit-loading">…</span> : <svg viewBox="0 0 24 24" aria-hidden="true"><line x1="12" y1="19" x2="12" y2="5" /><polyline points="5 12 12 5 19 12" /></svg>}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </form>
          {voiceNotice && <p className="voice-notice" role="status">{voiceNotice}</p>}
        </footer>}
      </main>
      {!isGuest && accountPanel && (
        <div className="argus-settings-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setAccountPanel(null); }}>
          <section className="argus-settings-panel" role="dialog" aria-modal="true" aria-labelledby="argus-settings-title">
            <header><div><p>ARGUS ACCOUNT</p><h2 id="argus-settings-title">{accountPanel === "profile" ? "Your profile" : "Preferences"}</h2></div><button type="button" aria-label="Close" onClick={() => setAccountPanel(null)}>×</button></header>
            {accountPanel === "profile" ? (
              <div className="argus-profile-details"><div className="argus-profile-large-avatar">{user.name.trim().charAt(0).toUpperCase()}</div><div><strong>{user.name}</strong><small>{user.email}</small></div><p>Your ARGUS account is active. Your conversations are stored in your account and kept separate from other users.</p><div className="argus-account-sync-state"><span className={syncStatus === "error" ? "is-error" : ""}/>{syncStatus === "synced" ? "Conversation sync is active" : syncStatus === "syncing" ? "Connecting account history…" : "Sync needs attention — local history is preserved"}{syncStatus === "error" && <button type="button" onClick={() => { setError(""); setSyncStatus("syncing"); setAccountSyncReady(false); setSyncAttempt((value) => value + 1); }}>Retry</button>}</div></div>
            ) : (
              <div className="argus-preference-list"><label><span><strong>Enter to send</strong><small>Press Enter to send; Shift + Enter creates a new line.</small></span><input type="checkbox" checked={enterToSend} onChange={(event) => { setEnterToSend(event.target.checked); localStorage.setItem(`argus.enterToSend.${user.id}`, String(event.target.checked)); }}/></label><label><span><strong>Default workspace</strong><small>Choose which workspace opens for a new session.</small></span><select value={localStorage.getItem(`argus.defaultMode.${user.id}`) || "agent"} onChange={(event) => { localStorage.setItem(`argus.defaultMode.${user.id}`, event.target.value); setMode(event.target.value as Mode); }}><option value="agent">Repository Agent</option><option value="chat">AI Assistant</option></select></label><p>Preferences are saved locally for this account on this device.</p></div>
            )}
            <footer><button type="button" onClick={() => setAccountPanel(null)}>Done</button></footer>
          </section>
        </div>
      )}
    </div>
  );
}

type ArgusUser = { id: string; name: string; email: string; created_at?: string };

export default function App() {
  const [user, setUser] = useState<ArgusUser | null>(null);
  const [showAuth, setShowAuth] = useState(false);
  const [checking, setChecking] = useState(true);
  const [view, setView] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [authError, setAuthError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/api/auth/me`, { credentials: "include" })
      .then(async (response) => response.ok ? (await response.json()).user as ArgusUser : null)
      .then((account) => {
        setUser(account);
      })
      .catch(() => setUser(null))
      .finally(() => setChecking(false));
  }, []);

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setAuthError(""); setBusy(true);
    try {
      if (view === "register" && password !== confirmPassword) {
        throw new Error("Your passwords do not match.");
      }
      const endpoint = view === "register" ? "/api/auth/register" : "/api/auth/login";
      const body = view === "register" ? { name, email, password } : { email, password };
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to authenticate.");
      const account = data.user as ArgusUser;
      const scopedKey = `argus.sessions.${account.id}`;
      if (!localStorage.getItem(scopedKey) && !localStorage.getItem("argus.legacy-migrated.global")) {
        const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
        if (legacy) localStorage.setItem(scopedKey, legacy);
        localStorage.setItem("argus.legacy-migrated.global", "1");
      }
      setUser(account);
      setShowAuth(false);
    } catch (error) { setAuthError(error instanceof Error ? error.message : "Something went wrong."); }
    finally { setBusy(false); }
  }

  async function signOut() {
    try { await fetch(`${API_URL}/api/auth/logout`, { method: "POST", credentials: "include" }); }
    finally { setUser(null); setPassword(""); setConfirmPassword(""); setView("login"); setShowAuth(false); }
  }

  if (checking) return <div className="argus-auth-loading"><span className="argus-auth-orb" />Loading ARGUS…</div>;
  if (user || !showAuth) {
    const workspaceUser = user ?? { id: "guest", name: "Guest", email: "Sign in to save your workspace" };
    return <div className="argus-account-shell"><Workspace key={user?.id ?? "guest"} user={workspaceUser} onSignOut={signOut} onSignIn={() => setShowAuth(true)} isGuest={!user} /></div>;
  }
  return <div className="argus-auth-screen">
    <div className="argus-auth-panel">
      <div className="argus-auth-brand"><div className="argus-auth-mark">A</div><div><strong>ARGUS</strong><span>Where Code Meets Intelligence.</span></div></div>
      <div className="argus-auth-heading"><p>YOUR PRIVATE AI WORKSPACE</p><h1>{view === "login" ? "Welcome back." : "Create your account."}</h1><span>{view === "login" ? "Sign in to continue where your code left off." : "Set up your personal engineering workspace."}</span></div>
      {view === "register" && <button type="button" className="argus-register-back" onClick={() => { setView("login"); setAuthError(""); }}>← Back to Sign In</button>}
      <form className="argus-auth-form" onSubmit={submitAuth}>
        {view === "register" && <label>Full name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" autoComplete="name" minLength={2} required /></label>}
        <label>Email address<input type="email" value={email} onChange={(e) => { setEmail(e.target.value); setAuthError(""); }} placeholder="you@example.com" autoComplete="email" required /></label>
        <label>Password<div className="argus-password-wrap"><input type={showPassword ? "text" : "password"} value={password} onChange={(e) => { setPassword(e.target.value); setAuthError(""); }} placeholder={view === "register" ? "At least 8 characters" : "Enter your password"} autoComplete={view === "register" ? "new-password" : "current-password"} minLength={view === "register" ? 8 : 1} required /><button type="button" onClick={() => setShowPassword(!showPassword)}>{showPassword ? "Hide" : "Show"}</button></div></label>
        {view === "register" && <label>Confirm password<input type={showPassword ? "text" : "password"} value={confirmPassword} onChange={(e) => { setConfirmPassword(e.target.value); setAuthError(""); }} placeholder="Re-enter your password" autoComplete="new-password" minLength={8} required /></label>}
        {authError && <ErrorAlert title="Sign-in failed" message={authError} onClose={() => setAuthError("")} />}
        <button className="argus-auth-submit" type="submit" disabled={busy}>{busy ? "Please wait…" : view === "login" ? "Sign in" : "Create account"}<span>→</span></button>
      </form>
      <div className="argus-auth-switch">{view === "login" ? "New to ARGUS?" : "Already have an account?"}<button type="button" onClick={() => { setView(view === "login" ? "register" : "login"); setAuthError(""); }}> {view === "login" ? "Create an account" : "Back to Login"}</button></div>
      <button type="button" className="argus-back-workspace" onClick={() => { setShowAuth(false); setAuthError(""); }}>← Back to Workspace</button>
      <div className="argus-auth-foot"><span>ARGUS  •  2026</span></div>
    </div><div className="argus-auth-side"><div className="argus-auth-grid"/><div className="argus-auth-side-copy"><span>ENGINEER WITH CLARITY.</span><h2>Your codebase.<br/>Your intelligence.<br/><em>Your workspace.</em></h2><p>A focused environment for understanding, reviewing and improving software — powered by Gemini AI.</p></div></div>
  </div>;
}
