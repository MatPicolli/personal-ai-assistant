#!/usr/bin/env python3
"""Small local AI assistant powered by Ollama.

The assistant uses a simple JSON tool protocol so it works with models that do
not expose native tool-calling reliably through every Ollama build.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import html
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

try:
    import readline  # noqa: F401  # Enables GNU readline editing for input().
except ImportError:
    readline = None


DEFAULT_MODEL = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_CWD = "/home/mateus" if os.path.isdir("/home/mateus") else os.path.expanduser("~")
MAX_TOOL_ROUNDS = 8
CHAT_DIR = os.path.join(DEFAULT_CWD, ".local", "share", "picoassistant", "chats")
CONTEXT_FILES = [
    os.path.join(DEFAULT_CWD, "README.md"),
    os.path.join(DEFAULT_CWD, "AGENTS.md"),
    os.path.join(DEFAULT_CWD, "CLAUDE.md"),
]
CONTEXT_MAX_CHARS = 12000
EXTENDED_THINKING_PROMPT = """

Public extended thinking mode is enabled. For every tool call and final answer, include a `thought` field with one concise sentence explaining the visible next step or key evidence. Do not reveal private chain-of-thought, hidden scratch work, or step-by-step internal reasoning.
"""
SAFE_TERMINAL_COMMANDS = {
    "awk", "cat", "du", "echo", "file", "find", "git", "grep", "head", "id",
    "ls", "pgrep", "pwd", "python3", "rg", "sed", "stat", "tail", "tree", "uname",
    "wc", "which",
}
SAFE_GIT_SUBCOMMANDS = {"branch", "diff", "log", "show", "status"}
UNSAFE_TERMINAL_TOKENS = {
    "-delete", "-exec", "-execdir", "-i", "--in-place", "--delete", "--remove",
    "--force", "--hard", "--mixed", "--soft",
}


SYSTEM_PROMPT = """You are Mateus's local personal AI assistant running on his computer.

You can use tools by replying with one JSON object and no extra text:
{"tool":"terminal.run","arguments":{"command":"pwd","cwd":"/home/mateus","timeout":30}}
{"tool":"file.read","arguments":{"path":"~/Projects/local-ai-assistant/assistant.py","start_line":1,"max_lines":220}}
{"tool":"file.list","arguments":{"path":"~/Projects/local-ai-assistant","max_depth":2}}
{"tool":"file.search","arguments":{"path":"~/Projects/local-ai-assistant","pattern":"terminal.run","glob":"*.py","max_results":20}}
{"tool":"file.write","arguments":{"path":"~/Projects/example/README.md","content":"# Example\n","overwrite":false}}
{"tool":"file.write","arguments":{"path":"~/Projects/example/README.md","content_lines":["# Example", ""],"overwrite":false}}
{"tool":"file.patch","arguments":{"path":"~/Projects/example/README.md","old":"# Example\n","new":"# Better Example\n"}}
{"tool":"dir.make","arguments":{"path":"~/Projects/example"}}
{"tool":"project.create","arguments":{"name":"hello-csharp","kind":"csharp-console","git":true}}
{"tool":"verify.path","arguments":{"path":"~/Projects/example/README.md","expect":"file"}}
{"tool":"verify.command","arguments":{"command":"python3 -m py_compile assistant.py","cwd":"~/Projects/local-ai-assistant","timeout":30}}
{"tool":"git.status","arguments":{"path":"~/Projects/local-ai-assistant"}}
{"tool":"git.diff","arguments":{"path":"~/Projects/local-ai-assistant","max_chars":12000}}
{"tool":"web.search","arguments":{"query":"Arch Linux Ollama systemd service","limit":5}}
{"tool":"web.fetch","arguments":{"url":"https://example.com","max_chars":6000}}

When you have the answer, reply with:
{"final":"your answer to Mateus"}

When public extended thinking mode is enabled, include a short public `thought` field:
{"thought":"Need to inspect the project files first.","tool":"file.list","arguments":{"path":"~/Projects/example"}}
{"thought":"The requested file was created successfully.","final":"Created the file."}

Rules:
- Think privately. Do not reveal hidden chain-of-thought. Give concise reasons and useful results.
- Public `thought` fields must be brief summaries of intent or evidence, not hidden chain-of-thought.
- Use exactly one tool call per response. After a tool result, you may request the next tool.
- Do not print raw tool JSON to Mateus as the final answer.
- For file.write, prefer content_lines for multi-line files so JSON stays valid.
- Do not use tools for casual conversation, greetings, opinions that need no inspection, or questions you can answer directly.
- Use tools when the answer depends on the local machine, files, commands, or current web data.
- Prefer file.*, git.*, and web.* tools before terminal.run when they fit the task.
- Prefer project.create for new coding projects.
- Prefer file.patch over file.write when modifying an existing file.
- Verify important file/project changes with verify.path or verify.command before claiming success.
- Prefer read-only commands first when inspecting the system.
- Never run destructive shell commands unless Mateus explicitly asks for them.
- Keep command working directories explicit when relevant.
- If a command fails, explain the failure and try a reasonable next diagnostic step.
- If verification fails, do not claim success. Explain what failed and either fix it or ask Mateus how to proceed.
- For code review, inspect enough of the relevant files before judging. Lead with concrete issues and file/line references.
"""


@dataclass
class Config:
    model: str
    host: str
    cwd: str
    terminal_mode: str
    show_tool_json: bool
    color: bool
    trace: bool
    thinking: bool


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    TRACE = "\033[90m"
    GRAY_BG = "\033[48;5;238m"
    CODE_BG = "\033[48;5;236m"


def paint(text: str, style: str, config: Config) -> str:
    if not config.color:
        return text
    return f"{style}{text}{Style.RESET}"


class Spinner:
    def __init__(self, label: str, config: Config) -> None:
        self.label = label
        self.config = config
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if not self.config.color or not sys.stderr.isatty():
            return self
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=0.2)
        if self.config.color and sys.stderr.isatty():
            sys.stderr.write("\r" + " " * (len(self.label) + 8) + "\r")
            sys.stderr.flush()

    def _spin(self) -> None:
        frames = "|/-\\"
        index = 0
        while not self.stop_event.is_set():
            frame = frames[index % len(frames)]
            sys.stderr.write("\r" + paint(frame, Style.CYAN, self.config) + f" {self.label}")
            sys.stderr.flush()
            index += 1
            time.sleep(0.12)


def terminal_width(default: int = 88) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def print_user_box(text: str, config: Config) -> None:
    width = min(max(44, terminal_width() - 4), 100)
    inner = width - 4
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        wrapped.extend(textwrap.wrap(paragraph, width=inner) or [""])
    top = "+" + "-" * (width - 2) + "+"
    print(paint(top, Style.GRAY_BG, config))
    for line in wrapped:
        print(paint(f"| {line.ljust(inner)} |", Style.GRAY_BG, config))
    print(paint(top, Style.GRAY_BG, config))


def format_ai_text(text: str, config: Config) -> str:
    if not config.color:
        return text
    lines = text.splitlines()
    formatted: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            formatted.append(paint(line, Style.DIM, config))
        elif in_code:
            formatted.append(paint(line, Style.CODE_BG, config))
        elif re.match(r"^#{1,6}\s+", stripped):
            formatted.append(paint(line, Style.BOLD + Style.CYAN, config))
        elif re.match(r"^[-*]\s+", stripped):
            formatted.append(paint(line, Style.GREEN, config))
        elif re.match(r"^\d+\.\s+", stripped):
            formatted.append(paint(line, Style.GREEN, config))
        elif stripped.startswith(("Error:", "Failed:", "Warning:")):
            formatted.append(paint(line, Style.YELLOW, config))
        elif "`" in line:
            formatted.append(re.sub(r"`([^`]+)`", lambda m: paint(m.group(1), Style.YELLOW, config), line))
        else:
            formatted.append(line)
    return "\n".join(formatted)


def print_ai(answer: str, config: Config) -> None:
    print("\n" + paint("Pico> ", Style.BOLD + Style.CYAN, config) + format_ai_text(answer, config) + "\n")


def trace_line(message: str, config: Config) -> None:
    if not config.trace:
        return
    print(paint(f"[trace] {message}", Style.TRACE, config))


def thinking_line(message: str, config: Config) -> None:
    if not config.thinking:
        return
    message = re.sub(r"\s+", " ", message.strip())
    if not message:
        return
    if len(message) > 180:
        message = message[:177] + "..."
    print(paint(f"[thinking] {message}", Style.TRACE, config))


def summarize_tool_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "terminal.run":
        return f"terminal.run: {arguments.get('command', '')}"
    if name in {"file.read", "file.write", "file.patch", "file.list", "file.search", "dir.make", "git.status", "git.diff", "verify.path"}:
        return f"{name}: {arguments.get('path', '')}"
    if name == "project.create":
        return f"project.create: {arguments.get('name', '')} ({arguments.get('kind', 'basic')})"
    if name == "verify.command":
        return f"verify.command: {arguments.get('command', '')}"
    if name == "web.search":
        return f"web.search: {arguments.get('query', '')}"
    if name == "web.fetch":
        return f"web.fetch: {arguments.get('url', '')}"
    return name or "unknown tool"


def summarize_tool_result(result: dict[str, Any]) -> str:
    status = "ok" if result.get("ok") else "failed"
    details = result.get("path") or result.get("error") or result.get("returncode") or ""
    if isinstance(details, str) and len(details) > 90:
        details = details[:87] + "..."
    return f"tool result: {status}" + (f" - {details}" if details != "" else "")


def readline_prompt(text: str, style: str, config: Config) -> str:
    if not config.color:
        return text
    # GNU readline treats / wrapped sequences as zero-width.
    return f"\001{style}\002{text}\001{Style.RESET}\002"




def slugify_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-._")
    return slug or time.strftime("chat-%Y%m%d-%H%M%S")


def chat_path(name: str) -> str:
    return os.path.join(CHAT_DIR, slugify_name(name) + ".json")


def ensure_chat_dir() -> None:
    os.makedirs(CHAT_DIR, exist_ok=True)


def save_chat(name: str, history: list[dict[str, str]], config: Config) -> str:
    ensure_chat_dir()
    path = chat_path(name)
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError):
            existing = {}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    data = {
        "name": slugify_name(name),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "model": config.model,
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def load_chat(name: str) -> tuple[str, list[dict[str, str]]]:
    path = chat_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    return str(data.get("name") or slugify_name(name)), history


def list_chats() -> list[dict[str, Any]]:
    ensure_chat_dir()
    chats: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(CHAT_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(CHAT_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        chats.append({
            "name": data.get("name") or filename[:-5],
            "updated_at": data.get("updated_at") or "",
            "messages": len(data.get("history") or []),
        })
    chats.sort(key=lambda item: str(item["updated_at"]), reverse=True)
    return chats


def command_help() -> str:
    return """Slash commands:
/new                 start a fresh unsaved chat
/save [name]         save the current chat
/resume <name>       load a saved chat
/chats               list saved chats
/tools               show available tools
/model               show the active model and terminal mode
/context             show loaded context files
/trace [on|off]      toggle dim-gray activity trace
/thinking [on|off]   toggle public extended thinking summaries
/help                show this help
/exit                quit Pico

Regular exit still works: exit, quit, or Ctrl-D."""


def handle_slash_command(
    user_text: str,
    history: list[dict[str, str]],
    session_name: str | None,
    config: Config,
) -> tuple[bool, list[dict[str, str]], str | None, bool]:
    command, _, rest = user_text.partition(" ")
    command = command.lower()
    arg = rest.strip()

    if command in {"/exit", "/quit"}:
        return True, history, session_name, True
    if command == "/help":
        print_ai(command_help(), config)
        return True, history, session_name, False
    if command == "/new":
        print_ai("Started a fresh chat.", config)
        return True, [], None, False
    if command == "/save":
        name = arg or session_name or time.strftime("chat-%Y%m%d-%H%M%S")
        path = save_chat(name, history, config)
        print_ai(f"Saved chat `{slugify_name(name)}` to `{path}`.", config)
        return True, history, slugify_name(name), False
    if command == "/resume":
        if not arg:
            print_ai("Usage: `/resume <name>`. Run `/chats` to see saved chats.", config)
            return True, history, session_name, False
        try:
            name, loaded = load_chat(arg)
        except FileNotFoundError:
            print_ai(f"No saved chat named `{slugify_name(arg)}`. Run `/chats` to see saved chats.", config)
            return True, history, session_name, False
        print_ai(f"Resumed chat `{name}` with {len(loaded)} stored messages.", config)
        return True, loaded, name, False
    if command == "/chats":
        chats = list_chats()
        if not chats:
            print_ai("No saved chats yet. Use `/save name` to save one.", config)
            return True, history, session_name, False
        lines = ["Saved chats:"]
        for item in chats[:30]:
            lines.append(f"- {item['name']} ({item['messages']} messages, updated {item['updated_at']})")
        print_ai("\n".join(lines), config)
        return True, history, session_name, False
    if command == "/tools":
        print_ai("""Available tools:
- terminal.run
- file.read
- file.list
- file.search
- file.write
- file.patch
- dir.make
- project.create
- verify.path
- verify.command
- git.status
- git.diff
- web.search
- web.fetch""", config)
        return True, history, session_name, False
    if command == "/model":
        print_ai(f"Model: `{config.model}`\nTerminal mode: `{config.terminal_mode}`\nCWD: `{config.cwd}`", config)
        return True, history, session_name, False
    if command == "/context":
        print_ai(loaded_context_summary(), config)
        return True, history, session_name, False
    if command == "/trace":
        if arg.lower() in {"on", "true", "1"}:
            config.trace = True
        elif arg.lower() in {"off", "false", "0"}:
            config.trace = False
        elif arg:
            print_ai("Usage: `/trace`, `/trace on`, or `/trace off`.", config)
            return True, history, session_name, False
        else:
            config.trace = not config.trace
        print_ai(f"Trace mode is now `{'on' if config.trace else 'off'}`.", config)
        return True, history, session_name, False
    if command == "/thinking":
        if arg.lower() in {"on", "true", "1"}:
            config.thinking = True
        elif arg.lower() in {"off", "false", "0"}:
            config.thinking = False
        elif arg:
            print_ai("Usage: `/thinking`, `/thinking on`, or `/thinking off`.", config)
            return True, history, session_name, False
        else:
            config.thinking = not config.thinking
        print_ai(f"Public extended thinking is now `{'on' if config.thinking else 'off'}`.", config)
        return True, history, session_name, False

    print_ai(f"Unknown command `{command}`. Run `/help`.", config)
    return True, history, session_name, False


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "tr", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        raw = " ".join(self.parts)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def load_context_files() -> str:
    sections: list[str] = []
    used = 0
    for path in CONTEXT_FILES:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except OSError:
            continue
        remaining = CONTEXT_MAX_CHARS - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n[truncated]"
        sections.append(f"Context from {path}:\n{content.strip()}")
        used += len(content)
    if not sections:
        return ""
    return "\n\nAdditional local context for Pico:\n" + "\n\n---\n\n".join(sections)


def loaded_context_summary() -> str:
    lines = ["Context files Pico loads:"]
    for path in CONTEXT_FILES:
        status = "loaded" if os.path.isfile(path) else "missing"
        lines.append(f"- `{path}`: {status}")
    return "\n".join(lines)


def http_json(url: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_chat(config: Config, messages: list[dict[str, str]]) -> str:
    with Spinner("Pico is thinking...", config):
        data = http_json(
            f"{config.host.rstrip('/')}/api/chat",
            {
                "model": config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 32768,
                },
            },
        )
    return data["message"]["content"].strip()


def escape_control_chars_inside_json_strings(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            output.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            output.append(char)
            escaped = True
            continue
        if char == '"':
            output.append(char)
            in_string = not in_string
            continue
        if in_string and char == "\n":
            output.append("\\n")
            continue
        if in_string and char == "\t":
            output.append("\\t")
            continue
        output.append(char)
    return "".join(output)


def decode_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        candidate = text[match.start():]
        for variant in (candidate, escape_control_chars_inside_json_strings(candidate)):
            try:
                value, _ = decoder.raw_decode(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    return decode_first_json_object(text)


def prompt_yes_no(question: str) -> bool:
    while True:
        try:
            answer = input(f"{question} [y/N] ").strip().lower()
        except EOFError:
            print()
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False


def normalize_path(value: str, base: str | None = None) -> str:
    raw = str(value or "").strip()
    if raw == "~":
        raw = DEFAULT_CWD
    elif raw.startswith("~/"):
        raw = os.path.join(DEFAULT_CWD, raw[2:])
    elif raw.startswith("/Projects/"):
        raw = os.path.join(DEFAULT_CWD, raw.lstrip("/"))
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not os.path.isabs(expanded):
        expanded = os.path.join(base or DEFAULT_CWD, expanded)
    return os.path.abspath(expanded)


def is_safe_terminal_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    if any(marker in command for marker in (";", "&&", "||", "|", ">", "<", "$(", "`")):
        return False
    if any(token in UNSAFE_TERMINAL_TOKENS for token in tokens):
        return False

    binary = os.path.basename(tokens[0])
    if binary not in SAFE_TERMINAL_COMMANDS:
        return False
    if binary == "python3":
        return len(tokens) >= 4 and tokens[1:3] == ["-m", "py_compile"]
    if binary != "git":
        return True

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-C":
            index += 2
            continue
        if token == "--no-pager":
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token in SAFE_GIT_SUBCOMMANDS
    return False


def run_terminal(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    command = str(arguments.get("command", "")).strip()
    cwd = normalize_path(str(arguments.get("cwd") or config.cwd), config.cwd)
    timeout = int(arguments.get("timeout") or 30)

    if not command:
        return {"ok": False, "error": "Missing command."}
    if not os.path.isdir(cwd):
        return {"ok": False, "error": f"Working directory does not exist: {cwd}"}

    print(f"\n[terminal] cwd={cwd}\n$ {command}")
    if config.terminal_mode == "ask":
        if not prompt_yes_no("Run this command?"):
            return {"ok": False, "error": "Mateus declined to run the command."}
    elif config.terminal_mode == "safe" and not is_safe_terminal_command(command):
        if not prompt_yes_no("This command is outside safe auto mode. Run it?"):
            return {"ok": False, "error": "Mateus declined to run the command."}

    started = time.time()
    try:
        env = os.environ.copy()
        env["HOME"] = DEFAULT_CWD
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            executable="/usr/bin/bash",
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"Command timed out after {timeout}s.",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def file_read(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or ""), config.cwd)
    start_line = max(1, int(arguments.get("start_line") or 1))
    max_lines = max(1, min(int(arguments.get("max_lines") or 160), 300))
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File does not exist: {path}"}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    start_index = start_line - 1
    selected = lines[start_index:start_index + max_lines]
    numbered = "".join(f"{i}: {line}" for i, line in enumerate(selected, start=start_line))
    return {
        "ok": True,
        "path": path,
        "start_line": start_line,
        "end_line": start_line + len(selected) - 1,
        "total_lines": len(lines),
        "text": numbered,
    }


def file_list(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or config.cwd), config.cwd)
    max_depth = max(0, min(int(arguments.get("max_depth") or 2), 6))
    if not os.path.isdir(path):
        return {"ok": False, "error": f"Directory does not exist: {path}"}

    entries: list[dict[str, Any]] = []
    root_depth = path.rstrip(os.sep).count(os.sep)
    for current, dirs, files in os.walk(path):
        depth = current.rstrip(os.sep).count(os.sep) - root_depth
        dirs[:] = [d for d in sorted(dirs) if d not in {".git", "__pycache__", ".venv", "node_modules"}]
        for name in sorted(dirs + files):
            full = os.path.join(current, name)
            rel = os.path.relpath(full, path)
            entries.append({
                "path": rel,
                "type": "dir" if os.path.isdir(full) else "file",
                "size": os.path.getsize(full) if os.path.isfile(full) else None,
            })
            if len(entries) >= 300:
                return {"ok": True, "path": path, "truncated": True, "entries": entries}
        if depth >= max_depth:
            dirs[:] = []
    return {"ok": True, "path": path, "truncated": False, "entries": entries}


def file_search(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or config.cwd), config.cwd)
    pattern = str(arguments.get("pattern") or "")
    glob = str(arguments.get("glob") or "*")
    max_results = max(1, min(int(arguments.get("max_results") or 30), 100))
    if not pattern:
        return {"ok": False, "error": "Missing search pattern."}
    if not os.path.isdir(path):
        return {"ok": False, "error": f"Directory does not exist: {path}"}

    results: list[dict[str, Any]] = []
    lowered = pattern.lower()
    for current, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
        for filename in files:
            if not fnmatch.fnmatch(filename, glob):
                continue
            full = os.path.join(current, filename)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if lowered in line.lower():
                            results.append({
                                "path": os.path.relpath(full, path),
                                "line": line_number,
                                "text": line.rstrip("\n")[:500],
                            })
                            if len(results) >= max_results:
                                return {"ok": True, "path": path, "pattern": pattern, "results": results, "truncated": True}
            except OSError:
                continue
    return {"ok": True, "path": path, "pattern": pattern, "results": results, "truncated": False}


def dir_make(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or ""), config.cwd)
    if not path:
        return {"ok": False, "error": "Missing directory path."}
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path}
    return {"ok": True, "path": path}


def file_write(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or ""), config.cwd)
    content = arguments.get("content")
    content_lines = arguments.get("content_lines")
    overwrite = bool(arguments.get("overwrite", False))
    if isinstance(content_lines, list) and all(isinstance(line, str) for line in content_lines):
        content = "\n".join(content_lines)
    if not path:
        return {"ok": False, "error": "Missing file path."}
    if not isinstance(content, str):
        return {"ok": False, "error": "Missing string content or content_lines."}
    if os.path.isdir(path):
        return {"ok": False, "error": f"Path is a directory: {path}"}
    if os.path.exists(path) and not overwrite:
        return {"ok": False, "error": f"File already exists. Retry with overwrite=true if intended: {path}"}

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        print(f"\n[file.write] overwrite {path}")
        if not prompt_yes_no("Overwrite this file?"):
            return {"ok": False, "error": "Mateus declined to overwrite the file."}
    else:
        print(f"\n[file.write] create {path}")

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path}
    return {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}


def file_patch(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or ""), config.cwd)
    old = arguments.get("old")
    new = arguments.get("new")
    expected_replacements = int(arguments.get("expected_replacements") or 1)
    if not path:
        return {"ok": False, "error": "Missing file path."}
    if not isinstance(old, str) or not isinstance(new, str):
        return {"ok": False, "error": "file.patch requires string old and new values."}
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File does not exist: {path}"}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            before = handle.read()
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path}

    count = before.count(old)
    if count != expected_replacements:
        return {
            "ok": False,
            "error": f"Expected {expected_replacements} replacement(s), found {count}.",
            "path": path,
        }
    after = before.replace(old, new, expected_replacements)
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=path + " before",
            tofile=path + " after",
            n=3,
        )
    )
    print(f"\n[file.patch] {path}")
    if diff:
        print(diff[-4000:])
    if not prompt_yes_no("Apply this patch?"):
        return {"ok": False, "error": "Mateus declined to apply the patch.", "path": path}

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(after)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path}
    return {"ok": True, "path": path, "replacements": expected_replacements}


def verify_path(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    path = normalize_path(str(arguments.get("path") or ""), config.cwd)
    expect = str(arguments.get("expect") or "exists").lower()
    if expect not in {"exists", "file", "dir", "missing"}:
        return {"ok": False, "error": "expect must be one of: exists, file, dir, missing."}

    exists = os.path.exists(path)
    is_file = os.path.isfile(path)
    is_dir = os.path.isdir(path)
    ok = (
        (expect == "exists" and exists)
        or (expect == "file" and is_file)
        or (expect == "dir" and is_dir)
        or (expect == "missing" and not exists)
    )
    result: dict[str, Any] = {
        "ok": ok,
        "path": path,
        "expect": expect,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
    }
    if is_file:
        result["size"] = os.path.getsize(path)
    if not ok:
        result["error"] = f"Path verification failed for {path}: expected {expect}."
    return result


def verify_command(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    return run_terminal(arguments, config)


def write_text_file(path: str, content: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def project_create(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    name = slugify_name(str(arguments.get("name") or "new-project"))
    kind = str(arguments.get("kind") or "basic").lower()
    root = normalize_path(str(arguments.get("root") or "~/Projects"), config.cwd)
    path_arg = arguments.get("path")
    path = normalize_path(str(path_arg or os.path.join(root, name)), config.cwd)
    if path_arg and os.path.isdir(path) and os.path.basename(path.rstrip(os.sep)) != name:
        path = os.path.join(path, name)
    git = bool(arguments.get("git", True))
    overwrite = bool(arguments.get("overwrite", False))

    if os.path.exists(path) and os.listdir(path) and not overwrite:
        return {"ok": False, "error": f"Project directory already exists and is not empty: {path}", "path": path}

    os.makedirs(path, exist_ok=True)
    created: list[str] = []
    try:
        if kind in {"basic", "empty"}:
            readme = os.path.join(path, "README.md")
            write_text_file(readme, f"# {name}\n", overwrite=overwrite)
            created.append(readme)
        elif kind in {"python", "python-cli", "python-console"}:
            pyproject = os.path.join(path, "pyproject.toml")
            main_py = os.path.join(path, "main.py")
            write_text_file(pyproject, f'[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n', overwrite=overwrite)
            write_text_file(main_py, 'def main() -> None:\n    print("Hello from Python!")\n\n\nif __name__ == "__main__":\n    main()\n', overwrite=overwrite)
            created.extend([pyproject, main_py])
        elif kind in {"csharp", "csharp-console", "dotnet-console"}:
            csproj_name = re.sub(r"[^A-Za-z0-9_.-]", "", name) or "App"
            csproj = os.path.join(path, f"{csproj_name}.csproj")
            program = os.path.join(path, "Program.cs")
            write_text_file(csproj, '<Project Sdk="Microsoft.NET.Sdk">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net8.0</TargetFramework>\n    <ImplicitUsings>enable</ImplicitUsings>\n    <Nullable>enable</Nullable>\n  </PropertyGroup>\n</Project>\n', overwrite=overwrite)
            write_text_file(program, 'Console.WriteLine("Hello, World!");\n', overwrite=overwrite)
            created.extend([csproj, program])
        elif kind in {"node", "node-js", "javascript"}:
            package_json = os.path.join(path, "package.json")
            index_js = os.path.join(path, "index.js")
            write_text_file(package_json, json.dumps({"name": name, "version": "0.1.0", "type": "module", "scripts": {"start": "node index.js"}}, indent=2) + "\n", overwrite=overwrite)
            write_text_file(index_js, 'console.log("Hello from Node.js!");\n', overwrite=overwrite)
            created.extend([package_json, index_js])
        else:
            return {"ok": False, "error": f"Unknown project kind: {kind}", "path": path}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": path, "created": created}

    git_result: dict[str, Any] | None = None
    if git:
        completed = subprocess.run(["git", "-C", path, "init"], text=True, capture_output=True, timeout=30)
        git_result = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    return {
        "ok": True,
        "path": path,
        "kind": kind,
        "created": created,
        "git": git_result,
    }


def run_git(path: str, args: list[str], timeout: int = 30) -> dict[str, Any]:
    workdir = normalize_path(path or DEFAULT_CWD, DEFAULT_CWD)
    if os.path.isfile(workdir):
        workdir = os.path.dirname(workdir)
    env = os.environ.copy()
    env["HOME"] = DEFAULT_CWD
    completed = subprocess.run(
        ["git", "-C", workdir, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "path": workdir,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def git_status(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    return run_git(str(arguments.get("path") or config.cwd), ["status", "--short"])


def git_diff(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    result = run_git(str(arguments.get("path") or config.cwd), ["diff", "--"])
    max_chars = max(1000, min(int(arguments.get("max_chars") or 12000), 50000))
    if "stdout" in result:
        result["stdout"] = result["stdout"][:max_chars]
    return result


def fetch_url(url: str, timeout: int = 20) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "local-ai-assistant/0.1 (+https://localhost)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(2_000_000)
    encoding = "utf-8"
    match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
    if match:
        encoding = match.group(1)
    return content_type, raw.decode(encoding, errors="replace")


def web_fetch(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "")).strip()
    max_chars = int(arguments.get("max_chars") or 6000)
    if not url:
        return {"ok": False, "error": "Missing URL."}
    if not re.match(r"^https?://", url):
        return {"ok": False, "error": "Only http:// and https:// URLs are supported."}

    try:
        content_type, body = fetch_url(url)
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}

    if "html" in content_type.lower() or "<html" in body[:1000].lower():
        parser = TextExtractor()
        parser.feed(body)
        body = parser.text()
    return {"ok": True, "url": url, "content_type": content_type, "text": body[:max_chars]}


def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    limit = max(1, min(int(arguments.get("limit") or 5), 10))
    if not query:
        return {"ok": False, "error": "Missing query."}

    params = urllib.parse.urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{params}"
    try:
        _, body = fetch_url(url)
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}

    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>.*?)".*?>(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
        flags=re.DOTALL,
    )
    for match in pattern.finditer(body):
        href = html.unescape(re.sub(r"<.*?>", "", match.group("href")))
        title = html.unescape(re.sub(r"<.*?>", "", match.group("title")))
        snippet = html.unescape(re.sub(r"<.*?>", "", match.group("snippet")))
        if "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            href = qs.get("uddg", [href])[0]
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break

    return {"ok": True, "query": query, "results": results}


def looks_like_casual_chat(text: str) -> bool:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9?' ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if len(lowered) > 120:
        return False
    action_words = {
        "build", "create", "make", "write", "edit", "run", "check", "search",
        "find", "open", "install", "fix", "help", "project", "program", "file",
    }
    if any(re.search(rf"\b{word}\b", lowered) for word in action_words):
        return False
    greeting_starts = (
        "hey", "hello", "hi", "how are you", "how are u", "how you doing",
        "good morning", "good afternoon", "good evening", "thanks", "thank you",
    )
    return lowered.startswith(greeting_starts)


def should_reject_tool_for_user_text(user_text: str, tool_name: str, arguments: dict[str, Any]) -> bool:
    if not looks_like_casual_chat(user_text):
        return False
    if tool_name != "terminal.run":
        return True
    command = str(arguments.get("command") or "").strip().lower()
    return command.startswith("echo ") or command in {"pwd", "whoami", "id"}


def call_tool(name: str, arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    if name == "terminal.run":
        return run_terminal(arguments, config)
    if name == "file.read":
        return file_read(arguments, config)
    if name == "file.list":
        return file_list(arguments, config)
    if name == "file.search":
        return file_search(arguments, config)
    if name == "file.write":
        return file_write(arguments, config)
    if name == "file.patch":
        return file_patch(arguments, config)
    if name == "dir.make":
        return dir_make(arguments, config)
    if name == "project.create":
        return project_create(arguments, config)
    if name == "verify.path":
        return verify_path(arguments, config)
    if name == "verify.command":
        return verify_command(arguments, config)
    if name == "git.status":
        return git_status(arguments, config)
    if name == "git.diff":
        return git_diff(arguments, config)
    if name == "web.search":
        return web_search(arguments)
    if name == "web.fetch":
        return web_fetch(arguments)
    return {"ok": False, "error": f"Unknown tool: {name}"}


def one_turn(user_text: str, history: list[dict[str, str]], config: Config) -> str:
    system_content = SYSTEM_PROMPT + (EXTENDED_THINKING_PROMPT if config.thinking else "") + load_context_files()
    messages = [{"role": "system", "content": system_content}, *history, {"role": "user", "content": user_text}]

    for _ in range(MAX_TOOL_ROUNDS):
        raw = ollama_chat(config, messages)
        if config.show_tool_json:
            print(f"\n[model]\n{raw}\n")

        command = extract_json_object(raw)
        if not command:
            trace_line("model returned final text", config)
            thinking_line("Ready to answer based on the information gathered.", config)
            return raw
        thought = command.get("thought") or command.get("thinking")
        if isinstance(thought, str):
            thinking_line(thought, config)
        if "final" in command:
            if not isinstance(thought, str):
                thinking_line("Ready to give the final answer.", config)
            trace_line("model returned final answer", config)
            return str(command["final"])

        tool_name = str(command.get("tool", ""))
        arguments = command.get("arguments") or {}
        if isinstance(arguments, dict):
            if not isinstance(thought, str):
                thinking_line("Using " + summarize_tool_call(tool_name, arguments) + " to gather needed information.", config)
            trace_line("calling " + summarize_tool_call(tool_name, arguments), config)
        if not isinstance(arguments, dict):
            result = {"ok": False, "error": "Tool arguments must be a JSON object."}
        elif should_reject_tool_for_user_text(user_text, tool_name, arguments):
            result = {"ok": False, "error": "Do not use tools for casual conversation. Answer Mateus directly."}
        else:
            result = call_tool(tool_name, arguments, config)

        if isinstance(result, dict):
            trace_line(summarize_tool_result(result), config)
        messages.append({"role": "assistant", "content": json.dumps(command)})
        messages.append({"role": "user", "content": "Tool result:\n" + json.dumps(result, ensure_ascii=False)})

    return "I hit the maximum tool-call limit for this turn. Ask me to continue if you want me to keep going."


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Local Ollama-powered assistant with terminal, file, git, and web tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              ./assistant.py
              ./assistant.py --show-tool-json
              ./assistant.py --safe-auto-terminal
              ./assistant.py --auto-approve-terminal
            """
        ),
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    parser.add_argument("--cwd", default=DEFAULT_CWD)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--safe-auto-terminal",
        action="store_true",
        help="Auto-run simple read-only terminal commands; ask before anything else.",
    )
    mode.add_argument(
        "--auto-approve-terminal",
        action="store_true",
        help="Run all model-requested terminal commands without asking. Use carefully.",
    )
    parser.add_argument("--show-tool-json", action="store_true", help="Print raw model JSON tool requests.")
    parser.add_argument("--trace", action="store_true", help="Show dim-gray activity trace for tool calls and final-answer boundaries.")
    parser.add_argument("--thinking", action="store_true", help="Show public extended thinking summaries before tool calls and final answers.")
    args = parser.parse_args()
    terminal_mode = "ask"
    if args.safe_auto_terminal:
        terminal_mode = "safe"
    if args.auto_approve_terminal:
        terminal_mode = "auto"
    return Config(
        model=args.model,
        host=args.host,
        cwd=normalize_path(args.cwd, DEFAULT_CWD),
        terminal_mode=terminal_mode,
        show_tool_json=args.show_tool_json,
        color=sys.stdout.isatty() and os.environ.get("NO_COLOR") is None,
        trace=args.trace,
        thinking=args.thinking,
    )


def main() -> int:
    config = parse_args()
    history: list[dict[str, str]] = []
    session_name: str | None = None
    print(paint(f"Local assistant using {config.model}", Style.BOLD + Style.CYAN, config))
    print(paint(f"Terminal mode: {config.terminal_mode}", Style.DIM, config))
    print("Type `/help` for commands. Type `exit` or Ctrl-D to quit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        if user_text.lower() in {"exit", "quit"}:
            return 0
        if not user_text:
            continue

        if user_text.startswith("/"):
            handled, history, session_name, should_exit = handle_slash_command(user_text, history, session_name, config)
            if should_exit:
                return 0
            if handled:
                continue

        print_user_box(user_text, config)
        try:
            answer = one_turn(user_text, history, config)
        except urllib.error.URLError as exc:
            answer = f"Ollama request failed: {exc}"
        except KeyboardInterrupt:
            print()
            continue

        print_ai(answer, config)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
        history = history[-40:]
        if session_name:
            save_chat(session_name, history, config)


if __name__ == "__main__":
    raise SystemExit(main())
