# Local AI Assistant

A small personal assistant powered by your local Ollama Qwen model.

It can:

- chat through Ollama
- ask to run terminal commands
- auto-run simple read-only terminal commands in safe mode
- read, list, and search local project files without raw shell commands
- patch existing files with review before applying
- create basic Python, C#, Node, and generic projects
- verify paths and commands before claiming success
- inspect git status and diffs
- search the web through DuckDuckGo HTML results
- fetch and summarize web pages
- keep short conversation history during the session
- save and resume chats with slash commands
- show a colored terminal UI with user-message boxes and a thinking spinner

## Run

```bash
pico
```

## Launcher

The terminal command is installed at `~/.local/bin/pico`. The old `Picoassistant` command forwards to `pico` for compatibility.

Your installed model is used by default:

```text
hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL
```

Override it with:

```bash
OLLAMA_MODEL='model-name-here' ./assistant.py
```

## Local Context

Pico loads workstation context at the start of each model turn from:

```text
~/README.md
~/AGENTS.md
~/CLAUDE.md
```

Use `/context` inside Pico to check which context files are available. Edit `~/README.md` for Pico-specific notes about your system and preferences.

## Slash Commands

```text
/new                 start a fresh unsaved chat
/save [name]         save the current chat
/resume <name>       load a saved chat
/chats               list saved chats
/tools               show available tools
/model               show the active model and terminal mode
/context             show loaded context files
/trace [on|off]      toggle dim-gray activity trace
/thinking [on|off]   toggle public extended thinking summaries
/help                show command help
/exit                quit Pico
```

Saved chats live here:

```text
~/.local/share/picoassistant/chats/
```

When a resumed or saved chat has a name, Pico autosaves it after each reply.

## Public Extended Thinking

Pico cannot expose private chain-of-thought, but it can show short public thinking summaries before tool calls and final answers. This makes the agent easier to follow and nudges it toward more deliberate tool use.

Inside Pico:

```text
/thinking
/thinking on
/thinking off
```

Or start with it enabled:

```bash
pico --thinking
```

These summaries are brief intent/evidence labels, not raw hidden reasoning.

## Trace Mode

Pico does not show private chain-of-thought, but it can show a concise activity trace in dim gray so you can separate intermediate work from the final answer.

Inside Pico:

```text
/trace
/trace on
/trace off
```

Or start with trace enabled:

```bash
pico --trace
```

## Terminal UI

Pico uses ANSI terminal formatting when stdout is a TTY:

- your messages are repeated in a gray box
- Pico replies use light Markdown-style coloring
- diff additions use a green background and removals use a red background
- a small spinner appears while the model is thinking

Set `NO_COLOR=1` to disable colors.

## Terminal Safety

By default, every terminal command proposed by the model is shown and requires
your approval.

Safe auto mode runs simple read-only commands automatically and still asks before
anything outside the allowlist:

```bash
./assistant.py --safe-auto-terminal
```

Full autonomy runs every model-requested terminal command without asking:

```bash
./assistant.py --auto-approve-terminal
```

Use full autonomy carefully. The model can make mistakes.

## Built-In Tools

The model can call these tools:

- `terminal.run`: run shell commands
- `file.read`: read a bounded line range from a file
- `file.list`: list a directory tree
- `file.search`: search text in files
- `file.write`: create or overwrite files
- `file.patch`: replace exact text in an existing file after showing a diff
- `dir.make`: create directories
- `project.create`: scaffold basic, Python, C#, or Node projects
- `verify.path`: check whether a file or directory exists
- `verify.command`: run a command as an explicit verification step
- `git.status`: show short git status
- `git.diff`: show the current diff
- `web.search`: search the web
- `web.fetch`: fetch a web page and extract text

The assistant also normalizes `/Projects/...` to `~/Projects/...`, so accidental
absolute-looking project paths still work on this workstation.

## Debugging

To see the raw JSON tool calls:

```bash
./assistant.py --show-tool-json
```

## Example Prompts

```text
What model is Ollama running?
Search the web for the latest Ollama release notes and summarize them.
Check disk usage in my home folder.
Review ~/Projects/local-ai-assistant/assistant.py.
Find where terminal mode is implemented.
Create a tiny Python hello world project in ~/Projects/test-agent-output.
```

## Reliability Workflow

For coding tasks, Pico is prompted to follow this shape:

```text
inspect -> create/edit -> verify -> final answer
```

The most important reliability tools are `project.create`, `file.patch`,
`verify.path`, and `verify.command`.
