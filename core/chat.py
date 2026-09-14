#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CLI Chat with LOCAL_AI_CORE (AGY-style UI).
Provides:
- Antigravity-styled terminal interface with subtle horizontal separators (─)
- Persistent '>' input prompt replacing legacy 'You:'
- Animated braille spinner ('⠿ Generating...') during inference
- Real-time token streaming with seamless spinner replacement
- Automatic reasoning/thinking detection ('▸ Thought for Xs' header + dimmed thought lines)
- Slash commands (/clear, /reset, /status, /think, /help, /exit)
- Robust UTF-8 encoding & Windows VT console handling
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 I/O encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# Enable Virtual Terminal (VT) processing on Windows
try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    if os.name == "nt":
        os.system("")

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm_client import LLMClient
from core.config import load_config

# Styling Constants (ANSI Escape Sequences)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Foreground Colors
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
MAGENTA = "\033[1;35m"
GRAY = "\033[90m"
WHITE = "\033[1;37m"

SYSTEM_PROMPT = (
    "You are Qwen 2.5 Coder 3B, an intelligent offline AI assistant running locally "
    "on the user's machine (Intel Core i7-12700, 8GB RAM, AMD Radeon RX 6300 2GB VRAM via Vulkan). "
    "When asked about your current model ('model hiện tại', 'bạn là ai', or 'current model'), "
    "clearly identify yourself as Qwen 2.5 Coder 3B (Q4_K_M) running locally and offline via Vulkan. "
    "Provide clear, concise, well-structured, and helpful answers in the same language as the user "
    "(prefer fluent Vietnamese for Vietnamese queries, English for English queries)."
)

THINK_DIRECTIVE = (
    "\nBefore providing your final response, analyze the question step-by-step and "
    "enclose your internal thought process inside <think> and </think> tags. "
    "After </think>, output your direct response."
)

ANSI_REGEX = re.compile(r"\033\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for accurate character length calculation."""
    return ANSI_REGEX.sub("", text)


def get_rule_line() -> str:
    """Subtle horizontal separator matching AGY CLI."""
    cols = shutil.get_terminal_size((80, 24)).columns
    width = max(30, min(cols - 1, 72))
    return f"{GRAY}{'─' * width}{RESET}"


def pad_box_line(text: str, width: int) -> str:
    """Pad a box line accounting for invisible ANSI escape sequences."""
    vlen = len(strip_ansi(text))
    pad = max(0, width - 2 - vlen)
    return f"{GRAY}│{RESET}{text}{' ' * pad}{GRAY}│{RESET}"


COMMAND_SUGGESTIONS: list[tuple[str, str]] = [
    ("/boost", "Toggle high-performance reasoning boost"),
    ("/think", "Toggle step-by-step thought reasoning output"),
    ("/clear", "Clear the console screen"),
    ("/reset", "Reset conversation context memory"),
    ("/status", "Show model, hardware, and server health"),
    ("/help", "Show available commands and usage"),
    ("/exit", "Return to 00_MENU.bat"),
]


def filter_commands(query: str, commands: list[tuple[str, str]] = COMMAND_SUGGESTIONS) -> list[tuple[str, str]]:
    """Filter slash commands by query prefix. Returns empty list if query does not start with '/'."""
    if not query.startswith("/"):
        return []
    q = query.lower()
    return [item for item in commands if item[0].startswith(q)]


def build_popup_lines(
    filtered: list[tuple[str, str]],
    selected_idx: int,
    model_name: str = "Qwen 2.5 Coder 3B",
    width: Optional[int] = None,
) -> list[str]:
    """
    Format popup suggestions matching AGY CLI autocomplete specifications:
      ────────────────────────────────────────────────────────────────────────
      > /boost                          Toggle high-performance reasoning boost
        /think                          Toggle step-by-step thought reasoning output
        ...

        ↑/↓ Navigate · enter Select · tab Complete
      esc to cancel                                         Qwen 2.5 Coder 3B
    """
    cols = shutil.get_terminal_size((80, 24)).columns
    rule_width = width if width is not None else max(30, min(cols - 1, 72))
    rule = f"{GRAY}{'─' * rule_width}{RESET}"
    lines = [rule]
    avail = max(10, cols - 1 - 34)
    for idx, (cmd, desc) in enumerate(filtered):
        desc_str = desc if len(desc) <= avail else (desc[:max(0, avail - 3)] + "...")
        if idx == selected_idx:
            line = f"{CYAN}{BOLD}> {cmd:<32}{RESET}{WHITE}{desc_str}{RESET}"
        else:
            line = f"  {WHITE}{cmd:<32}{RESET}{GRAY}{desc_str}{RESET}"
        lines.append(line)
    lines.append("")
    lines.append(
        f"  {WHITE}↑/↓{RESET} {GRAY}Navigate{RESET} {DIM}·{RESET} "
        f"{WHITE}enter{RESET} {GRAY}Select{RESET} {DIM}·{RESET} "
        f"{WHITE}tab{RESET} {GRAY}Complete{RESET}"
    )
    esc_text = "esc to cancel"
    spaces = " " * max(1, rule_width - len(esc_text) - len(model_name))
    lines.append(f"{GRAY}{esc_text}{spaces}{model_name}{RESET}")
    return lines


def _read_char_posix() -> str:
    """Read single character on POSIX terminal."""
    return sys.stdin.read(1)


def _kbhit_posix(timeout: float = 0.0) -> bool:
    """Non-blocking keyboard hit check on POSIX using select."""
    import select
    dr, _, _ = select.select([sys.stdin], [], [], timeout)
    return len(dr) > 0


def is_interactive_console() -> bool:
    """Check if standard input/output is an interactive console supporting character-by-character input."""
    try:
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            return False
        if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
            return False
    except Exception:
        return False

    if os.name == "nt":
        try:
            import msvcrt  # noqa: F401
            return True
        except ImportError:
            return False
    else:
        try:
            import select  # noqa: F401
            import termios  # noqa: F401
            import tty  # noqa: F401
            return True
        except ImportError:
            return False


def _render_terminal_state(
    stream,
    prompt: str,
    buffer: list[str],
    cursor_pos: int,
    popup_lines: list[str],
    rendered_popup_lines: int,
) -> int:
    """
    Renders the prompt line and the popup underneath it with flicker-free VT codes.
    Returns the new number of rendered popup lines.
    """
    out = ["\033[?25l"]  # Hide cursor during redraw
    new_count = len(popup_lines)
    max_lines = max(rendered_popup_lines, new_count)

    for i in range(max_lines):
        if i < rendered_popup_lines:
            out.append("\033[1B\r\033[2K")
        else:
            out.append("\n\r\033[2K")
        if i < new_count:
            out.append(popup_lines[i])

    if max_lines > 0:
        out.append(f"\033[{max_lines}A")

    buf_str = "".join(buffer)
    out.append(f"\r\033[2K{prompt}{buf_str}")
    cols_back = len(buffer) - cursor_pos
    if cols_back > 0:
        out.append(f"\033[{cols_back}D")
    out.append("\033[?25h")

    stream.write("".join(out))
    stream.flush()
    return new_count


def _run_interactive_loop(
    prompt: str,
    model_name: str,
    stream,
    read_char,
    kbhit_fn=None,
) -> str:
    """Read interactive characters, update popup dynamically on '/'."""
    buffer: list[str] = []
    cursor_pos = 0
    selected_idx = 0
    popup_dismissed = False
    rendered_popup_lines = 0

    stream.write(f"\r\033[2K{prompt}")
    stream.flush()

    try:
        while True:
            if kbhit_fn is not None:
                while not kbhit_fn():
                    time.sleep(0.02)

            ch = read_char()
            if not ch:
                raise EOFError()

            # Handle scan code prefixes (\x00, \xe0) and ANSI escape sequences (\x1b)
            if ch in ("\x00", "\xe0"):
                ch2 = read_char()
                if not ch2:
                    raise EOFError()
            elif ch == "\x1b":
                is_escape_seq = False
                has_next = False
                if kbhit_fn is not None:
                    try:
                        has_next = kbhit_fn(timeout=0.05)
                    except TypeError:
                        has_next = kbhit_fn()
                if has_next:
                    ch2_candidate = read_char()
                    if not ch2_candidate:
                        raise EOFError()
                    if ch2_candidate in ("[", "O"):
                        ch3 = read_char()
                        if not ch3:
                            raise EOFError()
                        if ch3 == "3":
                            has_tilde = False
                            if kbhit_fn is not None:
                                try:
                                    has_tilde = kbhit_fn(timeout=0.05)
                                except TypeError:
                                    has_tilde = kbhit_fn()
                            if has_tilde:
                                read_char()  # consume '~'
                                ch = "\xe0"
                                ch2 = "S"
                                is_escape_seq = True
                        else:
                            ansi_map = {
                                "A": "H",  # Up
                                "B": "P",  # Down
                                "C": "M",  # Right
                                "D": "K",  # Left
                                "H": "G",  # Home
                                "F": "O",  # End
                            }
                            if ch3 in ansi_map:
                                ch = "\xe0"
                                ch2 = ansi_map[ch3]
                                is_escape_seq = True
                if not is_escape_seq:
                    popup_dismissed = True

            if ch in ("\x00", "\xe0"):
                buf_str = "".join(buffer)
                filtered = (
                    filter_commands(buf_str, COMMAND_SUGGESTIONS)
                    if (buf_str.startswith("/") and not popup_dismissed)
                    else []
                )
                if ch2 == "H":  # Up Arrow
                    if filtered:
                        selected_idx = (selected_idx - 1) % len(filtered)
                elif ch2 == "P":  # Down Arrow
                    if filtered:
                        selected_idx = (selected_idx + 1) % len(filtered)
                elif ch2 == "K":  # Left Arrow
                    cursor_pos = max(0, cursor_pos - 1)
                elif ch2 == "M":  # Right Arrow
                    cursor_pos = min(len(buffer), cursor_pos + 1)
                elif ch2 == "G":  # Home
                    cursor_pos = 0
                elif ch2 == "O":  # End
                    cursor_pos = len(buffer)
                elif ch2 == "S":  # Delete
                    if cursor_pos < len(buffer):
                        buffer.pop(cursor_pos)
                        selected_idx = 0
                        if not buffer or buffer[0] != "/":
                            popup_dismissed = False
            elif ch in ("\r", "\n"):
                buf_str = "".join(buffer)
                filtered = (
                    filter_commands(buf_str, COMMAND_SUGGESTIONS)
                    if (buf_str.startswith("/") and not popup_dismissed)
                    else []
                )
                if filtered:
                    idx = max(0, min(selected_idx, len(filtered) - 1))
                    selected_cmd = filtered[idx][0]
                    buffer = list(selected_cmd)
                    cursor_pos = len(buffer)
                    _render_terminal_state(stream, prompt, buffer, cursor_pos, [], rendered_popup_lines)
                    stream.write("\n")
                    stream.flush()
                    return selected_cmd
                else:
                    if rendered_popup_lines > 0:
                        _render_terminal_state(stream, prompt, buffer, cursor_pos, [], rendered_popup_lines)
                    stream.write("\n")
                    stream.flush()
                    return "".join(buffer)
            elif ch == "\t":
                buf_str = "".join(buffer)
                filtered = (
                    filter_commands(buf_str, COMMAND_SUGGESTIONS)
                    if (buf_str.startswith("/") and not popup_dismissed)
                    else []
                )
                if filtered:
                    idx = max(0, min(selected_idx, len(filtered) - 1))
                    selected_cmd = filtered[idx][0]
                    buffer = list(selected_cmd)
                    cursor_pos = len(buffer)
            elif ch == "\x1b":  # Esc handled above
                pass
            elif ch in ("\x08", "\b"):  # Backspace
                if cursor_pos > 0:
                    buffer.pop(cursor_pos - 1)
                    cursor_pos -= 1
                    selected_idx = 0
                    if not buffer or buffer[0] != "/":
                        popup_dismissed = False
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt()
            elif ch in ("\x04", "\x1a"):  # Ctrl+D / Ctrl+Z
                raise EOFError()
            elif ord(ch) >= 32 and ord(ch) != 127:
                if not buffer or buffer[0] != "/":
                    popup_dismissed = False
                buffer.insert(cursor_pos, ch)
                cursor_pos += 1
                selected_idx = 0

            buf_str = "".join(buffer)
            if buf_str.startswith("/") and not popup_dismissed:
                filtered = filter_commands(buf_str, COMMAND_SUGGESTIONS)
                if filtered:
                    selected_idx = max(0, min(selected_idx, len(filtered) - 1))
                    popup_lines = build_popup_lines(filtered, selected_idx, model_name=model_name)
                else:
                    popup_lines = []
            else:
                filtered = []
                popup_lines = []

            rendered_popup_lines = _render_terminal_state(
                stream, prompt, buffer, cursor_pos, popup_lines, rendered_popup_lines
            )
    except (KeyboardInterrupt, EOFError):
        if rendered_popup_lines > 0:
            try:
                _render_terminal_state(stream, prompt, buffer, cursor_pos, [], rendered_popup_lines)
            except Exception:
                pass
        stream.write("\n")
        stream.flush()
        raise


def read_interactive_input(
    prompt: str = f"{CYAN}{BOLD}>{RESET} ",
    model_name: str = "Qwen 2.5 Coder 3B",
    stream=None,
    getwch_fn=None,
    kbhit_fn=None,
) -> str:
    """
    Read input with real-time interactive autocomplete for slash commands.
    Falls back gracefully to standard input() if non-interactive / piped / non-Windows.
    """
    if getwch_fn is None and not is_interactive_console():
        return input(prompt)

    out_stream = stream or sys.stdout
    read_char = getwch_fn
    kbhit_func = kbhit_fn

    if read_char is None:
        if os.name == "nt":
            import msvcrt
            read_char = msvcrt.getwch
            kbhit_func = msvcrt.kbhit
        else:
            read_char = _read_char_posix
            kbhit_func = _kbhit_posix

    if os.name != "nt":
        fd = None
        old_settings = None
        cbreak_set = False
        try:
            if hasattr(sys.stdin, "fileno") and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                import termios
                import tty
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                cbreak_set = True
        except Exception:
            cbreak_set = False

        try:
            return _run_interactive_loop(prompt, model_name, out_stream, read_char, kbhit_fn=kbhit_func)
        finally:
            if cbreak_set and old_settings is not None and fd is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
    else:
        return _run_interactive_loop(prompt, model_name, out_stream, read_char, kbhit_fn=kbhit_func)


class Spinner:
    """
    Threaded braille spinner matching AGY CLI.
    In interactive consoles (is_tty), renders the exact persistent bottom prompt structure:
      ⠿ Generating...
      ────────────────────────────────────────────────────────────────────────
      > 
    and seamlessly clears it when streaming begins.
    In non-interactive pipes / tests, falls back to a clean single-line \\r spinner.
    """

    def __init__(self, message: str = "Generating...", stream=None, is_tty: Optional[bool] = None):
        self.message = message
        self.stream = stream or sys.stdout
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Frames cycle through braille, starting with ⠿ matching Screenshot 2
        self.frames = ["⠿", "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._lock = threading.Lock()
        self.is_running = False
        if is_tty is not None:
            self.is_tty = is_tty
        else:
            self.is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def set_message(self, message: str):
        with self._lock:
            self.message = message

    def start(self):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
            self._stop_event.clear()
            if self.is_tty:
                # Persistent bottom prompt structure from Screenshot 2
                rule = get_rule_line()
                prompt = f"{CYAN}{BOLD}>{RESET} "
                self.stream.write(f"{CYAN}{self.frames[0]}{RESET} {GRAY}{self.message}{RESET}\n{rule}\n{prompt}")
                self.stream.flush()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def _spin(self):
        idx = 1
        while not self._stop_event.is_set():
            time.sleep(0.08)
            with self._lock:
                frame = self.frames[idx % len(self.frames)]
                msg = self.message
            try:
                if self.is_tty:
                    # Move up 2 lines to the spinner row, rewrite frame, and move back down to prompt row
                    self.stream.write(f"\033[2A\r\033[K{CYAN}{frame}{RESET} {GRAY}{msg}{RESET}\033[2B\r\033[2C")
                else:
                    self.stream.write(f"\r\033[K{CYAN}{frame}{RESET} {GRAY}{msg}{RESET}   ")
                self.stream.flush()
            except Exception:
                pass
            idx += 1

    def stop(self, clear: bool = True):
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)

        if clear:
            try:
                if self.is_tty:
                    # Clear line 3 (prompt), move up and clear line 2 (rule), move up and clear line 1 (spinner),
                    # then write \n to leave a clean blank line above the streaming output.
                    self.stream.write("\r\033[K\033[1A\r\033[K\033[1A\r\033[K\n")
                else:
                    self.stream.write("\r\033[K" + " " * 40 + "\r")
                self.stream.flush()
            except Exception:
                pass


class StreamRenderer:
    """
    Renders streaming LLM chunks in real-time.
    - Seamlessly replaces the braille spinner on first token
    - Automatically parses and renders <think>...</think> reasoning blocks matching AGY CLI:
      ▸ Thought for Xs
      [dimmed reasoning text]
      [final answer in normal text]
    """

    def __init__(self, spinner: Optional[Spinner] = None):
        self.spinner = spinner
        self.buffer = ""
        self.decided = False
        self.in_think = False
        self.thought_buffer = ""
        self.content_buffer = ""
        self.start_time = time.perf_counter()
        self.thought_duration = 0

    def feed(self, chunk: str):
        if not chunk:
            return

        if not self.decided:
            self.buffer += chunk
            stripped = self.buffer.lstrip()

            if stripped.startswith("<think>"):
                self.decided = True
                self.in_think = True
                self.thought_buffer = stripped[len("<think>"):].lstrip("\r\n")
                if self.spinner:
                    self.spinner.set_message("Thinking...")
                chunk = ""  # ingested into thought_buffer
            elif not "<think>".startswith(stripped) or len(stripped) >= 7 or "\n" in stripped:
                self.decided = True
                if self.spinner and self.spinner.is_running:
                    self.spinner.stop(clear=True)
                sys.stdout.write(self.buffer)
                sys.stdout.flush()
                self.content_buffer += self.buffer
                self.buffer = ""
                return
            else:
                return

        if self.in_think:
            if chunk:
                self.thought_buffer += chunk
            if "</think>" in self.thought_buffer:
                self.in_think = False
                parts = self.thought_buffer.split("</think>", 1)
                thought = parts[0]
                remaining = parts[1].lstrip("\r\n")

                if self.spinner and self.spinner.is_running:
                    self.spinner.stop(clear=True)

                self.thought_duration = max(1, int(round(time.perf_counter() - self.start_time)))
                print(f"{GRAY}▸ Thought for {self.thought_duration}s{RESET}")
                for line in thought.strip().splitlines():
                    print(f"{DIM}{line}{RESET}")

                if remaining:
                    sys.stdout.write(remaining)
                    sys.stdout.flush()
                    self.content_buffer += remaining
        else:
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                self.content_buffer += chunk

    def finish(self):
        if self.spinner and getattr(self.spinner, "is_running", False):
            self.spinner.stop(clear=True)

        if not self.decided and self.buffer:
            sys.stdout.write(self.buffer)
            sys.stdout.flush()
            self.content_buffer += self.buffer
            self.buffer = ""
        elif self.in_think:
            self.thought_duration = max(1, int(round(time.perf_counter() - self.start_time)))
            print(f"{GRAY}▸ Thought for {self.thought_duration}s{RESET}")
            for line in self.thought_buffer.strip().splitlines():
                print(f"{DIM}{line}{RESET}")

    def get_full_response(self) -> str:
        if self.content_buffer.strip():
            return self.content_buffer.strip()
        return self.thought_buffer.strip()


def print_banner(server_online: bool, config, thinking_mode: bool = False):
    """Render sleek modern developer terminal header."""
    status_badge = f"{GREEN}[ONLINE - Port {config.server_port}]{RESET}" if server_online else f"{YELLOW}[STANDALONE CLI]{RESET}"
    cols = shutil.get_terminal_size((80, 24)).columns
    width = max(50, min(cols - 1, 72))

    top_border = f"{GRAY}┌{'─' * (width - 2)}┐{RESET}"
    bot_border = f"{GRAY}└{'─' * (width - 2)}┘{RESET}"

    l1 = f"  {CYAN}{BOLD}LOCAL_AI_CORE{RESET} {GRAY}—{RESET} {WHITE}Antigravity Assistant{RESET}"
    l2 = f"  Model:   {GREEN}Qwen 2.5 Coder 3B (Q4_K_M){RESET}"
    l3 = f"  Engine:  {MAGENTA}Vulkan Offline{RESET} {GRAY}|{RESET} Ctx: {WHITE}{config.ctx_size}{RESET} {GRAY}|{RESET} Threads: {WHITE}{config.threads}{RESET}"
    l4 = f"  Status:  {status_badge}"

    print()
    print(top_border)
    print(pad_box_line(l1, width))
    print(pad_box_line(l2, width))
    print(pad_box_line(l3, width))
    print(pad_box_line(l4, width))
    print(bot_border)
    print(f"  {GRAY}Commands: {WHITE}/clear{GRAY}, {WHITE}/reset{GRAY}, {WHITE}/status{GRAY}, {WHITE}/think{GRAY}, {WHITE}/boost{GRAY}, {WHITE}/help{GRAY}, {WHITE}/exit{RESET}")
    if not server_online:
        print(f"  {YELLOW}* Tip: Start server via [1] in 00_MENU.bat for instant streaming.{RESET}")
    print()


def show_status(client: LLMClient, thinking_mode: bool = False):
    online = client.is_server_available()
    cfg = getattr(client, "config", None) or load_config()
    model_name = getattr(cfg, "model_name", None) or (cfg.model_path.stem if hasattr(cfg, "model_path") else "Qwen 2.5 Coder 3B")
    print(f"\n{BOLD}System Status{RESET}")
    print(f"  {CYAN}▸{RESET} Model:          {GREEN}{model_name}{RESET}")
    print(f"  {CYAN}▸{RESET} Model Path:     {getattr(cfg, 'model_path', 'models/qwen25-coder-3b-q4km.gguf')}")
    print(f"  {CYAN}▸{RESET} Server:         {GREEN if online else YELLOW}{'ONLINE (HTTP ' + str(getattr(cfg, 'server_port', 8080)) + ')' if online else 'OFFLINE (Standalone CLI)'}{RESET}")
    print(f"  {CYAN}▸{RESET} Context Limit:  {getattr(cfg, 'ctx_size', 6144)} tokens")
    print(f"  {CYAN}▸{RESET} CPU Threads:    {getattr(cfg, 'threads', 8)} (i7-12700 P-cores)")
    print(f"  {CYAN}▸{RESET} GPU Layers:     {getattr(cfg, 'gpu_layers', 12)} (RX 6300 2GB VRAM)")
    print(f"  {CYAN}▸{RESET} Vulkan Binary:  {getattr(cfg, 'llama_server_path', Path('llama-server.exe')).name}")
    print(f"  {CYAN}▸{RESET} Reasoning Mode: {GREEN if thinking_mode else GRAY}{'ENABLED (/think, /boost)' if thinking_mode else 'AUTO (detects <think>)'}{RESET}\n")


def build_prompt(history: list, thinking_mode: bool = False) -> str:
    system_text = SYSTEM_PROMPT + (THINK_DIRECTIVE if thinking_mode else "")
    parts = [f"<|im_start|>system\n{system_text}<|im_end|>"]
    for turn in history[-8:]:  # keep last 4 exchanges to preserve context & speed
        parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def main():
    config = load_config()
    client = LLMClient(config)
    server_online = client.is_server_available()

    thinking_mode = False
    print_banner(server_online, config, thinking_mode=thinking_mode)

    history = []
    model_name = "Qwen 2.5 Coder 3B"

    while True:
        try:
            # Subtle horizontal rule line separating turns
            print(get_rule_line())

            # Loop until non-empty input or command
            while True:
                user_input = read_interactive_input(
                    prompt=f"{CYAN}{BOLD}>{RESET} ",
                    model_name=model_name,
                ).strip()
                if user_input:
                    break

            # Slash command handling
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit", "exit", "quit", ":q"):
                print(f"{GRAY}Exiting chat session... Return to menu.{RESET}\n")
                break
            elif cmd in ("/clear", "cls", "clear"):
                os.system("cls" if os.name == "nt" else "clear")
                server_online = client.is_server_available()
                print_banner(server_online, config, thinking_mode=thinking_mode)
                continue
            elif cmd in ("/reset", "/new"):
                history.clear()
                print(f"\n{GREEN}✓ Conversation memory cleared.{RESET}\n")
                continue
            elif cmd in ("/status", "/info"):
                show_status(client, thinking_mode=thinking_mode)
                continue
            elif cmd.startswith("/think") or cmd.startswith("/boost"):
                parts = cmd.split(maxsplit=1)
                arg = parts[1].strip() if len(parts) > 1 else ""
                if arg in ("on", "1", "true"):
                    thinking_mode = True
                elif arg in ("off", "0", "false"):
                    thinking_mode = False
                else:
                    thinking_mode = not thinking_mode
                status_str = f"{GREEN}ON{RESET}" if thinking_mode else f"{GRAY}OFF{RESET}"
                mode_name = "Boost / Reasoning" if "boost" in cmd else "Reasoning"
                print(f"\n{CYAN}▸ {mode_name} mode:{RESET} {status_str} {GRAY}({'Chain-of-thought enabled via <think>' if thinking_mode else 'Direct answers'}){RESET}\n")
                continue
            elif cmd in ("/help", "?"):
                print(f"\n{BOLD}Available Commands:{RESET}")
                print(f"  {WHITE}/clear{RESET}       - Clear the console screen and display banner")
                print(f"  {WHITE}/reset{RESET}       - Reset conversation context memory")
                print(f"  {WHITE}/status{RESET}      - Show model, hardware, and server health")
                print(f"  {WHITE}/think [on|off]{RESET} - Toggle step-by-step reasoning/thought output")
                print(f"  {WHITE}/boost [on|off]{RESET} - Toggle high-performance reasoning boost")
                print(f"  {WHITE}/exit{RESET}        - Return to 00_MENU.bat\n")
                continue

            history.append({"role": "user", "content": user_input})
            full_prompt = build_prompt(history, thinking_mode=thinking_mode)

            # Start animated braille spinner with persistent bottom prompt structure
            spinner = Spinner("Generating...")
            spinner.start()

            renderer = StreamRenderer(spinner=spinner)

            try:
                for chunk in client.stream_generate(full_prompt, max_tokens=1024, temperature=0.3):
                    renderer.feed(chunk)

                renderer.finish()
                print()  # Single clean blank line before next separator

                full_response = renderer.get_full_response()
                if full_response:
                    history.append({"role": "assistant", "content": full_response})

            except KeyboardInterrupt:
                if spinner.is_running:
                    spinner.stop(clear=True)
                renderer.finish()
                print(f"\n{YELLOW}[Generation stopped by user]{RESET}\n")
                partial = renderer.get_full_response()
                if partial:
                    history.append({"role": "assistant", "content": partial})

        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}Session ended. Returning to menu...{RESET}\n")
            break


if __name__ == "__main__":
    main()
