#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for AGY-style chat components in LOCAL_AI_CORE/core/chat.py."""

import io
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

from core.chat import (
    strip_ansi,
    pad_box_line,
    get_rule_line,
    Spinner,
    StreamRenderer,
    build_prompt,
    print_banner,
    show_status,
    THINK_DIRECTIVE,
    SYSTEM_PROMPT,
    COMMAND_SUGGESTIONS,
    filter_commands,
    build_popup_lines,
    is_interactive_console,
    read_interactive_input,
)
from core.config import load_config


class TestChatComponents(unittest.TestCase):

    def test_strip_ansi(self):
        text = "\033[1;36m>\033[0m Hello \033[90mworld\033[0m"
        self.assertEqual(strip_ansi(text), "> Hello world")
        self.assertEqual(len(strip_ansi(text)), 13)

    def test_pad_box_line(self):
        colored_title = "\033[1;36mLOCAL_AI_CORE\033[0m"
        line = pad_box_line(colored_title, 30)
        plain = strip_ansi(line)
        self.assertTrue(plain.startswith("│"))
        self.assertTrue(plain.endswith("│"))
        self.assertEqual(len(plain), 30)

    def test_get_rule_line(self):
        rule = get_rule_line()
        plain = strip_ansi(rule)
        self.assertTrue(all(ch == "─" for ch in plain))
        self.assertGreaterEqual(len(plain), 30)
        self.assertLessEqual(len(plain), 72)

    def test_spinner_lifecycle(self):
        buf = io.StringIO()
        spinner = Spinner("Testing...", stream=buf, is_tty=False)
        spinner.start()
        self.assertTrue(spinner.is_running)
        time.sleep(0.15)
        spinner.set_message("Still testing...")
        self.assertEqual(spinner.message, "Still testing...")
        spinner.stop(clear=True)
        self.assertFalse(spinner.is_running)
        self.assertIn("Testing...", buf.getvalue())

    def test_spinner_interactive_tty_mode(self):
        buf = io.StringIO()
        spinner = Spinner("Generating...", stream=buf, is_tty=True)
        spinner.start()
        self.assertTrue(spinner.is_running)
        time.sleep(0.18)
        spinner.stop(clear=True)
        self.assertFalse(spinner.is_running)

        output = buf.getvalue()
        # Verify initial 3-line persistent prompt structure from Screenshot 2
        self.assertIn("Generating...", output)
        self.assertIn("─", output)
        self.assertIn(">", output)
        # Verify cursor repositioning sequences \033[2A and clear-up \033[1A
        self.assertIn("\033[2A", output)
        self.assertIn("\033[1A", output)

    def test_stream_renderer_normal_text(self):
        spinner = MagicMock()
        renderer = StreamRenderer(spinner=spinner)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            renderer.feed("Hello ")
            renderer.feed("world!")
            renderer.finish()

        self.assertEqual(captured.getvalue(), "Hello world!")
        self.assertEqual(renderer.get_full_response(), "Hello world!")
        spinner.stop.assert_called_with(clear=True)

    def test_stream_renderer_thinking_mode_single_chunk(self):
        spinner = MagicMock()
        renderer = StreamRenderer(spinner=spinner)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            renderer.feed("<think>\nAnalyzing query...\n</think>\nFinal direct answer.")
            renderer.finish()

        output = captured.getvalue()
        plain = strip_ansi(output)
        self.assertIn("▸ Thought for", plain)
        self.assertIn("Analyzing query...", plain)
        self.assertIn("Final direct answer.", plain)
        self.assertEqual(renderer.get_full_response(), "Final direct answer.")

    def test_stream_renderer_thinking_mode_multi_chunk(self):
        spinner = MagicMock()
        renderer = StreamRenderer(spinner=spinner)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            renderer.feed("<th")
            renderer.feed("ink>\nStep 1: Check constraints\n")
            renderer.feed("Step 2: Generate response\n")
            renderer.feed("</th")
            renderer.feed("ink>\nHere is the completed code.")
            renderer.finish()

        output = captured.getvalue()
        plain = strip_ansi(output)
        self.assertIn("▸ Thought for", plain)
        self.assertIn("Step 1: Check constraints", plain)
        self.assertIn("Step 2: Generate response", plain)
        self.assertIn("Here is the completed code.", plain)
        self.assertEqual(renderer.get_full_response(), "Here is the completed code.")

    def test_stream_renderer_html_or_non_think_tag(self):
        spinner = MagicMock()
        renderer = StreamRenderer(spinner=spinner)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            renderer.feed("<di")
            renderer.feed("v class='app'>Content</div>")
            renderer.finish()

        self.assertEqual(captured.getvalue(), "<div class='app'>Content</div>")
        self.assertEqual(renderer.get_full_response(), "<div class='app'>Content</div>")

    def test_stream_renderer_unclosed_think(self):
        spinner = MagicMock()
        renderer = StreamRenderer(spinner=spinner)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            renderer.feed("<think>\nThinking that never ended...")
            renderer.finish()

        plain = strip_ansi(captured.getvalue())
        self.assertIn("▸ Thought for", plain)
        self.assertIn("Thinking that never ended...", plain)
        self.assertEqual(renderer.get_full_response(), "Thinking that never ended...")

    def test_stream_renderer_empty_stream_cleanup(self):
        spinner = MagicMock()
        spinner.is_running = True
        renderer = StreamRenderer(spinner=spinner)
        # Calling finish on empty stream must stop spinner without error
        renderer.finish()
        spinner.stop.assert_called_with(clear=True)

    def test_build_prompt(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        # Standard prompt (thinking_mode=False)
        p_std = build_prompt(history, thinking_mode=False)
        self.assertIn("<|im_start|>system", p_std)
        self.assertIn("Qwen 2.5 Coder 3B", p_std)
        self.assertNotIn("<think>", p_std)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", p_std)
        self.assertIn("<|im_start|>assistant\nHi there!<|im_end|>", p_std)
        self.assertIn("<|im_start|>user\nHow are you?<|im_end|>", p_std)
        self.assertTrue(p_std.endswith("<|im_start|>assistant\n"))

        # Thinking prompt (thinking_mode=True)
        p_think = build_prompt(history, thinking_mode=True)
        self.assertIn("<think>", p_think)
        self.assertIn(THINK_DIRECTIVE.strip(), p_think)

    def test_banner_and_status(self):
        config = load_config()
        client_mock = MagicMock()
        client_mock.is_server_available.return_value = True
        client_mock.config = config

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_banner(server_online=True, config=config, thinking_mode=True)
            show_status(client_mock, thinking_mode=True)

        plain = strip_ansi(captured.getvalue())
        self.assertIn("LOCAL_AI_CORE", plain)
        self.assertIn("ONLINE - Port 8080", plain)
        self.assertIn("/think", plain)
        self.assertIn("/boost", plain)
        self.assertIn("System Status", plain)

    def test_main_cli_simulation(self):
        from core.chat import main

        simulated_inputs = [
            "/status",
            "/think on",
            "/boost off",
            "/boost on",
            "/reset",
            "hello world",
            "/exit",
        ]

        client_mock = MagicMock()
        client_mock.is_server_available.return_value = False
        client_mock.config = load_config()
        client_mock.stream_generate.return_value = iter([
            "<think>\nSimulated reasoning\n</think>\nSimulated assistant answer"
        ])

        captured = io.StringIO()
        with patch("core.chat.LLMClient", return_value=client_mock), \
             patch("builtins.input", side_effect=simulated_inputs), \
             patch("sys.stdout", captured):
            main()

        output = strip_ansi(captured.getvalue())
        self.assertIn("LOCAL_AI_CORE", output)
        self.assertIn("System Status", output)
        self.assertIn("Reasoning mode:", output)
        self.assertIn("Boost / Reasoning mode:", output)
        self.assertIn("Conversation memory cleared", output)
        self.assertIn("▸ Thought for", output)
        self.assertIn("Simulated reasoning", output)
        self.assertIn("Simulated assistant answer", output)
        self.assertIn("Exiting chat session", output)

    def test_main_cli_keyboard_interrupt(self):
        from core.chat import main

        def stream_with_interrupt(*args, **kwargs):
            yield "First token"
            raise KeyboardInterrupt()

        client_mock = MagicMock()
        client_mock.is_server_available.return_value = False
        client_mock.config = load_config()
        client_mock.stream_generate.side_effect = stream_with_interrupt

        # First prompt raises KeyboardInterrupt during streaming, second input exits session via KeyboardInterrupt at prompt
        simulated_inputs = ["trigger interrupt"]

        captured = io.StringIO()
        with patch("core.chat.LLMClient", return_value=client_mock), \
             patch("builtins.input", side_effect=[simulated_inputs[0], KeyboardInterrupt()]), \
             patch("sys.stdout", captured):
            main()

        output = strip_ansi(captured.getvalue())
        self.assertIn("Generation stopped by user", output)
        self.assertIn("Session ended", output)

    def test_filter_commands(self):
        # Empty or non-slash inputs return empty list
        self.assertEqual(filter_commands(""), [])
        self.assertEqual(filter_commands("hello"), [])
        self.assertEqual(filter_commands("boost"), [])

        # Single slash returns all commands in original order
        all_cmds = filter_commands("/")
        self.assertEqual(len(all_cmds), 7)
        self.assertEqual(all_cmds[0][0], "/boost")
        self.assertEqual(all_cmds[1][0], "/think")
        self.assertEqual(all_cmds[2][0], "/clear")
        self.assertEqual(all_cmds[3][0], "/reset")
        self.assertEqual(all_cmds[4][0], "/status")
        self.assertEqual(all_cmds[5][0], "/help")
        self.assertEqual(all_cmds[6][0], "/exit")

        # Incremental prefix filtering
        self.assertEqual([c[0] for c in filter_commands("/b")], ["/boost"])
        self.assertEqual([c[0] for c in filter_commands("/t")], ["/think"])
        self.assertEqual([c[0] for c in filter_commands("/c")], ["/clear"])
        self.assertEqual([c[0] for c in filter_commands("/r")], ["/reset"])
        self.assertEqual([c[0] for c in filter_commands("/s")], ["/status"])
        self.assertEqual([c[0] for c in filter_commands("/st")], ["/status"])
        self.assertEqual([c[0] for c in filter_commands("/h")], ["/help"])
        self.assertEqual([c[0] for c in filter_commands("/e")], ["/exit"])

        # Case-insensitive
        self.assertEqual([c[0] for c in filter_commands("/BOOST")], ["/boost"])

        # Unknown / non-matching
        self.assertEqual(filter_commands("/xyz"), [])
        self.assertEqual(filter_commands("/boost "), [])

    def test_build_popup_lines(self):
        filtered = COMMAND_SUGGESTIONS
        lines = build_popup_lines(filtered, selected_idx=0, model_name="Qwen 2.5 Coder 3B", width=72)
        # Expected total lines: 1 rule + 7 commands + 1 blank + 1 nav + 1 footer = 11 lines
        self.assertEqual(len(lines), 11)

        plain_lines = [strip_ansi(line) for line in lines]

        # Top separator
        self.assertEqual(plain_lines[0], "─" * 72)

        # Selected item has prefix "> /boost" and starts description at col 34
        self.assertTrue(plain_lines[1].startswith("> /boost"))
        self.assertIn("Toggle high-performance reasoning boost", plain_lines[1])
        self.assertIn("\033[1;36m", lines[1])  # Cyan

        # Unselected item has prefix "  /think"
        self.assertTrue(plain_lines[2].startswith("  /think"))
        self.assertIn("Toggle step-by-step thought reasoning output", plain_lines[2])

        # Blank line before navigation hints
        self.assertEqual(plain_lines[8], "")

        # Navigation hint line
        self.assertIn("↑/↓ Navigate · enter Select · tab Complete", plain_lines[9])

        # Footer hints line
        self.assertTrue(plain_lines[10].startswith("esc to cancel"))
        self.assertTrue(plain_lines[10].endswith("Qwen 2.5 Coder 3B"))
        self.assertEqual(len(plain_lines[10]), 72)

    def test_read_interactive_input_direct_slash_enter(self):
        # Typing '/' then Enter should select the first highlighted item (/boost)
        keys = ["/", "\r"]
        getwch = iter(keys).__next__
        buf = io.StringIO()
        res = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=getwch)
        self.assertEqual(res, "/boost")

    def test_read_interactive_input_arrow_navigation(self):
        # Down Arrow: '/' -> Down -> Enter => '/think'
        keys_down = ["/", "\xe0", "P", "\r"]
        buf = io.StringIO()
        res_down = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(keys_down).__next__)
        self.assertEqual(res_down, "/think")

        # Up Arrow wrap around: '/' -> Up -> Enter => '/exit'
        keys_up = ["/", "\xe0", "H", "\r"]
        buf2 = io.StringIO()
        res_up = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf2, getwch_fn=iter(keys_up).__next__)
        self.assertEqual(res_up, "/exit")

    def test_read_interactive_input_filtering_and_selection(self):
        # Filter to /status
        keys = ["/", "s", "t", "\r"]
        buf = io.StringIO()
        res = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(keys).__next__)
        self.assertEqual(res, "/status")

        # Filter to /clear
        keys_clear = ["/", "c", "\r"]
        buf2 = io.StringIO()
        res_clear = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf2, getwch_fn=iter(keys_clear).__next__)
        self.assertEqual(res_clear, "/clear")

    def test_read_interactive_input_tab_completion(self):
        # Tab complete command name, then Enter
        keys = ["/", "b", "\t", "\r"]
        buf = io.StringIO()
        res = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(keys).__next__)
        self.assertEqual(res, "/boost")

        # Tab complete, type arguments, then Enter
        keys_args = ["/", "t", "\t", " ", "o", "n", "\r"]
        buf2 = io.StringIO()
        res_args = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf2, getwch_fn=iter(keys_args).__next__)
        self.assertEqual(res_args, "/think on")

    def test_read_interactive_input_esc_and_backspace(self):
        # Esc cancels popup, continue normal typing
        keys_esc = ["/", "\x1b", "\x08", "h", "i", "\r"]
        buf = io.StringIO()
        res_esc = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(keys_esc).__next__)
        self.assertEqual(res_esc, "hi")

        # Esc cancels popup, subsequent character typing must NOT reopen popup or select from it
        keys_esc_type = ["/", "\x1b", "b", "\r"]
        buf_esc_type = io.StringIO()
        res_esc_type = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf_esc_type, getwch_fn=iter(keys_esc_type).__next__)
        self.assertEqual(res_esc_type, "/b")
        # Ensure popup was rendered once on initial '/' and never rendered again after Esc
        esc_parts = buf_esc_type.getvalue().split("esc to cancel")
        self.assertEqual(len(esc_parts) - 1, 1)

        # Backspace past '/' closes popup cleanly
        keys_bs = ["/", "\x08", "h", "e", "y", "\r"]
        buf2 = io.StringIO()
        res_bs = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf2, getwch_fn=iter(keys_bs).__next__)
        self.assertEqual(res_bs, "hey")

        # Esc cancels popup, backspacing past '/' and retyping '/' restores popup
        keys_restore = ["/", "\x1b", "b", "\x08", "\x08", "/", "\r"]
        buf3 = io.StringIO()
        res_restore = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf3, getwch_fn=iter(keys_restore).__next__)
        self.assertEqual(res_restore, "/boost")

    def test_read_interactive_input_ansi_arrow_navigation(self):
        # Test ANSI Down Arrow sequence \x1b[B navigating to /think
        keys_ansi_down = ["/", "\x1b", "[", "B", "\r"]
        ansi_iter = iter(keys_ansi_down)
        buf = io.StringIO()
        res_ansi_down = read_interactive_input(
            "> ", "Qwen 2.5 Coder 3B",
            stream=buf,
            getwch_fn=ansi_iter.__next__,
            kbhit_fn=lambda: True,
        )
        self.assertEqual(res_ansi_down, "/think")

    def test_read_interactive_input_normal_text(self):
        # Typing regular text without slash
        keys = list("viet code python\r")
        buf = io.StringIO()
        res = read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(keys).__next__)
        self.assertEqual(res, "viet code python")

    def test_read_interactive_input_ctrl_c_and_ctrl_d(self):
        buf = io.StringIO()
        with self.assertRaises(KeyboardInterrupt):
            read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf, getwch_fn=iter(["/", "\x03"]).__next__)

        buf2 = io.StringIO()
        with self.assertRaises(EOFError):
            read_interactive_input("> ", "Qwen 2.5 Coder 3B", stream=buf2, getwch_fn=iter(["\x04"]).__next__)

    def test_read_interactive_input_fallback(self):
        # When getwch_fn is None and is_interactive_console returns False, calls input()
        with patch("core.chat.is_interactive_console", return_value=False), \
             patch("builtins.input", return_value="fallback response") as mock_input:
            res = read_interactive_input(prompt="> ")
            self.assertEqual(res, "fallback response")
            mock_input.assert_called_once_with("> ")


if __name__ == "__main__":
    unittest.main()
