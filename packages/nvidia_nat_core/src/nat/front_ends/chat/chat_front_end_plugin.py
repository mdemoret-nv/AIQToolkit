# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from nat.builder.context import Context
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import Message
from nat.data_models.api_server import UserMessageContentRoleType
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.front_ends.chat.chat_front_end_config import ChatFrontEndConfig
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from nat.front_ends.console.console_front_end_plugin import prompt_for_input_cli
from nat.front_ends.simple_base.simple_front_end_plugin_base import SimpleFrontEndPluginBase
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class _ChatUI:
    """Rich console UI for displaying multi-turn chat with live intermediate steps.

    Layout:
    - Left panel (70%): Streams LLM response tokens as they arrive.
    - Right panel (30%): Shows tool calls and LLM events as they occur.
    """

    _RESPONSE_PANEL_TITLE = "Response"
    _STEPS_PANEL_TITLE = "Steps"

    def __init__(self, console: Console):
        self.console = console
        self.layout = Layout()
        self.layout.split_row(
            Layout(name="main", ratio=7),
            Layout(name="side", ratio=3),
        )
        self._main_text = Text()
        self._full_main_text = Text()
        self._side_text = Text()
        self._update_panels()
        self._live: Live | None = None
        self._clear_main_on_next_token = False

    def start(self) -> None:
        if not self._live:
            self._live = Live(self.layout, console=self.console, refresh_per_second=12, transient=False)
            self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def reset(self) -> None:
        """Clear all panels for a new query."""
        self._main_text = Text()
        self._full_main_text = Text()
        self._side_text = Text()
        self._clear_main_on_next_token = False
        self._update_panels()

    def on_step(self, step: IntermediateStep) -> None:
        """Dispatch an intermediate step to the appropriate UI handler."""
        event_type = step.payload.event_type

        if event_type == IntermediateStepType.LLM_NEW_TOKEN:
            self._handle_llm_token(step)
        elif event_type == IntermediateStepType.LLM_START:
            self._handle_llm_start(step)
        elif event_type == IntermediateStepType.LLM_END:
            self._handle_llm_end(step)
        elif event_type == IntermediateStepType.TOOL_START:
            self._handle_tool_start(step)
        elif event_type == IntermediateStepType.TOOL_END:
            self._handle_tool_end(step)

    def _handle_llm_token(self, step: IntermediateStep) -> None:
        if self._clear_main_on_next_token:
            self._main_text = Text()
            self._full_main_text = Text()
            self._clear_main_on_next_token = False

        token = ""
        if step.payload.data is not None and step.payload.data.chunk is not None:
            token = str(step.payload.data.chunk)

        if not token:
            return

        self._full_main_text.append(token, style="green")

        # Smart tailing: keep only the last N visible lines to avoid console overflow
        max_height = max(10, self.console.height - 6)
        lines = self._full_main_text.split("\n", allow_blank=True)
        if len(lines) > max_height:
            self._main_text = Text("... (continued) ...\n", style="dim green")
            self._main_text.append(Text("\n").join(lines[-max_height:]))
        else:
            self._main_text = self._full_main_text

        self.layout["main"].update(Panel(self._main_text, title=self._RESPONSE_PANEL_TITLE, border_style="blue"))

    def _handle_llm_start(self, step: IntermediateStep) -> None:
        name = step.payload.name or "LLM"
        self._side_text.append(f"[LLM] {name}\n", style="cyan bold")
        self.layout["side"].update(Panel(self._side_text, title=self._STEPS_PANEL_TITLE, border_style="yellow"))

    def _handle_llm_end(self, step: IntermediateStep) -> None:
        self._side_text.append("  ✓ done\n", style="dim cyan")
        self.layout["side"].update(Panel(self._side_text, title=self._STEPS_PANEL_TITLE, border_style="yellow"))

    def _handle_tool_start(self, step: IntermediateStep) -> None:
        name = step.payload.name or "tool"
        tool_input = ""
        if step.payload.data is not None and step.payload.data.input is not None:
            tool_input = str(step.payload.data.input)
            if len(tool_input) > 120:
                tool_input = tool_input[:117] + "..."

        self._side_text.append(f"\n[Tool] {name}\n", style="magenta bold")
        if tool_input:
            self._side_text.append(f"  {tool_input}\n", style="magenta")
        self.layout["side"].update(Panel(self._side_text, title=self._STEPS_PANEL_TITLE, border_style="yellow"))

        # After a tool call the LLM will emit a new response; clear the main panel then
        self._clear_main_on_next_token = True

    def _handle_tool_end(self, step: IntermediateStep) -> None:
        tool_output = ""
        if step.payload.data is not None and step.payload.data.output is not None:
            tool_output = str(step.payload.data.output)
            if len(tool_output) > 200:
                tool_output = tool_output[:197] + "..."

        self._side_text.append(f"  → {tool_output}\n", style="green")
        self._side_text.append("  " + "-" * 18 + "\n", style="dim")
        self.layout["side"].update(Panel(self._side_text, title=self._STEPS_PANEL_TITLE, border_style="yellow"))

    def _update_panels(self) -> None:
        self.layout["main"].update(Panel(self._main_text, title=self._RESPONSE_PANEL_TITLE, border_style="blue"))
        self.layout["side"].update(Panel(self._side_text, title=self._STEPS_PANEL_TITLE, border_style="yellow"))


class ChatFrontEndPlugin(SimpleFrontEndPluginBase[ChatFrontEndConfig]):
    """Interactive multi-turn chat front end for NAT workflows.

    Launched via ``nat chat --config_file <path>``.  Streams LLM tokens and
    shows tool / LLM events in a live split-pane Rich console as they occur.
    Conversation history is accumulated across turns and passed to the workflow
    as a ``ChatRequest`` so the underlying LLM receives the full message history.
    """

    def __init__(self, full_config):
        super().__init__(full_config=full_config)
        self._console = Console()
        self._ui = _ChatUI(self._console)
        self._auth_flow_handler = ConsoleAuthenticationFlowHandler()
        self._history: list[Message] = []

    async def run_workflow(self, session_manager: SessionManager) -> None:
        self._display_welcome()

        async with session_manager.session(
                user_id=self.front_end_config.user_id,
                user_input_callback=prompt_for_input_cli,
                user_authentication_callback=self._auth_flow_handler.authenticate,
        ) as session:
            await self._chat_loop(session)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _display_welcome(self) -> None:
        self._console.print(
            Panel(
                "[bold green]NeMo Agent Toolkit — Interactive Chat[/bold green]\n\n"
                "[dim]Multi-turn conversation with your workflow.[/dim]\n\n"
                "[yellow]/quit[/yellow] or [yellow]/exit[/yellow]  Exit the session",
                title="[bold white]nat chat[/bold white]",
                border_style="cyan",
                padding=(1, 2),
            ))

    async def _chat_loop(self, session) -> None:
        try:
            while True:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                if user_input.strip().lower() in ("/quit", "/exit"):
                    self._console.print("[yellow]Exiting chat...[/yellow]")
                    break

                await self._process_message(session, user_input)

        except KeyboardInterrupt:
            self._console.print("\n[yellow]Chat interrupted by user.[/yellow]")

    async def _process_message(self, session, user_input: str) -> None:
        self._console.print()
        self._console.print(Panel(user_input, title="[bold]You[/bold]", border_style="blue"))

        self._history.append(Message(content=user_input, role=UserMessageContentRoleType.USER))
        chat_request = ChatRequest(messages=list(self._history))

        self._ui.reset()
        self._ui.start()

        result: str = ""
        try:
            result = await self._run_with_steps(session, chat_request)
        except Exception as exc:
            logger.exception("Error processing message")
            self._console.print(f"[bold red]Error:[/bold red] {exc}")
            # Remove the user message that failed so history stays consistent
            self._history.pop()
        finally:
            self._ui.stop()

        if result:
            self._history.append(Message(content=result, role=UserMessageContentRoleType.ASSISTANT))
            self._console.print(Rule("[bold blue]Response[/bold blue]"))
            self._console.print(Markdown(result))
            self._console.print(Rule())

    async def _run_with_steps(self, session, chat_request: ChatRequest) -> str:
        """Run the workflow and subscribe to intermediate steps for live display."""
        loop = asyncio.get_running_loop()

        async with session.run(chat_request) as runner:
            # The event stream is active inside session.run(); subscribe now.
            step_manager = Context.get().intermediate_step_manager

            def _on_next(step: IntermediateStep) -> None:
                # Called synchronously from the event loop; schedule the UI
                # update as a task so it never blocks the reactive pipeline.
                loop.call_soon(self._ui.on_step, step)

            step_manager.subscribe(on_next=_on_next)

            return await runner.result(to_type=str)
