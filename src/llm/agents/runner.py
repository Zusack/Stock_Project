"""Agent execution with tool-calling."""

from __future__ import annotations

import threading
from typing import Callable

from src.analysis.llm_store import create_agent_run, finish_agent_run, get_agent
from src.llm.conversation import trim_messages
from src.llm.manager import llm_manager
from src.llm.tools.registry import resolve_tools
from src.services.event_bus import event_bus
from src.services.global_control_service import GlobalControlService
from src.services.stock_config import stock_config


class AgentRunner:
    """Runs an agent with optional tool-calling."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._gc = GlobalControlService()

    def cancel(self) -> None:
        self._cancel.set()

    def run_agent(
        self,
        agent_id: int,
        user_message: str,
        *,
        ticker: str = "",
        on_token: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        cfg = stock_config()
        agent = get_agent(cfg.db_path, agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        run_id = create_agent_run(cfg.db_path, agent_id=agent_id, ticker=ticker)
        self._gc.active_process_name = f"Agent: {agent['name']}"
        self._cancel.clear()

        def _status(msg: str) -> None:
            event_bus.emit("llm_agent_status", message=msg, agent_id=agent_id, run_id=run_id)
            if on_status:
                on_status(msg)

        try:
            mgr = llm_manager()
            if not mgr.is_enabled():
                raise RuntimeError("Local AI is disabled. Enable it in Settings.")

            model = agent.get("model") or cfg.llm_chat_model
            backend = mgr.connect()
            handle = mgr.load_model(model)

            autonomy = agent.get("autonomy_level") or cfg.llm_autonomy_level
            tool_names = agent.get("allowed_tools", [])
            tools = resolve_tools(tool_names) if autonomy != "manual" and cfg.llm_agents_enabled else []

            system = agent["system_prompt"]
            if ticker:
                user_message = f"Ticker: {ticker.upper()}\n\n{user_message}"

            messages = trim_messages([
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ])

            _status("Running...")

            if tools and backend.supports_tools():
                fragments: list[str] = []

                def on_fragment(frag: str) -> None:
                    fragments.append(frag)
                    if on_token:
                        on_token(frag)
                    event_bus.emit("llm_token", token=frag)

                def on_message(msg) -> None:
                    content = getattr(msg, "content", None) or str(msg)
                    event_bus.emit("llm_agent_tool", message=content)

                if self._cancel.is_set():
                    raise InterruptedError("Cancelled")

                result = backend.act_with_tools(
                    handle,
                    [m for m in messages if m["role"] != "system"],
                    tools,
                    mgr.inference_config(),
                    on_message=on_message,
                    on_prediction_fragment=on_fragment,
                )
                text = result.response_text or "".join(fragments)
                tokens = result.tokens_generated or len(text.split())
            else:
                text = mgr.chat_stream(
                    messages,
                    cancel_event=self._cancel,
                    on_token=on_token,
                    model=model,
                )
                tokens = len(text.split())

            finish_agent_run(
                cfg.db_path,
                run_id,
                status="completed",
                summary=text[:2000],
                tokens=tokens,
            )
            event_bus.emit("llm_agent_done", run_id=run_id, summary=text[:500])
            return text

        except InterruptedError:
            finish_agent_run(cfg.db_path, run_id, status="cancelled", error="Cancelled by user")
            raise
        except Exception as ex:
            finish_agent_run(cfg.db_path, run_id, status="error", error=str(ex))
            event_bus.emit("llm_agent_done", run_id=run_id, error=str(ex))
            raise
        finally:
            self._gc.active_process_name = None

    def run_in_background(
        self,
        agent_id: int,
        user_message: str,
        *,
        ticker: str = "",
        on_done: Callable[[str | None, Exception | None], None] | None = None,
    ) -> None:
        def _work():
            try:
                result = self.run_agent(agent_id, user_message, ticker=ticker)
                if on_done:
                    on_done(result, None)
            except Exception as ex:
                if on_done:
                    on_done(None, ex)

        threading.Thread(target=_work, daemon=True).start()
