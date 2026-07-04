"""Assistant tab — LLM chat, agents, models, research, and insights."""

from __future__ import annotations

import threading

import flet as ft

from src.analysis.llm_store import (
    add_message,
    create_conversation,
    list_agent_runs,
    list_agents,
    list_ai_insights,
    list_conversations,
    list_open_questions,
    load_messages,
)
from src.llm.agents.runner import AgentRunner
from src.llm.manager import llm_manager
from src.services.stock_config import stock_config
from src.services.tab_indices import TAB_ASSISTANT
from src.utils.logger_utils import app_logger
from src.views.base_view import BaseView
from src.views.components.model_setup_bar import ModelSetupBar, active_model_id
from src.views.components.feedback import show_snackbar
from src.views.components.layouts import SectionHeader, ViewTitleBar
from src.views.theme import ButtonStyles, InputStyles, ThemeHelper
from src.views.ui_helpers import on_ticker_field_blur


class AssistantView(BaseView):
    _tab_index = TAB_ASSISTANT

    def __init__(self, page: ft.Page):
        super().__init__(page)
        self._sub_key = "chat"
        self._conversation_id: int | None = None
        self._streaming = False
        self._cancel_event = threading.Event()
        self._agent_runner = AgentRunner()
        self._pending_ticker = ""

        self.ticker_field = InputStyles.text_field(
            page,
            label="Ticker (optional)",
            width=140,
            on_blur=on_ticker_field_blur(),
        )
        self.chat_input = InputStyles.text_field(
            page,
            label="Message",
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=4,
            shift_enter=True,
            on_submit=self._on_send_chat,
        )
        self.send_btn = ft.ElevatedButton(
            "Send",
            icon=ft.Icons.SEND,
            style=ButtonStyles.primary(),
            on_click=self._on_send_chat,
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop",
            icon=ft.Icons.STOP,
            visible=False,
            on_click=self._on_stop,
        )
        self.chat_list = ft.ListView(expand=True, spacing=8, auto_scroll=True)
        self.conv_dropdown = InputStyles.dropdown(
            page,
            label="Conversation",
            width=280,
            on_select=self._on_conv_select,
        )
        self.new_chat_btn = ft.IconButton(
            icon=ft.Icons.ADD_COMMENT,
            tooltip="New chat",
            on_click=self._on_new_chat,
        )
        self.status_text = ft.Text("", size=12, color=ThemeHelper.text_muted(page))

        self.agent_dropdown = InputStyles.dropdown(page, label="Agent", width=240)
        self.run_agent_btn = ft.ElevatedButton(
            "Run agent",
            icon=ft.Icons.PLAY_ARROW,
            style=ButtonStyles.primary(),
            on_click=self._on_run_agent,
        )
        self.agent_log = ft.ListView(expand=True, spacing=6, auto_scroll=True)

        self.refresh_models_btn = ft.ElevatedButton(
            "Refresh models",
            icon=ft.Icons.REFRESH,
            on_click=self._on_refresh_models,
        )
        self.load_model_btn = ft.ElevatedButton(
            "Load selected",
            icon=ft.Icons.DOWNLOAD,
            style=ButtonStyles.primary(),
            on_click=self._on_load_model,
        )
        self.selected_model_label = ft.Text(
            "Click a model below to select it, then choose Load selected.",
            size=12,
            color=ThemeHelper.text_muted(page),
        )
        self.models_list = ft.ListView(expand=True, spacing=6)
        self._selected_model = active_model_id()
        self._setup_mode = False
        self._models_loading = False
        self._model_catalog: list = []
        self._models_load_error: str | None = None

        self.questions_list = ft.ListView(expand=True, spacing=6)
        self.research_btn = ft.ElevatedButton(
            "Research open questions",
            icon=ft.Icons.SEARCH,
            on_click=self._on_research_questions,
        )

        self.insights_list = ft.ListView(expand=True, spacing=6)
        self.insights_ticker = InputStyles.text_field(
            page,
            label="Filter ticker",
            width=140,
            on_blur=on_ticker_field_blur(),
        )
        self.refresh_insights_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh insights",
            on_click=lambda e: self._load_insights(),
        )

        self._panels = {
            "chat": self._build_chat_panel(page),
            "agents": self._build_agents_panel(page),
            "research": self._build_research_panel(page),
            "insights": self._build_insights_panel(page),
        }
        self._workflow_switcher = ft.AnimatedSwitcher(
            content=self._panels["chat"],
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=200,
            expand=True,
        )
        self._workflow_segmented = ft.SegmentedButton(
            selected=["chat"],
            allow_empty_selection=False,
            allow_multiple_selection=False,
            segments=[
                ft.Segment(value="chat", label=ft.Text("Chat"), icon=ft.Icon(ft.Icons.CHAT)),
                ft.Segment(value="agents", label=ft.Text("Agents"), icon=ft.Icon(ft.Icons.SMART_TOY)),
                ft.Segment(value="research", label=ft.Text("Research"), icon=ft.Icon(ft.Icons.QUESTION_ANSWER)),
                ft.Segment(value="insights", label=ft.Text("Insights"), icon=ft.Icon(ft.Icons.LIGHTBULB)),
            ],
            on_change=self._on_sub_change,
        )
        self._workflows_column = ft.Column(
            [
                ft.Text(
                    "Chat, agents, research, and insights use the active model above.",
                    size=11,
                    italic=True,
                    color=ThemeHelper.text_muted(page),
                ),
                self._workflow_segmented,
                self.status_text,
                self._workflow_switcher,
            ],
            expand=True,
            spacing=8,
        )
        self._models_column = ft.Column(
            [
                SectionHeader("Model setup", icon=ft.Icons.MEMORY, page_ref=page),
                ft.Text(
                    "Step 1 — choose a model from your local backend and load it into memory. "
                    "This is required before chat, agents, or research can run.",
                    size=12,
                    color=ThemeHelper.text_muted(page),
                ),
                ft.Row([self.refresh_models_btn, self.load_model_btn], spacing=8),
                self.selected_model_label,
                ft.Container(
                    content=self.models_list,
                    expand=True,
                    border=ft.border.all(1, ThemeHelper.border_default(page)),
                    border_radius=8,
                    padding=10,
                ),
            ],
            expand=True,
            spacing=8,
        )
        self.model_setup_bar = ModelSetupBar(
            page,
            on_open_setup=self._open_models_setup,
            on_close_setup=self._close_models_setup,
        )
        self._body_switcher = ft.AnimatedSwitcher(
            content=self._workflows_column,
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=200,
            expand=True,
        )

        self.controls = [
            ViewTitleBar("Assistant"),
            self.model_setup_bar,
            self._body_switcher,
        ]

        self._refresh_model_setup_bar()

        self._setup_pubsub({
            "llm_token": self._on_llm_token,
            "llm_agent_status": self._on_agent_status,
            "llm_agent_done": self._on_agent_done,
            "navigate_assistant": self._on_navigate_assistant,
        })

    def _build_chat_panel(self, page: ft.Page) -> ft.Container:
        return ft.Container(
            expand=True,
            content=ft.Column(
                [
                    ft.Row([self.conv_dropdown, self.new_chat_btn, self.ticker_field], spacing=8),
                    ft.Container(content=self.chat_list, expand=True, border=ft.border.all(1, ThemeHelper.border_default(page)), border_radius=8, padding=10),
                    ft.Row([self.chat_input, self.send_btn, self.stop_btn], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END),
                ],
                expand=True,
                spacing=8,
            ),
        )

    def _build_agents_panel(self, page: ft.Page) -> ft.Container:
        return ft.Container(
            expand=True,
            content=ft.Column(
                [
                    ft.Row([self.agent_dropdown, self.ticker_field, self.run_agent_btn], spacing=8),
                    ft.Container(content=self.agent_log, expand=True, border=ft.border.all(1, ThemeHelper.border_default(page)), border_radius=8, padding=10),
                ],
                expand=True,
                spacing=8,
            ),
        )

    def _refresh_model_setup_bar(self) -> None:
        self.model_setup_bar.setup_mode = self._setup_mode
        self.model_setup_bar.refresh()

    def _open_models_setup(self) -> None:
        self._setup_mode = True
        self._body_switcher.content = self._models_column
        self._refresh_model_setup_bar()
        self._update_selected_model_label()
        self._flush_assistant_ui()
        self._load_models()

    def _close_models_setup(self) -> None:
        self._setup_mode = False
        self._body_switcher.content = self._workflows_column
        self._refresh_model_setup_bar()
        self._flush_assistant_ui()

    def _flush_assistant_ui(self) -> None:
        try:
            self.model_setup_bar.update()
            self._body_switcher.update()
            if self.page_ref:
                self.page_ref.update()
        except RuntimeError:
            pass

    def _flush_models_ui(self) -> None:
        try:
            self.models_list.update()
            self.refresh_models_btn.update()
            self.load_model_btn.update()
            self.selected_model_label.update()
            self._flush_assistant_ui()
        except RuntimeError:
            pass
    def _build_research_panel(self, page: ft.Page) -> ft.Container:
        return ft.Container(
            expand=True,
            content=ft.Column(
                [
                    self.research_btn,
                    SectionHeader("Open questions", page_ref=page),
                    ft.Container(content=self.questions_list, expand=True, border=ft.border.all(1, ThemeHelper.border_default(page)), border_radius=8, padding=10),
                ],
                expand=True,
                spacing=8,
            ),
        )

    def _build_insights_panel(self, page: ft.Page) -> ft.Container:
        return ft.Container(
            expand=True,
            content=ft.Column(
                [
                    ft.Row([self.insights_ticker, self.refresh_insights_btn], spacing=8),
                    ft.Container(content=self.insights_list, expand=True, border=ft.border.all(1, ThemeHelper.border_default(page)), border_radius=8, padding=10),
                ],
                expand=True,
                spacing=8,
            ),
        )

    def _on_sub_change(self, e) -> None:
        key = (e.control.selected or ["chat"])[0]
        self._sub_key = key
        self._workflow_switcher.content = self._panels.get(key, self._panels["chat"])
        self._flush_assistant_ui()
        if key == "research":
            self._load_questions()
        elif key == "insights":
            self._load_insights()
        elif key == "agents":
            self._load_agents()

    def _on_navigate_assistant(self, **kwargs) -> None:
        ticker = kwargs.get("ticker", "")
        if ticker:
            self._pending_ticker = str(ticker).upper()
            self.ticker_field.value = self._pending_ticker
        if kwargs.get("open_models"):
            self._open_models_setup()
        else:
            self._safe_update(lambda: None)

    def refresh_data(self) -> None:
        self._load_conversations()
        self._load_agents()
        self._refresh_model_setup_bar()
        if self._setup_mode:
            self._render_models_list()
            self._flush_models_ui()

    def _fetch_data(self):
        db = stock_config().db_path
        return {
            "conversations": list_conversations(db),
            "agents": list_agents(db, enabled_only=True),
        }

    def _apply_data(self, data) -> None:
        self._populate_conversations(data.get("conversations", []))
        self._populate_agents(data.get("agents", []))
        if self._pending_ticker:
            self.ticker_field.value = self._pending_ticker

    def _load_conversations(self) -> None:
        db = stock_config().db_path
        self._populate_conversations(list_conversations(db))

    def _populate_conversations(self, convs: list) -> None:
        self.conv_dropdown.options = [
            ft.dropdown.Option(str(c["id"]), c["title"][:40]) for c in convs
        ]
        if convs and not self._conversation_id:
            self._conversation_id = convs[0]["id"]
            self.conv_dropdown.value = str(convs[0]["id"])
            self._load_chat_messages()

    def _load_agents(self) -> None:
        db = stock_config().db_path
        self._populate_agents(list_agents(db, enabled_only=True))

    def _populate_agents(self, agents: list) -> None:
        self.agent_dropdown.options = [
            ft.dropdown.Option(str(a["id"]), a["name"]) for a in agents
        ]
        if agents:
            self.agent_dropdown.value = str(agents[0]["id"])

    def _on_conv_select(self, e) -> None:
        if self.conv_dropdown.value:
            self._conversation_id = int(self.conv_dropdown.value)
            self._load_chat_messages()

    def _on_new_chat(self, e) -> None:
        db = stock_config().db_path
        cfg = stock_config()
        cid = create_conversation(
            db,
            title="New chat",
            model=cfg.llm_chat_model,
            backend_type=cfg.llm_backend_type,
        )
        self._conversation_id = cid
        self.chat_list.controls.clear()
        self._load_conversations()
        self.conv_dropdown.value = str(cid)

    def _load_chat_messages(self) -> None:
        if not self._conversation_id:
            return
        db = stock_config().db_path
        msgs = load_messages(db, self._conversation_id)
        self.chat_list.controls.clear()
        for m in msgs:
            self._append_bubble(m["role"], m["content"])
        self._safe_update(lambda: None)

    def _append_bubble(self, role: str, content: str) -> None:
        align = ft.MainAxisAlignment.END if role == "user" else ft.MainAxisAlignment.START
        bg = ft.Colors.PRIMARY_CONTAINER if role == "user" else ft.Colors.SURFACE_CONTAINER_HIGHEST
        self.chat_list.controls.append(
            ft.Row(
                [
                    ft.Container(
                        content=ft.Text(content, selectable=True, size=13),
                        bgcolor=bg,
                        padding=10,
                        border_radius=8,
                        width=500,
                    )
                ],
                alignment=align,
            )
        )

    def _on_send_chat(self, e) -> None:
        if self._streaming:
            return
        text = (self.chat_input.value or "").strip()
        if not text:
            return
        if not stock_config().lm_studio_enabled:
            show_snackbar(self.page_ref, "Enable Local AI in Settings first.", severity="warning")
            return

        cfg = stock_config()
        if not (cfg.llm_chat_model or "").strip():
            show_snackbar(
                self.page_ref,
                "Set Default chat model in Settings → Local AI, then try again.",
                severity="warning",
            )
            return

        if not self._conversation_id:
            self._on_new_chat(None)

        db = stock_config().db_path
        conv_id = self._conversation_id
        add_message(db, conversation_id=conv_id, role="user", content=text)
        self._append_bubble("user", text)
        self.chat_input.value = ""
        self._begin_streaming()
        app_logger.log(
            "ASSISTANT",
            f"Chat send (conv={conv_id}, chars={len(text)}).",
            model=cfg.llm_chat_model,
            backend=cfg.llm_backend_type,
        )

        messages = load_messages(db, conv_id)

        def _work():
            err_msg = ""
            response = ""
            try:
                mgr = llm_manager()
                response = mgr.chat_stream(
                    messages,
                    cancel_event=self._cancel_event,
                    on_token=self._on_stream_token,
                )
                if response.strip():
                    add_message(db, conversation_id=conv_id, role="assistant", content=response)
                elif not self._cancel_event.is_set():
                    err_msg = "Model returned an empty response."
                    self._on_stream_token(f"\n[{err_msg}]")
            except InterruptedError:
                self._on_stream_token("\n[Stopped]")
                app_logger.log("ASSISTANT", "Chat stopped by user.", level="INFO", conv_id=conv_id)
            except Exception as ex:
                err_msg = str(ex)
                self._on_stream_token(f"\n[Error: {err_msg}]")
                app_logger.log(
                    "ASSISTANT",
                    f"Chat failed: {err_msg}",
                    level="ERROR",
                    conv_id=conv_id,
                )
            finally:
                self._end_streaming(err_msg or None)

        threading.Thread(target=_work, daemon=True).start()

    def _begin_streaming(self) -> None:
        self._streaming = True
        self._cancel_event.clear()
        self.send_btn.disabled = True
        self.chat_input.disabled = True
        self.stop_btn.visible = True
        self._assistant_bubble_idx = len(self.chat_list.controls)
        self._append_bubble("assistant", "")
        self._stream_buffer = ""
        with self._throttle_lock:
            self._throttle_last_update.pop("llm_stream", None)
        self._flush_chat_ui()

    def _end_streaming(self, error: str | None = None) -> None:
        self._streaming = False

        def _ui():
            self.send_btn.disabled = False
            self.chat_input.disabled = False
            self.stop_btn.visible = False
            self._flush_assistant_bubble()
            if error:
                self.status_text.value = error[:120]
                show_snackbar(self.page_ref, error[:200], severity="error")
            self._flush_chat_ui()

        self._safe_update_critical(_ui)

    def _flush_assistant_bubble(self) -> None:
        idx = getattr(self, "_assistant_bubble_idx", None)
        buffer = getattr(self, "_stream_buffer", "")
        if idx is None or idx >= len(self.chat_list.controls):
            return
        row = self.chat_list.controls[idx]
        if row.controls and hasattr(row.controls[0], "content"):
            row.controls[0].content.value = buffer

    def _flush_chat_ui(self) -> None:
        try:
            self.chat_input.update()
            self.send_btn.update()
            self.stop_btn.update()
            self.chat_list.update()
            if self.page_ref:
                self.page_ref.update()
        except RuntimeError:
            pass

    def _on_stream_token(self, token: str) -> None:
        self._stream_buffer = getattr(self, "_stream_buffer", "") + token

        def _ui():
            self._flush_assistant_bubble()
            self._flush_chat_ui()

        self._safe_update_throttled("llm_stream", 0.05, _ui)

    def _on_llm_token(self, **kwargs) -> None:
        if not self._is_active_tab():
            return

    def _on_stop(self, e) -> None:
        self._cancel_event.set()
        self._agent_runner.cancel()

    def _on_run_agent(self, e) -> None:
        if not self.agent_dropdown.value:
            return
        agent_id = int(self.agent_dropdown.value)
        ticker = (self.ticker_field.value or "").strip().upper()
        prompt = f"Analyze {ticker}" if ticker else "Run your configured analysis workflow."
        self.agent_log.controls.append(ft.Text(f"Starting agent run...", size=12))
        self._safe_update(lambda: None)

        def on_done(result, error):
            def _ui():
                if error:
                    self.agent_log.controls.append(
                        ft.Text(f"Error: {error}", color=ThemeHelper.text_error(self.page_ref))
                    )
                else:
                    self.agent_log.controls.append(ft.Text(result[:3000] if result else "(no output)", selectable=True, size=12))
                self._load_agent_runs()
            self._safe_update(_ui)

        self._agent_runner.run_in_background(agent_id, prompt, ticker=ticker, on_done=on_done)

    def _on_agent_status(self, **kwargs) -> None:
        msg = kwargs.get("message", "")
        if msg:
            self._safe_update(lambda: self.agent_log.controls.append(ft.Text(msg, size=11, italic=True)))

    def _on_agent_done(self, **kwargs) -> None:
        summary = kwargs.get("summary", "")
        if summary:
            self._safe_update(lambda: self.status_text.__setattr__("value", summary[:120]))

    def _load_agent_runs(self) -> None:
        runs = list_agent_runs(stock_config().db_path, limit=10)
        for r in reversed(runs[-5:]):
            self.agent_log.controls.append(
                ft.Text(f"[{r['status']}] {r['agent_name']} {r['ticker']}: {r['summary'][:80] if r['summary'] else ''}", size=11)
            )

    def _update_selected_model_label(self) -> None:
        if self._selected_model:
            active = active_model_id()
            if self._selected_model == active:
                self.selected_model_label.value = f"Selected: {self._selected_model} (active)"
            else:
                self.selected_model_label.value = (
                    f"Selected: {self._selected_model} — click Load selected to activate."
                )
        else:
            self.selected_model_label.value = (
                "Click a model below to select it, then choose Load selected."
            )

    def _build_model_tile(self, page: ft.Page, model) -> ft.Container:
        ident = model.identifier
        is_selected = ident == self._selected_model
        is_active = ident == active_model_id()
        label = (
            f"{model.display_name or ident} | ctx={model.max_context_length} "
            f"| tools={model.trained_for_tool_use}"
        )
        if is_active:
            label = f"ACTIVE — {label}"
        border_color = (
            ThemeHelper.outline_selection(page)
            if is_selected
            else ThemeHelper.border_default(page)
        )
        bgcolor = (
            ThemeHelper.status_bg(page, "info")
            if is_selected
            else ThemeHelper.surface_raised(page)
        )
        trailing = ft.Icon(ft.Icons.CHECK_CIRCLE, color=ThemeHelper.accent_green(page), size=18) if is_active else None
        if is_selected and not is_active:
            trailing = ft.Icon(ft.Icons.RADIO_BUTTON_CHECKED, color=ThemeHelper.accent_blue(page), size=18)

        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(ident, size=13, weight=ft.FontWeight.W_600),
                            ft.Text(label, size=11, color=ThemeHelper.text_muted(page)),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    trailing or ft.Container(width=0),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding(12, 10, 12, 10),
            border_radius=8,
            border=ft.border.all(
                ThemeHelper.outline_selection_width() if is_selected else 1,
                border_color,
            ),
            bgcolor=bgcolor,
            on_click=lambda _e, model_id=ident: self._select_model(model_id),
            ink=True,
        )

    def _render_models_list(self) -> None:
        page = self.page_ref
        self.models_list.controls.clear()
        if self._models_loading:
            self.models_list.controls.append(
                ft.Row(
                    [
                        ft.ProgressRing(width=22, height=22, stroke_width=2),
                        ft.Text("Loading models from backend…", size=12),
                    ],
                    spacing=10,
                )
            )
            return
        if self._models_load_error:
            self.models_list.controls.append(
                ft.Text(
                    f"Error: {self._models_load_error}",
                    color=ThemeHelper.text_error(page),
                    size=12,
                )
            )
            return
        if not self._model_catalog:
            self.models_list.controls.append(ft.Text("No models found.", size=12))
            return
        for model in self._model_catalog:
            self.models_list.controls.append(self._build_model_tile(page, model))

    def _load_models(self) -> None:
        self._models_loading = True
        self._models_load_error = None
        self._render_models_list()
        self.refresh_models_btn.disabled = True
        self._flush_models_ui()

        def _work():
            models, err = llm_manager().list_models()

            def _ui():
                self._models_loading = False
                self.refresh_models_btn.disabled = False
                self._model_catalog = models or []
                self._models_load_error = err
                if not self._selected_model and self._model_catalog:
                    active = active_model_id()
                    self._selected_model = active or self._model_catalog[0].identifier
                self._render_models_list()
                self._update_selected_model_label()
                self._flush_models_ui()
                app_logger.log(
                    "ASSISTANT",
                    "Model list refreshed.",
                    count=len(self._model_catalog),
                    error=(err or "")[:120],
                )

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _select_model(self, ident: str) -> None:
        self._selected_model = ident
        self._render_models_list()
        self._update_selected_model_label()
        self._flush_models_ui()

    def _on_refresh_models(self, e) -> None:
        llm_manager().reset()
        self._load_models()

    def _on_load_model(self, e) -> None:
        model = self._selected_model or stock_config().llm_chat_model
        if not model:
            show_snackbar(self.page_ref, "Select a model first.", severity="warning")
            return

        self.load_model_btn.disabled = True
        self.selected_model_label.value = f"Loading {model}…"
        self._flush_models_ui()

        def _work():
            err = ""
            try:
                llm_manager().load_model(model)
                cfg = stock_config()
                cfg.llm_chat_model = model
                app_logger.log("ASSISTANT", f"Model loaded: {model}.", model=model)
            except Exception as ex:
                err = str(ex)
                app_logger.log("ASSISTANT", f"Model load failed: {err}", level="ERROR", model=model)

            def _ui():
                from src.views.components.model_setup_bar import note_backend_connection

                self.load_model_btn.disabled = False
                self._selected_model = model
                self._render_models_list()
                self._update_selected_model_label()
                note_backend_connection(not err)
                self._refresh_model_setup_bar()
                if err:
                    self.status_text.value = err[:120]
                    show_snackbar(self.page_ref, err[:200], severity="error")
                else:
                    self.status_text.value = f"Loaded: {model}"
                    show_snackbar(self.page_ref, f"Model loaded: {model}", severity="success")
                self._flush_models_ui()

            self._safe_update_critical(_ui)

        threading.Thread(target=_work, daemon=True).start()

    def _load_questions(self) -> None:
        db = stock_config().db_path
        questions = list_open_questions(db, status="open")
        self.questions_list.controls.clear()
        for q in questions:
            self.questions_list.controls.append(
                ft.ListTile(
                    title=ft.Text(q["question"][:120], size=13),
                    subtitle=ft.Text(f"{q['ticker']} — {q['created_at']}", size=11),
                )
            )
        if not questions:
            self.questions_list.controls.append(ft.Text("No open questions.", size=12))

    def _on_research_questions(self, e) -> None:
        agents = list_agents(stock_config().db_path, enabled_only=True)
        researcher = next((a for a in agents if a["name"] == "Researcher"), None)
        if not researcher:
            show_snackbar(self.page_ref, "Researcher agent not found.", severity="error")
            return
        self._sub_key = "agents"
        self._setup_mode = False
        self._body_switcher.content = self._workflows_column
        self._workflow_switcher.content = self._panels["agents"]
        self._workflow_segmented.selected = ["agents"]
        self._refresh_model_setup_bar()
        self._flush_assistant_ui()
        ticker = (self.ticker_field.value or "").strip().upper()
        self._agent_runner.run_in_background(
            researcher["id"],
            "Research all open questions and record findings.",
            ticker=ticker,
            on_done=lambda r, err: self._safe_update(self._load_questions),
        )

    def _load_insights(self) -> None:
        db = stock_config().db_path
        ticker = (self.insights_ticker.value or "").strip().upper() or None
        insights = list_ai_insights(db, ticker=ticker, limit=30)
        self.insights_list.controls.clear()
        for ins in insights:
            summary = ins["payload"].get("summary", str(ins["payload"])[:200])
            self.insights_list.controls.append(
                ft.ListTile(
                    title=ft.Text(f"{ins['ticker']} — {ins['kind']}", size=13),
                    subtitle=ft.Text(f"{ins['created_at']}: {summary[:150]}", size=11),
                )
            )
        if not insights:
            self.insights_list.controls.append(ft.Text("No insights stored yet.", size=12))
