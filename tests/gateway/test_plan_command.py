"""Tests for the gateway /plan and /unplan command handlers.

Verifies that:
- /plan saves original model state and switches to plan model
- /plan sets event.text to plan skill content
- /plan preserves user instruction args
- /unplan restores original model
- /unplan cleans up _session_plan_originals
- Session cleanup removes plan state
"""

from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace

import pytest


@pytest.fixture
def gateway_runner():
    """Create a minimal GatewayRunner for testing plan commands."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._restart_requested = False
    runner._failed_platforms = {}
    runner._start_time = 0
    runner._session_model_overrides = {}
    runner._session_plan_originals = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner.adapters = {}
    runner._evict_cached_agent = MagicMock()
    runner._current_session_agents = {}
    # Stub session_key so tests use a predictable key
    runner._session_key_for_source = MagicMock(return_value="telegram:12345")
    return runner


@pytest.fixture
def mock_event():
    """Create a minimal MessageEvent for testing."""
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id="12345",
        user_id="user1",
        thread_id=None,
        message_id="msg1",
        chat_type="dm",
    )
    event = SimpleNamespace(
        source=source,
        text="/plan",
        message_id="msg1",
        platform_update_id=None,
        get_command_args=lambda: "",
    )
    return event


# =============================================================================
# _handle_plan_command tests
# =============================================================================


class TestGatewayPlanCommand:
    """Tests for GatewayRunner._handle_plan_command()."""

    @pytest.mark.asyncio
    async def test_saves_original_model_state(self, gateway_runner, mock_event):
        """Plan saves current model state before switching."""
        gateway_runner._session_model_overrides["telegram:12345"] = {
            "model": "deepseek-chat",
            "provider": "deepseek",
        }

        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": "claude-sonnet-4"},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content",
            ):
                with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                    mock_result = SimpleNamespace(
                        success=True,
                        new_model="anthropic/claude-sonnet-4",
                        target_provider="anthropic",
                        api_key=None,
                        base_url=None,
                        api_mode=None,
                        error_message=None,
                        needs_aux=False,
                    )
                    mock_switch.return_value = mock_result

                    session_key = "telegram:12345"
                    await gateway_runner._handle_plan_command(mock_event)

        assert session_key in gateway_runner._session_plan_originals
        saved = gateway_runner._session_plan_originals[session_key]
        assert saved["model"] == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_sets_event_text_with_plan_skill(self, gateway_runner, mock_event):
        """Plan sets event.text to plan skill content for the next turn."""
        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": "claude-sonnet-4"},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content-loaded",
            ):
                with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                    mock_result = SimpleNamespace(
                        success=True,
                        new_model="anthropic/claude-sonnet-4",
                        target_provider="anthropic",
                        api_key=None,
                        base_url=None,
                        api_mode=None,
                        error_message=None,
                        needs_aux=False,
                    )
                    mock_switch.return_value = mock_result

                    result = await gateway_runner._handle_plan_command(mock_event)

        assert result is None
        assert mock_event.text == "plan-skill-content-loaded"

    @pytest.mark.asyncio
    async def test_preserves_user_instruction(self, gateway_runner):
        """User's instruction after /plan is passed to the skill builder."""
        event = SimpleNamespace(
            source=SimpleNamespace(
                platform=SimpleNamespace(value="telegram"),
                chat_id="12345",
                user_id="user1",
                thread_id=None,
                message_id="msg1",
                chat_type="dm",
            ),
            text="/plan design the DB schema",
            message_id="msg1",
            platform_update_id=None,
            get_command_args=lambda: "design the DB schema",
        )

        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": ""},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message"
            ) as mock_build:
                mock_build.return_value = "plan-skill-with-args"
                await gateway_runner._handle_plan_command(event)

        args, kwargs = mock_build.call_args
        assert "design the DB schema" in str(args) + str(kwargs.values())

    @pytest.mark.asyncio
    async def test_no_plan_model_just_loads_skill(self, gateway_runner, mock_event):
        """When plan.model is not configured, just load the skill without switching."""
        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": ""},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content",
            ):
                with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                    result = await gateway_runner._handle_plan_command(mock_event)

        mock_switch.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_already_in_plan_mode_reloads_skill(self, gateway_runner, mock_event):
        """/plan again in plan mode reloads the skill without re-saving state."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
            "provider": "deepseek",
        }

        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": "claude-sonnet-4"},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-reloaded",
            ):
                with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                    result = await gateway_runner._handle_plan_command(mock_event)

        mock_switch.assert_not_called()
        assert result is None
        assert mock_event.text == "plan-skill-reloaded"
        assert gateway_runner._session_plan_originals[session_key]["model"] == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_evicts_cached_agent_after_switch(self, gateway_runner, mock_event):
        """After a model switch, the cached agent is evicted."""
        with patch(
            "gateway.run._load_gateway_config",
            return_value={
                "plan": {"model": "claude-sonnet-4"},
                "model": {"default": "deepseek-chat", "provider": "deepseek"},
            },
        ):
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill",
            ):
                with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                    mock_result = SimpleNamespace(
                        success=True,
                        new_model="anthropic/claude-sonnet-4",
                        target_provider="anthropic",
                        api_key=None,
                        base_url=None,
                        api_mode=None,
                        error_message=None,
                        needs_aux=False,
                    )
                    mock_switch.return_value = mock_result
                    await gateway_runner._handle_plan_command(mock_event)

        gateway_runner._evict_cached_agent.assert_called_once_with("telegram:12345")


# =============================================================================
# _handle_unplan_command tests
# =============================================================================


class TestGatewayUnplanCommand:
    """Tests for GatewayRunner._handle_unplan_command()."""

    @pytest.mark.asyncio
    async def test_restores_original_model(self, gateway_runner, mock_event):
        """Unplan restores the saved model and clears plan state."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "",
            "api_mode": "",
        }

        text = await gateway_runner._handle_unplan_command(mock_event)

        restored = gateway_runner._session_model_overrides.get(session_key, {})
        assert restored.get("model") == "deepseek-chat"
        gateway_runner._evict_cached_agent.assert_called_once_with(session_key)
        assert session_key not in gateway_runner._session_plan_originals
        assert "Switched back" in text

    @pytest.mark.asyncio
    async def test_not_in_plan_mode_returns_early(self, gateway_runner, mock_event):
        """Unplan with no plan state returns a helpful message."""
        text = await gateway_runner._handle_unplan_command(mock_event)

        assert "Not in plan mode" in text
        gateway_runner._evict_cached_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_evicts_agent_after_restore(self, gateway_runner, mock_event):
        """Agent cache is evicted after restoring the original model."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
            "provider": "deepseek",
        }

        await gateway_runner._handle_unplan_command(mock_event)

        gateway_runner._evict_cached_agent.assert_called_once_with(session_key)

    @pytest.mark.asyncio
    async def test_cleans_up_session_originals(self, gateway_runner, mock_event):
        """_session_plan_originals is cleaned up after unplan."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
            "provider": "deepseek",
        }

        await gateway_runner._handle_unplan_command(mock_event)

        assert session_key not in gateway_runner._session_plan_originals


# =============================================================================
# Session cleanup tests
# =============================================================================


class TestPlanStateCleanup:
    """Plan state is cleaned up when sessions end."""

    def test_new_session_clears_plan_originals(self, gateway_runner):
        """Starting a new session clears plan state for that session key."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
        }
        gateway_runner._session_model_overrides[session_key] = {
            "model": "claude-sonnet-4",
        }

        gateway_runner._session_model_overrides.pop(session_key, None)
        gateway_runner._session_plan_originals.pop(session_key, None)

        assert session_key not in gateway_runner._session_plan_originals
        assert session_key not in gateway_runner._session_model_overrides

    def test_session_reset_clears_plan_originals(self, gateway_runner):
        """Session reset removes plan originals."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
        }
        gateway_runner._session_model_overrides[session_key] = {
            "model": "claude-sonnet-4",
        }

        gateway_runner._session_model_overrides.pop(session_key, None)
        gateway_runner._session_plan_originals.pop(session_key, None)

        assert session_key not in gateway_runner._session_plan_originals

    def test_undo_restore_clears_plan_originals(self, gateway_runner):
        """Model restore after undo clears plan originals."""
        session_key = "telegram:12345"
        gateway_runner._session_plan_originals[session_key] = {
            "model": "deepseek-chat",
        }

        gateway_runner._session_model_overrides.pop(session_key, None)
        gateway_runner._session_plan_originals.pop(session_key, None)

        assert session_key not in gateway_runner._session_plan_originals


# =============================================================================
# Command dispatch tests
# =============================================================================


class TestGatewayPlanDispatch:
    """plan/unplan commands dispatch to correct handlers."""

    def test_plan_command_in_registry(self):
        """/plan is in the command registry."""
        from hermes_cli.commands import resolve_command

        canonical = resolve_command("plan")
        assert canonical is not None
        assert canonical.name == "plan"

    def test_unplan_command_in_registry(self):
        """/unplan is in the command registry."""
        from hermes_cli.commands import resolve_command

        canonical = resolve_command("unplan")
        assert canonical is not None
        assert canonical.name == "unplan"


# =============================================================================
# Config tests
# =============================================================================


class TestPlanConfig:
    """plan block in DEFAULT_CONFIG."""

    def test_plan_config_exists(self):
        """DEFAULT_CONFIG has a plan block with model field."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert "plan" in DEFAULT_CONFIG
        assert "model" in DEFAULT_CONFIG["plan"]
        assert DEFAULT_CONFIG["plan"]["model"] == ""
