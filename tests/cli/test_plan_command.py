"""Tests for the /plan and /unplan commands.

Verifies that:
- /plan saves the current model state and switches to plan model
- /plan loads the plan skill when no plan.model is configured
- /plan skips switching when already on the plan model
- /plan handles switch failures gracefully
- /unplan restores the original model
- /unplan handles no-plan-mode and edge cases
- plan/unplan dispatch correctly in the command handler
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import cli as cli_mod

    return cli_mod


# =============================================================================
# Dispatch tests — /plan and /unplan route to the right handler
# =============================================================================


class TestPlanCommandDispatch(unittest.TestCase):
    """Verify that process_command dispatches /plan and /unplan correctly."""

    def _make_cli(self):
        cli_mod = _import_cli()
        cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
        cli.conversation_history = []
        cli._pending_input = MagicMock()
        cli._plan_original_model = None
        cli.model = "deepseek-chat"
        cli.provider = "deepseek"
        cli.api_key = None
        cli.base_url = None
        cli.api_mode = "chat_completions"
        cli._confirm_destructive_slash = MagicMock(return_value=True)
        cli._apply_model_switch_result = MagicMock()
        return cli

    def test_plan_dispatches_to_handle_plan_command(self):
        cli_mod = _import_cli()
        stub = self._make_cli()

        with patch.object(
            cli_mod.HermesCLI, "_handle_plan_command"
        ) as mock_handler:
            stub.process_command("/plan")

        mock_handler.assert_called_once()

    def test_unplan_dispatches_to_handle_unplan_command(self):
        cli_mod = _import_cli()
        stub = self._make_cli()

        with patch.object(
            cli_mod.HermesCLI, "_handle_unplan_command"
        ) as mock_handler:
            stub.process_command("/unplan")

        mock_handler.assert_called_once()

    def test_plan_available_during_active_session(self):
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS

        self.assertIn("plan", ACTIVE_SESSION_BYPASS_COMMANDS)
        self.assertIn("unplan", ACTIVE_SESSION_BYPASS_COMMANDS)

    def test_plan_in_command_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [cmd.name for cmd in COMMAND_REGISTRY]
        self.assertIn("plan", names)
        self.assertIn("unplan", names)


# =============================================================================
# _handle_plan_command tests
# =============================================================================


class TestHandlePlanCommand(unittest.TestCase):
    """Tests for HermesCLI._handle_plan_command()."""

    def _make_cli(self, model="deepseek-chat", provider="deepseek"):
        cli_mod = _import_cli()
        cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
        cli.model = model
        cli.provider = provider
        cli.api_key = None
        cli.base_url = None
        cli.api_mode = "chat_completions"
        cli.session_id = "test-session-1"
        cli._plan_original_model = None
        cli._pending_input = MagicMock()
        cli._apply_model_switch_result = MagicMock()
        return cli

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_saves_original_model_before_switch(self, _cprint, mock_load_cfg):
        """Plan saves the current model/provider before switching."""
        mock_load_cfg.return_value = {"plan": {"model": "claude-sonnet-4"}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")

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

            cli_mod.HermesCLI._handle_plan_command(stub, "/plan design DB")

        self.assertIsNotNone(stub._plan_original_model)
        self.assertEqual(stub._plan_original_model["model"], "deepseek-chat")
        self.assertEqual(stub._plan_original_model["provider"], "deepseek")

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_switches_to_plan_model(self, _cprint, mock_load_cfg):
        """Plan switches model when plan.model is configured and different."""
        mock_load_cfg.return_value = {"plan": {"model": "claude-sonnet-4"}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")

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

            cli_mod.HermesCLI._handle_plan_command(stub, "/plan design DB")

        stub._apply_model_switch_result.assert_called_once()

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_skip_switch_when_already_on_plan_model(self, _cprint, mock_load_cfg):
        """No switch when already on the plan model."""
        mock_load_cfg.return_value = {"plan": {"model": "claude-sonnet-4"}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="claude-sonnet-4", provider="anthropic")

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            cli_mod.HermesCLI._handle_plan_command(stub, "/plan")

        mock_switch.assert_not_called()

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_no_plan_model_loads_skill_without_switch(self, _cprint, mock_load_cfg):
        """When plan.model is empty, just load the plan skill."""
        mock_load_cfg.return_value = {"plan": {"model": ""}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content",
            ):
                cli_mod.HermesCLI._handle_plan_command(stub, "/plan")

        mock_switch.assert_not_called()
        stub._pending_input.put.assert_called_once()

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_switch_failure_graceful(self, _cprint, mock_load_cfg):
        """When model switch fails, log error but don't crash."""
        mock_load_cfg.return_value = {"plan": {"model": "claude-sonnet-4"}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            mock_result = SimpleNamespace(
                success=False,
                new_model="",
                target_provider="",
                api_key=None,
                base_url=None,
                api_mode=None,
                error_message="Model not found",
                needs_aux=False,
            )
            mock_switch.return_value = mock_result

            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content",
            ):
                result = cli_mod.HermesCLI._handle_plan_command(stub, "/plan")

        self.assertTrue(result)
        self.assertIsNone(stub._plan_original_model)

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_preserves_user_instruction(self, _cprint, mock_load_cfg):
        """User's instruction after /plan is passed to the skill."""
        mock_load_cfg.return_value = {"plan": {"model": ""}}

        cli_mod = _import_cli()
        stub = self._make_cli()

        with patch(
            "agent.skill_commands.build_skill_invocation_message"
        ) as mock_build:
            mock_build.return_value = "plan-skill-content"
            cli_mod.HermesCLI._handle_plan_command(stub, "/plan design the DB schema")

        args, kwargs = mock_build.call_args
        self.assertIn("design the DB schema", str(args) + str(kwargs.values()))

    @patch("hermes_cli.config.load_config")
    @patch("cli._cprint")
    def test_already_in_plan_mode_reloads_skill(self, _cprint, mock_load_cfg):
        """Calling /plan again in plan mode reloads the skill."""
        mock_load_cfg.return_value = {"plan": {"model": "claude-sonnet-4"}}

        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")
        stub._plan_original_model = {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "",
            "api_mode": "",
        }

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            with patch(
                "agent.skill_commands.build_skill_invocation_message",
                return_value="plan-skill-content",
            ):
                cli_mod.HermesCLI._handle_plan_command(stub, "/plan")

        mock_switch.assert_not_called()
        stub._pending_input.put.assert_called_once()


# =============================================================================
# _handle_unplan_command tests
# =============================================================================


class TestHandleUnplanCommand(unittest.TestCase):
    """Tests for HermesCLI._handle_unplan_command()."""

    def _make_cli(self, model="anthropic/claude-sonnet-4", provider="anthropic"):
        cli_mod = _import_cli()
        cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
        cli.model = model
        cli.provider = provider
        cli.api_key = None
        cli.base_url = None
        cli.api_mode = "chat_completions"
        cli._plan_original_model = None
        cli._apply_model_switch_result = MagicMock()
        return cli

    @patch("cli._cprint")
    def test_restores_original_model(self, _cprint):
        """Unplan restores the saved original model."""
        cli_mod = _import_cli()
        stub = self._make_cli(model="anthropic/claude-sonnet-4", provider="anthropic")
        stub._plan_original_model = {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            mock_result = SimpleNamespace(
                success=True,
                new_model="deepseek-chat",
                target_provider="deepseek",
                api_key=None,
                base_url=None,
                api_mode=None,
                error_message=None,
                needs_aux=False,
            )
            mock_switch.return_value = mock_result

            cli_mod.HermesCLI._handle_unplan_command(stub, "/unplan")

        mock_switch.assert_called_once()
        self.assertIsNone(stub._plan_original_model)

    @patch("cli._cprint")
    def test_not_in_plan_mode_does_nothing(self, _cprint):
        """Unplan with no plan state is a no-op."""
        cli_mod = _import_cli()
        stub = self._make_cli()

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            cli_mod.HermesCLI._handle_unplan_command(stub, "/unplan")

        mock_switch.assert_not_called()

    @patch("cli._cprint")
    def test_already_on_original_model_clears_state(self, _cprint):
        """Unplan when already on the original model clears state without switching."""
        cli_mod = _import_cli()
        stub = self._make_cli(model="deepseek-chat", provider="deepseek")
        stub._plan_original_model = {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "",
            "api_mode": "",
        }

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            cli_mod.HermesCLI._handle_unplan_command(stub, "/unplan")

        mock_switch.assert_not_called()
        self.assertIsNone(stub._plan_original_model)

    @patch("cli._cprint")
    def test_restore_failure_clears_state_anyway(self, _cprint):
        """Even if restore fails, clear plan state so user can retry fresh."""
        cli_mod = _import_cli()
        stub = self._make_cli(model="anthropic/claude-sonnet-4", provider="anthropic")
        stub._plan_original_model = {
            "model": "deepseek-chat",
            "provider": "deepseek",
            "api_key": "",
            "base_url": "",
            "api_mode": "",
        }

        with patch("hermes_cli.model_switch.switch_model") as mock_switch:
            mock_result = SimpleNamespace(
                success=False,
                new_model="",
                target_provider="",
                api_key=None,
                base_url=None,
                api_mode=None,
                error_message="Model not found",
                needs_aux=False,
            )
            mock_switch.return_value = mock_result

            cli_mod.HermesCLI._handle_unplan_command(stub, "/unplan")

        self.assertIsNone(stub._plan_original_model)


if __name__ == "__main__":
    unittest.main()
