"""Direct contracts for the Vast.ai feature surface.

The public ``manage_vastai`` tool and its private dispatch helpers
all return ``kestrel_sdk.tools.result.ToolResult``; these tests pin
the success and failure shapes so the framework's narration-honesty
audit hook (kestrel-sovereign issue #1042 layer 3) can trust the
wire format.

The ``test_vastai_e2e.py`` suite hits the real Vast.ai API and
asserts on the underlying ``VastAIManager`` dict shape. Those are
the manager's contract, not the feature's @tool surface — they
stay as-is and live behind the ``cloud_resource`` marker.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_cloud_vastai.feature import VastAIFeature
from kestrel_cloud_vastai.models import VastAIManagerError
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus


def _make_feature() -> VastAIFeature:
    feature = VastAIFeature(agent=SimpleNamespace())
    feature.manager = SimpleNamespace(
        profiles={"training": object(), "budget": object(), "llm": object()},
        get_status=AsyncMock(),
        start_session=AsyncMock(),
        stop_session=AsyncMock(),
        search_offers=AsyncMock(return_value=[]),
        show_instances=AsyncMock(return_value=[]),
        get_ssh_url=AsyncMock(),
    )
    feature.llm_service = MagicMock()
    # _router_status reads attributes that may not exist on the bare
    # SimpleNamespace; stub it to a plain dict so the helpers can
    # compose their data payload.
    feature._router_status = MagicMock(return_value={"backend": "cloud"})
    feature._attach_gpu_backend = MagicMock()
    feature._detach_gpu_backend = MagicMock()
    return feature


@pytest.mark.asyncio
async def test_manage_vastai_unknown_action_returns_failed():
    feature = _make_feature()

    result = await feature.manage_vastai(action="dance")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Unsupported Vast.ai action" in result.error
    assert result.data["available_actions"] == [
        "status", "search", "on", "off", "list", "ssh",
    ]


@pytest.mark.asyncio
async def test_status_returns_ok_with_session_data():
    feature = _make_feature()
    feature.manager.get_status.return_value = {
        "active": False,
        "status": "offline",
    }

    result = await feature._status()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "Vast.ai session status: offline" in result.confirmation
    assert result.data["action"] == "status"
    assert result.data["session"] == {"active": False, "status": "offline"}
    assert "router" in result.data


@pytest.mark.asyncio
async def test_search_returns_ok_with_offers_and_count():
    feature = _make_feature()
    feature.manager.search_offers.return_value = [
        {
            "id": 42,
            "gpu_name": "RTX 3090",
            "gpu_ram": 24,
            "num_gpus": 1,
            "dph_total": 0.4,
            "reliability": 0.987,
            "cuda_max_good": "12.4",
            "geolocation": "US",
        }
    ]

    result = await feature._search(query="gpu_ram >= 24", limit=5)

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "Found 1 Vast.ai offer(s)" in result.confirmation
    assert result.data["count"] == 1
    assert result.data["offers"][0]["gpu"] == "RTX 3090"


@pytest.mark.asyncio
async def test_start_returns_ok_with_session_data():
    feature = _make_feature()
    feature.manager.start_session.return_value = {
        "active": True,
        "status": "running",
        "instance_name": "vastai-1",
        "inference_url": "http://gpu.example/v1",
        "profile": "training",
    }

    result = await feature._start(
        profile_name="training",
        model_name="phi4",
        ttl_seconds="900",
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "training" in result.confirmation
    assert result.data["action"] == "start"
    assert result.data["session"]["instance_name"] == "vastai-1"


@pytest.mark.asyncio
async def test_stop_returns_ok_and_detaches_gpu_backend():
    feature = _make_feature()
    feature.manager.stop_session.return_value = {"status": "terminated"}

    result = await feature._stop()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "Stopped Vast.ai session" in result.confirmation
    assert result.data["session"] == {"status": "terminated"}
    feature._detach_gpu_backend.assert_called_once_with(
        "Requested via !vastai off"
    )


@pytest.mark.asyncio
async def test_list_instances_returns_ok_with_count():
    feature = _make_feature()
    feature.manager.show_instances.return_value = [
        {
            "id": 1,
            "actual_status": "running",
            "gpu_name": "RTX 4090",
            "dph_total": 0.6,
            "label": "kestrel-training",
            "ssh_host": "ssh.example",
            "ssh_port": 22,
        }
    ]

    result = await feature._list_instances()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "Listed 1 Vast.ai instance(s)" in result.confirmation
    assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_get_ssh_returns_ok_with_url():
    feature = _make_feature()
    feature.manager.get_ssh_url.return_value = "user@ssh.example:22"

    result = await feature._get_ssh()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "user@ssh.example:22" in result.confirmation
    assert result.data["ssh_url"] == "user@ssh.example:22"


@pytest.mark.asyncio
async def test_get_ssh_no_session_returns_ok_with_none_url():
    """When there's no active session, the SSH command can't be
    constructed but the call itself isn't an error — the user just
    needs to start a session first. Stays as ToolResult.ok with a
    None ssh_url and a hint."""
    feature = _make_feature()
    feature.manager.get_ssh_url.return_value = None

    result = await feature._get_ssh()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "No active session" in result.confirmation
    assert result.data["ssh_url"] is None


# ---------------------------------------------------------------------------
# Validation failure paths (#1042 codex round-1 catches)
#
# These tests pin the contract that user-error paths land in the
# ``ToolResult.failed`` envelope, NOT as raised exceptions. If a
# helper escapes the contract via ``raise``, the framework's
# narration audit hook (#1042 layer 3) cannot see the failure to
# block a confident-lie LLM reply.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_unknown_profile_returns_failed_not_raises():
    """A profile name the manager doesn't know is a user error, not
    an exception. The new contract requires ToolResult.failed."""
    feature = _make_feature()

    result = await feature._search(profile_name="not-a-real-profile")

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "not-a-real-profile" in result.error
    assert "available_profiles" in result.data


@pytest.mark.asyncio
async def test_start_without_profile_returns_failed_not_raises():
    """Empty profile is a user error, not an exception. The new
    contract requires ToolResult.failed."""
    feature = _make_feature()

    result = await feature._start(
        profile_name="",
        model_name="",
        ttl_seconds="",
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "Profile required" in result.error
    assert result.data["available_profiles"] == ["training", "budget", "llm"]
    assert result.data["usage"] == "!vastai on profile=<name>"


# ---------------------------------------------------------------------------
# Honesty: stop on no active session must NOT narrate "Stopped …"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_with_no_active_session_returns_no_op_confirmation():
    """When ``!vastai off`` runs with no active session, the manager
    returns ``{"active": False, "status": "offline"}``. Saying
    "Stopped Vast.ai session" in the confirmation is the #1042
    confident-lie failure mode (claiming an action happened when it
    didn't). The confirmation must reflect the no-op."""
    feature = _make_feature()
    feature.manager.stop_session.return_value = {
        "active": False,
        "status": "offline",
    }

    result = await feature._stop()

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.OK
    assert "no-op" in result.confirmation.lower()
    assert "Stopped Vast.ai session" not in result.confirmation, (
        "regression of #1042 honesty fix: confirmation claims an "
        "action happened when there was nothing to stop"
    )


# ---------------------------------------------------------------------------
# Manager-error escape hatches (#1042 codex round-2 catches)
#
# Every helper that calls into the underlying VastAIManager wraps
# the call with a try/except that converts VastAIManagerError into
# ToolResult.failed. Without that conversion the error surfaces
# through the SDK DynamicTool wrapper as the legacy
# ``{success: False, error: ...}`` shape WITHOUT a ``status`` key,
# which the audit hook can't read.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_unknown_profile_returns_failed_pre_flight():
    """The pre-flight profile-existence check catches unknown
    profile names before they reach manager.start_session, where a
    raise would escape the envelope."""
    feature = _make_feature()

    result = await feature._start(
        profile_name="not-a-real-profile",
        model_name="",
        ttl_seconds="",
    )

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "not-a-real-profile" in result.error
    assert "Unknown profile" in result.error
    assert result.data["available_profiles"] == ["training", "budget", "llm"]
    feature.manager.start_session.assert_not_awaited()


@pytest.mark.parametrize(
    "method,attr,kwargs",
    [
        ("_status", "get_status", {}),
        ("_search", "search_offers", {"profile_name": None, "limit": 5}),
        ("_start", "start_session", {
            "profile_name": "training",
            "model_name": "",
            "ttl_seconds": "",
        }),
        ("_stop", "stop_session", {}),
        ("_list_instances", "show_instances", {}),
        ("_get_ssh", "get_ssh_url", {}),
    ],
)
@pytest.mark.asyncio
async def test_helper_wraps_manager_error_in_tool_result(method, attr, kwargs):
    """Every helper that calls into VastAIManager catches
    VastAIManagerError and converts it to ToolResult.failed. Without
    this guard, the manager's raise escapes the envelope and the
    audit hook can't see a structured ToolResult.ERROR."""
    feature = _make_feature()
    getattr(feature.manager, attr).side_effect = VastAIManagerError("boom")

    result = await getattr(feature, method)(**kwargs)

    assert isinstance(result, ToolResult)
    assert result.status is ToolResultStatus.ERROR
    assert "boom" in result.error
