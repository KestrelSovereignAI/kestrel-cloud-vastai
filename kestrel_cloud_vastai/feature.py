"""
Vast.ai GPU Feature for Kestrel agents.

Exposes Vast.ai GPU orchestration via the tool system, providing commands
for searching, starting, stopping, and managing GPU instances.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.result import ToolResult
from kestrel_cloud_vastai.manager import VastAIManager
from kestrel_cloud_vastai.models import InstanceStatus, VastAIManagerError
from kestrel_sdk.llm.types import BackendType
from kestrel_sdk.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class VastAIFeature(Feature):
    """Feature layer exposing Vast.ai GPU orchestration via the tool system."""

    @property
    def tool_description(self) -> str:
        return (
            "Manage Vast.ai GPU instances - search offers, start and stop instances, "
            "check status, and route LLM traffic to remote GPU. Generally cheaper than "
            "RunPod but with variable reliability."
        )

    async def initialize(self):
        """Initialize the Vast.ai manager."""
        self.manager = VastAIManager()
        self.llm_service = getattr(self.agent, "llm_service", None)
        if not self.llm_service:
            logger.warning("LLMService not available; GPU routing disabled")

    @tool(
        name="manage_vastai",
        description="Search, start, stop, or inspect Vast.ai GPU instances (usage: !vastai <action> [...]).",
        category=ToolCategory.SYSTEM,
        command_prefix="!vastai",
    )
    async def manage_vastai(
        self,
        action: str = "status",
        profile: str = "",
        model_name: str = "",
        ttl_seconds: str = "",
        query: str = "",
        limit: str = "5",
    ) -> ToolResult:
        """
        Main entry point for Vast.ai instance management.

        Actions:
            - status: Show current session status
            - search: Search for available GPU offers
            - on/start: Start a new GPU instance
            - off/stop: Stop and destroy current instance
            - list: List all your instances
            - ssh: Get SSH connection URL

        Examples:
            !vastai status
            !vastai search query="gpu_ram >= 24"
            !vastai on profile=training
            !vastai off
        """
        action_normalized = (action or "status").lower()

        if action_normalized in {"status"}:
            return await self._status()

        if action_normalized in {"search", "offers"}:
            return await self._search(
                profile_name=profile or None,
                query=query or None,
                limit=self._coerce_optional_int(limit) or 5,
            )

        if action_normalized in {"on", "start"}:
            return await self._start(
                profile_name=profile,
                model_name=model_name,
                ttl_seconds=ttl_seconds,
            )

        if action_normalized in {"off", "stop"}:
            return await self._stop()

        if action_normalized in {"list", "instances"}:
            return await self._list_instances()

        if action_normalized in {"ssh", "ssh-url"}:
            return await self._get_ssh()

        return ToolResult.failed(
            f"Unsupported Vast.ai action: {action}. "
            "Use: status, search, on, off, list, ssh",
            data={
                "available_actions": ["status", "search", "on", "off", "list", "ssh"],
            },
        )

    async def _status(self) -> ToolResult:
        """Get current session status."""
        try:
            status = await self.manager.get_status()
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))
        return ToolResult.ok(
            confirmation=f"Vast.ai session status: {status.get('status', 'unknown')}",
            data={
                "action": "status",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _search(
        self,
        profile_name: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> ToolResult:
        """Search for available GPU offers."""
        profile = None
        if profile_name:
            profile = self.manager.profiles.get(profile_name)
            if not profile:
                # Validation failure → ToolResult.failed (NOT an
                # exception). With manage_vastai advertised as
                # returning ToolResult, the public surface contract
                # requires user-error paths to land in the envelope
                # so the audit hook (#1042 layer 3) can read them.
                available = list(self.manager.profiles.keys())
                return ToolResult.failed(
                    f"Unknown profile '{profile_name}'. Available: {available}",
                    data={"available_profiles": available},
                )

        try:
            offers = await self.manager.search_offers(
                profile=profile,
                query=query,
                limit=limit,
            )
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))

        # Format offers for display
        formatted = []
        for offer in offers:
            formatted.append({
                "id": offer.get("id"),
                "gpu": offer.get("gpu_name"),
                "gpu_ram": offer.get("gpu_ram"),
                "num_gpus": offer.get("num_gpus"),
                "price_per_hr": offer.get("dph_total"),
                "reliability": round(offer.get("reliability", 0), 3),
                "cuda": offer.get("cuda_max_good"),
                "location": offer.get("geolocation"),
            })

        return ToolResult.ok(
            confirmation=f"Found {len(formatted)} Vast.ai offer(s)",
            data={
                "action": "search",
                "query": query or f"profile:{profile_name}",
                "count": len(formatted),
                "offers": formatted,
            },
        )

    async def _start(
        self,
        profile_name: str,
        model_name: str,
        ttl_seconds: str,
    ) -> ToolResult:
        """Start a new GPU instance."""
        # Validation failures → ToolResult.failed (NOT an exception).
        # See _search docstring + #1042 layer 4b honesty contract.
        available = list(self.manager.profiles.keys())
        if not profile_name:
            return ToolResult.failed(
                f"Profile required. Available: {available}",
                data={
                    "available_profiles": available,
                    "usage": "!vastai on profile=<name>",
                },
            )
        if profile_name not in self.manager.profiles:
            # Pre-flight check: matches the validation _search does.
            # Without this guard, manager.start_session(...) raises
            # VastAIManagerError mid-flight and the audit hook can't
            # see a structured ToolResult.ERROR for what is, from the
            # user's perspective, just a typo.
            return ToolResult.failed(
                f"Unknown profile '{profile_name}'. Available: {available}",
                data={"available_profiles": available},
            )

        ttl = self._coerce_optional_int(ttl_seconds)
        target_model = model_name or None

        env_overrides = {
            "KESTREL_PROFILE": profile_name,
        }
        if target_model:
            env_overrides["TARGET_MODEL"] = target_model

        metadata = {
            "label": f"kestrel-{profile_name}-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            "env_overrides": env_overrides,
        }

        try:
            status = await self.manager.start_session(
                task_profile=profile_name,
                model_name=target_model,
                ttl_seconds=ttl,
                metadata=metadata,
            )
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))

        # Attach to LLM router if this is an LLM profile
        if profile_name in {"llm", "ollama"} and status.get("active"):
            self._attach_gpu_backend(status)

        return ToolResult.ok(
            confirmation=f"Started Vast.ai session (profile: {profile_name})",
            data={
                "action": "start",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _stop(self) -> ToolResult:
        """Stop and destroy current instance.

        The manager returns ``{"active": False, "status": "offline"}``
        when there was no session to stop (no-op) and
        ``{"active": False, "status": "terminated"}`` when an active
        session was actually stopped. The confirmation must reflect
        which happened — saying "Stopped Vast.ai session" on the
        no-op path is exactly the #1042 confident-lie failure mode
        (claiming an action happened when it didn't).
        """
        try:
            status = await self.manager.stop_session()
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))
        self._detach_gpu_backend("Requested via !vastai off")
        # "offline" status from stop_session = there was no session
        # to stop; anything else (typically "terminated") = we did
        # stop one.
        was_no_op = (
            status.get("status") == "offline"
            and not status.get("active")
        )
        confirmation = (
            "No active Vast.ai session to stop (no-op)"
            if was_no_op
            else "Stopped Vast.ai session"
        )
        return ToolResult.ok(
            confirmation=confirmation,
            data={
                "action": "stop",
                "session": status,
                "router": self._router_status(),
            },
        )

    async def _list_instances(self) -> ToolResult:
        """List all instances for this account."""
        try:
            instances = await self.manager.show_instances()
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))

        formatted = []
        for inst in instances:
            formatted.append({
                "id": inst.get("id"),
                "status": inst.get("actual_status"),
                "gpu": inst.get("gpu_name"),
                "price_per_hr": inst.get("dph_total"),
                "label": inst.get("label"),
                "ssh_host": inst.get("ssh_host"),
                "ssh_port": inst.get("ssh_port"),
            })

        return ToolResult.ok(
            confirmation=f"Listed {len(formatted)} Vast.ai instance(s)",
            data={
                "action": "list",
                "count": len(formatted),
                "instances": formatted,
            },
        )

    async def _get_ssh(self) -> ToolResult:
        """Get SSH connection URL for current session."""
        try:
            ssh_url = await self.manager.get_ssh_url()
        except VastAIManagerError as e:
            return ToolResult.failed(str(e))
        return ToolResult.ok(
            confirmation=(
                f"SSH ready: ssh {ssh_url}" if ssh_url else "No active session"
            ),
            data={
                "action": "ssh",
                "ssh_url": ssh_url,
                "hint": f"Connect with: ssh {ssh_url}" if ssh_url else "No active session",
            },
        )

    def _attach_gpu_backend(self, session_status: Dict[str, Any]) -> None:
        """Attach Vast.ai instance to LLM router."""
        if not self.llm_service:
            logger.warning("Cannot attach GPU backend without LLMService")
            return

        base_url = session_status.get("inference_url")
        if not base_url:
            logger.warning("Session missing inference URL; skipping activation")
            return

        remaining = session_status.get("remaining_ttl_seconds")
        profile_id = session_status.get("profile")
        profile = self.manager.profiles.get(profile_id) if profile_id else None

        config = {
            "base_url": base_url,
            "model": session_status.get("model_name"),
            "ttl_seconds": remaining,
            "metadata": {
                "instance_id": session_status.get("instance_id"),
                "profile": profile_id,
                "provider": "vastai",
                "gpu_name": session_status.get("gpu_name"),
                "cost_per_hr": session_status.get("actual_cost_per_hr"),
            },
        }
        self.llm_service.switch_backend(BackendType.REMOTE_GPU, config=config)

    def _detach_gpu_backend(self, reason: str) -> None:
        """Detach GPU backend from LLM router."""
        if self.llm_service:
            self.llm_service._deactivate_remote_backend(reason=reason)

    def _router_status(self) -> Optional[Dict[str, Any]]:
        """Get LLM router status."""
        if not self.llm_service:
            return None
        return self.llm_service.get_backend_status()

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        """Coerce value to optional int."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        if not text.isdigit():
            raise ValueError("Value must be an integer")
        return int(text)
