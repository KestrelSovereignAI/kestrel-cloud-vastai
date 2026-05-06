# kestrel-cloud-vastai

Vast.ai GPU marketplace provider for Kestrel Sovereign agents. Search the marketplace, provision spot/on-demand GPU instances, run training over SSH, manage lifecycle.

## Installation

```bash
uv pip install kestrel-cloud-vastai
```

The feature is auto-discovered by Kestrel Sovereign via the `kestrel_sovereign.features` entry point — install it alongside `kestrel-sovereign` and `VastAIFeature` registers itself at startup.

## Configuration

| Variable | Description |
|----------|-------------|
| `VASTAI_API_KEY` | Vast.ai API key (required) |

Optional `[vastai]` section in `kestrel.toml` for default profile preferences.

## What's provided

- `VastAIFeature` — agent-facing tools for instance search, provisioning, SSH training, lifecycle
- Standalone API: `VastAIManager` for direct programmatic use
- HTTP API + SSH-training helpers

## Dependencies

- `kestrel-sovereign-sdk>=0.2,<1` — base `Feature`, `tool`, `ToolCategory`, `BackendType`
- `kestrel-sovereign>=0.6,<1` — `kestrel.toml` unified-config loader (runtime)
- `vastai-sdk>=0.1.0`

## Development

```bash
uv pip install -e '.[test]'
uv run pytest
```

## License

Apache-2.0
