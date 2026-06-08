"""Cover the LLM context-type (KV-cache quantization) wiring.

Three layers: the launch-arg builder that turns a context-type preset into
llama-server flags, the stderr ring-buffer that surfaces a failed launch, and
the ``/api/llm/config`` endpoint that drives it from the UI.
"""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient
import pytest

from app.backends.base import ModelDescriptor
from app.backends.llm_llamacpp import LlamaCppBackend
from app.config import settings
from app.core.enums import ModelFamily
from app.main import app


def _backend() -> LlamaCppBackend:
    desc = ModelDescriptor(
        id="stub-llm",
        name="stub-llm",
        family=ModelFamily.GGUF,
        path=Path("model.gguf"),
        size_bytes=4,
    )
    return LlamaCppBackend(desc)


@pytest.fixture
def restore_llm_settings():
    """Snapshot/restore the global llama knobs so tests don't leak into each
    other (settings is a process-wide singleton)."""
    original = (settings.llama_context_type, settings.llama_backend)
    yield
    settings.llama_context_type, settings.llama_backend = original


# --------------------------------------------------------------- build args
def test_f16_emits_no_cache_or_flash_flags(restore_llm_settings):
    settings.llama_context_type = "f16"
    args = _backend()._build_server_args()
    assert "--cache-type-k" not in args
    assert "--cache-type-v" not in args
    assert "--flash-attn" not in args
    # ctx/ngl are still wired regardless of context type.
    assert "-c" in args and "-ngl" in args


def test_q8_0_adds_cache_types_and_flash_attn(restore_llm_settings):
    settings.llama_context_type = "q8_0"
    args = _backend()._build_server_args()
    k = args.index("--cache-type-k")
    v = args.index("--cache-type-v")
    assert args[k + 1] == "q8_0"
    assert args[v + 1] == "q8_0"
    # Quantized V cache requires flash-attention.
    fa = args.index("--flash-attn")
    assert args[fa + 1] == "on"


def test_turbo3_maps_to_turbo_cache_type(restore_llm_settings):
    settings.llama_context_type = "turbo3"
    args = _backend()._build_server_args()
    assert args[args.index("--cache-type-k") + 1] == "turbo3"
    assert args[args.index("--cache-type-v") + 1] == "turbo3"
    assert "--flash-attn" in args


def test_unknown_context_type_falls_back_to_f16(restore_llm_settings):
    settings.llama_context_type = "does-not-exist"
    args = _backend()._build_server_args()
    # resolve_context_type defaults unknown names to f16 -> no quant flags.
    assert "--cache-type-k" not in args


def test_turbo_backend_uses_its_own_binary(restore_llm_settings):
    settings.llama_backend = "turbo"
    args = _backend()._build_server_args()
    # argv[0] is the TurboQuant build's path, not the standard one.
    assert args[0] == str(settings.llama_server_bin_turbo)
    assert args[0] != str(settings.llama_server_bin)


def test_default_backend_uses_standard_binary(restore_llm_settings):
    settings.llama_backend = "default"
    args = _backend()._build_server_args()
    assert args[0] == str(settings.llama_server_bin)


# --------------------------------------------------------------- stderr tail
class _FakeStderr:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __aiter__(self) -> _FakeStderr:
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, stderr: _FakeStderr) -> None:
        self.stderr = stderr
        self.returncode = 1


async def test_drain_stderr_captures_tail_for_diagnostics():
    backend = _backend()
    backend._proc = _FakeProc(  # type: ignore[assignment]
        _FakeStderr([b"loading model\n", b"error: unknown cache type 'turbo3'\n"])
    )
    await backend._drain_stderr()
    detail = backend._stderr_detail()
    assert "unknown cache type 'turbo3'" in detail
    assert "llama-server stderr" in detail


async def test_drain_stderr_tail_is_bounded():
    backend = _backend()
    backend._proc = _FakeProc(  # type: ignore[assignment]
        _FakeStderr([f"line {i}\n".encode() for i in range(200)])
    )
    await backend._drain_stderr()
    # maxlen=40 ring buffer keeps only the most recent lines.
    assert len(backend._stderr_tail) == 40
    assert "line 199" in backend._stderr_tail[-1]


def test_empty_tail_yields_no_detail():
    assert _backend()._stderr_detail() == ""


# ------------------------------------------------------------------- the API
@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def test_config_exposes_context_types(client, restore_llm_settings):
    body = (await client.get("/api/llm/config")).json()
    assert "context_type" in body
    ids = {ct["id"] for ct in body["context_types"]}
    assert {"f16", "q8_0", "turbo3", "turbo4"} <= ids
    # TurboQuant presets are flagged experimental for the UI warning.
    turbo = next(ct for ct in body["context_types"] if ct["id"] == "turbo3")
    assert turbo["experimental"] is True
    f16 = next(ct for ct in body["context_types"] if ct["id"] == "f16")
    assert f16["experimental"] is False


async def test_config_exposes_backends(client, restore_llm_settings):
    body = (await client.get("/api/llm/config")).json()
    assert body["backend"] == "default"
    backends = {b["id"]: b for b in body["backends"]}
    assert {"default", "turbo"} <= backends.keys()
    # Each backend advertises which context types it can run + binary availability.
    assert "turbo3" in backends["turbo"]["context_types"]
    assert "turbo3" not in backends["default"]["context_types"]
    assert "available" in backends["default"] and "path" in backends["default"]


async def test_set_context_type_persists(client, restore_llm_settings):
    body = (await client.post("/api/llm/config", json={"context_type": "q8_0"})).json()
    assert body["changed"] is True
    assert body["context_type"] == "q8_0"
    assert settings.llama_context_type == "q8_0"


async def test_set_same_context_type_is_noop(client, restore_llm_settings):
    settings.llama_context_type = "f16"
    body = (await client.post("/api/llm/config", json={"context_type": "f16"})).json()
    assert body["changed"] is False


async def test_unknown_context_type_rejected(client, restore_llm_settings):
    before = settings.llama_context_type
    resp = await client.post("/api/llm/config", json={"context_type": "nope"})
    assert resp.status_code == 422
    # A rejected value must not mutate the live setting.
    assert settings.llama_context_type == before


async def test_unknown_backend_rejected(client, restore_llm_settings):
    before = settings.llama_backend
    resp = await client.post("/api/llm/config", json={"backend": "nope"})
    assert resp.status_code == 422
    assert settings.llama_backend == before


async def test_turbo_type_rejected_on_default_backend(client, restore_llm_settings):
    settings.llama_backend = "default"
    settings.llama_context_type = "f16"
    resp = await client.post("/api/llm/config", json={"context_type": "turbo3"})
    assert resp.status_code == 422
    # Rejected pairing leaves the live settings untouched.
    assert settings.llama_context_type == "f16"


async def test_turbo_backend_plus_turbo_type_in_one_request(client, restore_llm_settings):
    body = (await client.post(
        "/api/llm/config", json={"backend": "turbo", "context_type": "turbo3"}
    )).json()
    assert body["backend"] == "turbo"
    assert body["context_type"] == "turbo3"
    assert settings.llama_backend == "turbo"
    assert settings.llama_context_type == "turbo3"


async def test_backend_switch_resets_unsupported_context_type(client, restore_llm_settings):
    # Start in a valid turbo+turbo3 state...
    settings.llama_backend = "turbo"
    settings.llama_context_type = "turbo3"
    # ...then switch to the standard backend, which can't run turbo3.
    body = (await client.post("/api/llm/config", json={"backend": "default"})).json()
    assert body["backend"] == "default"
    # Graceful fallback rather than an error or an unlaunchable state.
    assert body["context_type"] == "f16"
    assert body["note"] and "reset" in body["note"]
    assert settings.llama_context_type == "f16"
