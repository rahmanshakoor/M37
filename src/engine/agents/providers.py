"""Which model answers — provider choice, keys and the disclosure line.

Why this module exists: stages 5 and 6 must not know how a key is found or which
client class answers; they ask for a :class:`~engine.agents.client.ModelClient` by
provider name and record the disclosure line this module writes. Three providers:
``anthropic`` (the SDK client, :class:`~engine.agents.client.AnthropicClient`),
``openrouter`` (:class:`~engine.agents.openrouter.OpenRouterClient`) and ``fake``
(:class:`~engine.agents.client.FakeClient` answering an empty document — the
pipeline's wiring end to end without a model or a key).

Why keys come from the environment or a dotenv *outside* the checkout: the
repository is shared and versioned; a key inside it is a key committed by accident.
:func:`load_env_file` reads ``$ENGINE_ENV`` if set, else ``<repo>/../.env`` (the
directory above the checkout, where the case file lives), refuses any path inside
the repository, and never overrides a variable the real environment already holds —
so a key exported in the shell wins over the file, as an operator expects.
``ENGINE_ENV=""`` (set, empty) reads no dotenv at all: the switch a CI job flips so
the operator's key beside the checkout cannot turn a scripted run into a paid one.
A pytest session needs no switch: while a test runs pytest exports
``PYTEST_CURRENT_TEST``, and :func:`load_env_file` treats that as ``ENGINE_ENV=""``
unless ``ENGINE_ENV`` names a file — so a plain ``pytest`` beside a real dotenv stays
network-free, and a test that wants the operator-shell behaviour says so by removing
the variable. No key is ever logged: :func:`status` reports presence, never a value.

Why model ids are settled here too (:func:`provider_model`): the stages carry the
Anthropic client's bare default (``claude-opus-5``) in their signatures, and
OpenRouter knows only ``author/slug`` ids. The bare default is the OpenRouter
default; any other bare id under ``openrouter`` is a :class:`ProviderError` before a
request is built, never a 400 from the API — and what the manifest records is what
was sent.

The default provider is ``openrouter`` when ``OPENROUTER_API_KEY`` is set, else
``anthropic`` when ``ANTHROPIC_API_KEY`` is set, else :class:`ProviderError` naming
both. The Anthropic SDK also resolves credentials from an auth token, a profile or
federation variables that this module does not read; ``default_provider(strict=False)``
falls back to ``anthropic`` for that case and lets the SDK decide, which is what the
stage CLIs do when ``--provider`` is omitted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

import anthropic
from pydantic import BaseModel

from engine.agents import client as ac
from engine.agents import openrouter as orc

PROVIDERS = ("anthropic", "openrouter", "fake")
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
"""The variable that carries each provider's key; ``fake`` needs none."""
ENV_PATH_VAR = "ENGINE_ENV"
TEST_SESSION_VAR = "PYTEST_CURRENT_TEST"
"""Exported by pytest for the duration of every test (setup, call, teardown)."""
DOTENV_NAME = ".env"
FAKE_MODEL = "fake-model"
DEFAULT_MODELS = {"anthropic": ac.DEFAULT_MODEL, "openrouter": orc.DEFAULT_MODEL, "fake": FAKE_MODEL}
APP_TITLE = "engine"


class ProviderError(ac.AgentError):
    """No usable provider: a provider whose key is absent, or a model id the
    provider cannot take (:func:`provider_model`)."""


# ------------------------------------------------------------------------- dotenv

def repo_root() -> Path:
    """The checkout: the directory holding ``pyproject.toml`` (``src/engine/agents/`` is
    three levels below it)."""
    return Path(__file__).resolve().parents[3]


def default_env_path(repo: Path | None = None) -> Path:
    """``<repo>/../.env`` — beside the case file, outside the checkout."""
    return (repo or repo_root()).resolve().parent / DOTENV_NAME


def test_session() -> bool:
    """Whether this process is running a pytest test (:data:`TEST_SESSION_VAR`)."""
    return TEST_SESSION_VAR in os.environ


def dotenv_disabled() -> bool:
    """Whether no dotenv is to be read: ``ENGINE_ENV`` set to the empty string — the
    explicit "read no dotenv" a CI job sets — or, with ``ENGINE_ENV`` unset, a pytest
    session (:func:`test_session`), so the operator's key beside a checkout cannot
    turn a scripted test into a paid call whether or not anyone flipped the switch.
    ``ENGINE_ENV`` naming a file is an explicit choice and is honoured either way."""
    if ENV_PATH_VAR in os.environ:
        return os.environ[ENV_PATH_VAR] == ""
    return test_session()


def env_path(path: Path | str | None = None, *, repo: Path | None = None) -> Path:
    """Where the dotenv is read from: an explicit ``path``, else ``$ENGINE_ENV``, else
    :func:`default_env_path`. Raises ``ValueError`` for a path inside the repository."""
    chosen = Path(path) if path is not None else Path(os.environ[ENV_PATH_VAR]) if os.environ.get(ENV_PATH_VAR) \
        else default_env_path(repo)
    chosen = chosen.expanduser().resolve()
    root = (repo or repo_root()).resolve()
    if chosen == root or root in chosen.parents:
        raise ValueError(f"refusing a dotenv inside the repository: {chosen} (keep keys outside {root})")
    return chosen


def load_env_file(path: Path | str | None = None, *, repo: Path | None = None) -> dict[str, str]:
    """Read a dotenv (``KEY=value`` lines; ``export`` prefix, quotes and ``#`` comments
    allowed) and set every variable the real environment does not already hold
    (a variable present but empty counts as absent). Returns the file's variables.
    An explicit path (argument or ``$ENGINE_ENV``) must exist; the default location
    may be absent (then nothing is loaded); ``ENGINE_ENV=""`` — or a pytest session
    with ``ENGINE_ENV`` unset — reads nothing at all unless a ``path`` is given
    (:func:`dotenv_disabled`)."""
    if path is None and dotenv_disabled():
        return {}
    explicit = path is not None or bool(os.environ.get(ENV_PATH_VAR))
    p = env_path(path, repo=repo)
    if not p.is_file():
        if explicit:
            raise FileNotFoundError(f"dotenv not found: {p}")
        return {}
    values = parse_dotenv(p.read_text())
    for name, value in values.items():
        if not os.environ.get(name):  # an empty variable is no key: the file may fill it
            os.environ[name] = value
    return values


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[name] = value
    return out


# ----------------------------------------------------------------------- selection

def key_present(provider: str) -> bool:
    var = KEY_ENV.get(_check(provider))
    return True if var is None else bool(os.environ.get(var))


def default_provider(*, strict: bool = True) -> str:
    """``openrouter`` when its key is set, else ``anthropic`` when its key is set. With
    neither: :class:`ProviderError` naming both, or — ``strict=False`` — ``anthropic``,
    leaving the SDK's own credential lookup (auth token, profile) to succeed or fail."""
    if key_present("openrouter"):
        return "openrouter"
    if key_present("anthropic"):
        return "anthropic"
    if not strict:
        return "anthropic"
    raise ProviderError(no_key_message())


def no_key_message() -> str:
    """What to tell an operator who has no key anywhere the engine looks."""
    return (f"no model key: set {KEY_ENV['openrouter']} or {KEY_ENV['anthropic']} in the environment or in "
            f"{_env_path_hint()}, or pass --provider fake")


MODEL_ENV = "OPENROUTER_MODEL"
"""Overrides the OpenRouter default model (``author/slug``); read from the environment
or the dotenv above the checkout, like the keys."""

OPENROUTER_MODELS: tuple[tuple[str, str], ...] = (
    ("anthropic/claude-opus-5", "Claude Opus 5 — Anthropic"),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro — DeepSeek's strongest"),
    ("moonshotai/kimi-k3", "Kimi K3 — Moonshot's strongest"),
    ("anthropic/claude-sonnet-5", "Claude Sonnet 5 — cheaper Anthropic"),
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash — cheap"),
)
"""Models the UI offers for OpenRouter. All support function tools, a JSON-schema
answer and the reasoning parameter (checked against /api/v1/models); any other
``author/slug`` id can still be typed."""


def default_model(provider: str) -> str:
    name = _check(provider)
    if name == "openrouter":
        override = os.environ.get(MODEL_ENV, "").strip()
        if override:
            if "/" not in override:
                raise ProviderError(f"{MODEL_ENV} must be an OpenRouter id like author/slug, got {override!r}")
            return override
    return DEFAULT_MODELS[name]


def model_choices(provider: str) -> list[dict[str, str]]:
    """``[{"id", "label"}]`` for the UI's model selector, the current default first."""
    name = _check(provider)
    if name != "openrouter":
        return [{"id": default_model(name), "label": default_model(name)}]
    default = default_model(name)
    out = [{"id": default, "label": dict(OPENROUTER_MODELS).get(default, default) + " (default)"}]
    out += [{"id": mid, "label": label} for mid, label in OPENROUTER_MODELS if mid != default]
    return out


def provider_model(provider: str, model: str | None = None) -> str:
    """The model id ``provider`` will actually send — what a request, a manifest and a
    disclosure line must all carry. ``None`` is the provider's default. Under
    ``openrouter`` an id is ``author/slug``: the Anthropic client's bare default
    (``claude-opus-5``, what the stage signatures carry) means the OpenRouter default,
    and any other bare id is a :class:`ProviderError` — OpenRouter would answer it
    with a 400 after the bundle had been sent."""
    name = _check(provider)
    if not model:
        return default_model(name)
    if name == "openrouter" and "/" not in model:
        if model == ac.DEFAULT_MODEL:
            return orc.DEFAULT_MODEL
        raise ProviderError(f"openrouter model ids are author/slug (e.g. {orc.DEFAULT_MODEL}), got {model!r}")
    return model


def resolve(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """What a stage CLI does before anything else: load the dotenv outside the
    repository (the real environment wins; nothing is printed), settle the provider —
    the given one, else :func:`default_provider` non-strict (``anthropic`` when no key
    is present: the SDK has credential sources of its own, and a dry run needs none) —
    and the model (:func:`provider_model`). Raises ``FileNotFoundError``/``ValueError``
    for a dotenv that is missing or inside the checkout, :class:`ProviderError` for a
    model id the provider cannot take."""
    load_env_file()
    name = provider or default_provider(strict=False)
    return name, provider_model(name, model)


def select_client(provider: str | None, model: str | None = None, effort: str = ac.DEFAULT_EFFORT, *,
                  log: Callable[[str], None] | None = None, http: Any = None) -> ac.ModelClient:
    """The client for ``provider`` (``None``: :func:`default_provider`). ``model``
    is settled by :func:`provider_model`; ``http`` injects the OpenRouter transport
    (tests). A missing key or a model id the provider cannot take is a
    :class:`ProviderError`; a constructor the Anthropic SDK refuses is an
    :class:`~engine.agents.client.AgentError`, as the stages expect."""
    name = default_provider() if provider is None else _check(provider)
    model = provider_model(name, model)
    if name == "fake":
        return ac.FakeClient(lambda request: empty_answer(request.output_model), model=model, effort=effort)
    if name == "openrouter":
        key = os.environ.get(KEY_ENV["openrouter"])
        if not key:
            raise ProviderError(f"provider openrouter needs {KEY_ENV['openrouter']} in the environment or in "
                                f"{_env_path_hint()}")
        return orc.OpenRouterClient(key, model=model, effort=effort, app_title=APP_TITLE, http=http, log=log)
    try:
        return ac.AnthropicClient(log=log)
    except (TypeError, anthropic.AnthropicError) as e:
        raise ac.AgentError(f"could not build the Anthropic client: {e}") from e


def disclosure(provider: str, model: str, effort: str) -> str:
    """The provider-disclosure line the manifests and reports carry — for OpenRouter
    it names the pinned upstream and the data-collection setting."""
    name = _check(provider)
    if name == "openrouter":
        return orc.disclosure(model, effort)
    if name == "fake":
        return f"FakeClient (scripted, no API call), model {model}, effort {effort}"
    return ac.disclosure(model, effort)


def status() -> dict[str, dict[str, Any]]:
    """``{provider: {"key_present": bool, "default_model": str}}`` — what the UI shows.
    Presence only; a key's value never leaves the environment."""
    return {name: {"key_present": key_present(name), "default_model": default_model(name), "models": model_choices(name)}
            for name in PROVIDERS}


# ---------------------------------------------------------------------- fake answer

def empty_answer(model: type[BaseModel]) -> dict[str, Any]:
    """The smallest document ``model`` accepts: every required field at its zero
    value (``""``, ``[]``, ``False``, ``0``, ``None``, a nested empty document). What
    the ``fake`` provider answers — enough to drive every stage to its outputs and
    nothing a report could mistake for a finding."""
    return {name: _zero(info.annotation) for name, info in model.model_fields.items() if info.is_required()}


def _zero(annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if isinstance(annotation, type):
            if issubclass(annotation, BaseModel):
                return empty_answer(annotation)
            if annotation is bool:
                return False
            if annotation is int:
                return 0
            if annotation is float:
                return 0.0
            if annotation is str:
                return ""
            if annotation in (list, tuple, set):
                return []
            if annotation is dict:
                return {}
        return None
    if origin in (list, tuple, set, frozenset):
        return []
    if origin is dict:
        return {}
    if type(None) in args:  # Optional[...] / X | None
        return None
    if origin is Literal and args:
        return args[0]
    if args:  # a Union without None: the first member's zero
        return _zero(args[0])
    return None


# ------------------------------------------------------------------------ helpers

def _check(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    return provider


def _env_path_hint() -> str:
    try:
        return str(env_path())
    except ValueError as e:  # $ENGINE_ENV points inside the repo: say so, do not hide it
        return f"a dotenv outside the repository ({e})"
