"""Project-local configuration helpers."""

import os
import re
from pathlib import Path


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_quotes(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        raise ValueError(f"invalid .env line: {line}")
    name, value = line.split("=", 1)
    name = name.strip()
    if not ENV_KEY_PATTERN.match(name):
        raise ValueError(f"invalid .env variable name: {name}")
    return name, _strip_quotes(value)


def global_env_path():
    """Return the user-level .env path, or None when it cannot be resolved.

    PICO_GLOBAL_ENV takes precedence, followed by %APPDATA%\\pico\\.env on
    Windows and $XDG_CONFIG_HOME/pico/.env (or ~/.config/pico/.env) elsewhere.
    """
    override = os.environ.get("PICO_GLOBAL_ENV")
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override)))

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        return Path(appdata) / "pico" / ".env"

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return base / "pico" / ".env"


def load_env_file(env_path, override=True):
    if env_path is None:
        return {}
    env_path = Path(env_path)
    if not env_path.exists():
        return {}

    loaded = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def load_global_env(override=False):
    """Load user-level defaults without overriding existing OS variables."""
    return load_env_file(global_env_path(), override=override)


def find_project_env(start):
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def load_project_env(start, override=True):
    return load_env_file(find_project_env(start), override=override)


def load_config_env(start):
    """Load global defaults first, then let the nearest project .env win."""
    loaded = load_global_env(override=False)
    loaded.update(load_project_env(start, override=True))
    return loaded


def provider_env(name, legacy_names=(), default=""):
    for env_name in (name, *legacy_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return default
