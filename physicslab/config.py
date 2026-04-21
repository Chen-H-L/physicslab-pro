import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_AI_SETTINGS = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", "").strip(),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip() or "https://api.deepseek.com",
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
}

DEFAULT_SETTINGS = {
    "ai": DEFAULT_AI_SETTINGS,
}


def _settings_path() -> Path:
    base_dir = (
        os.getenv("LOCALAPPDATA")
        or os.getenv("APPDATA")
        or str(Path.home() / ".physicslab_pro")
    )
    return Path(base_dir) / "PhysicsLabPro" / "settings.json"


def _merge_dict(defaults: dict, loaded: dict) -> dict:
    result = deepcopy(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return deepcopy(DEFAULT_SETTINGS)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return deepcopy(DEFAULT_SETTINGS)
        return _merge_dict(DEFAULT_SETTINGS, data)
    except Exception:
        return deepcopy(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_ai_settings() -> dict:
    settings = load_settings()
    ai_settings = settings.get("ai", {})
    merged = _merge_dict(DEFAULT_AI_SETTINGS, ai_settings if isinstance(ai_settings, dict) else {})
    merged["api_key"] = str(merged.get("api_key", "")).strip()
    merged["base_url"] = str(merged.get("base_url", "https://api.deepseek.com")).strip() or "https://api.deepseek.com"
    merged["model"] = str(merged.get("model", "deepseek-chat")).strip() or "deepseek-chat"
    return merged


def save_ai_settings(ai_settings: dict):
    settings = load_settings()
    settings["ai"] = {
        "api_key": str(ai_settings.get("api_key", "")).strip(),
        "base_url": str(ai_settings.get("base_url", "https://api.deepseek.com")).strip() or "https://api.deepseek.com",
        "model": str(ai_settings.get("model", "deepseek-chat")).strip() or "deepseek-chat",
    }
    save_settings(settings)


def get_settings_file_path() -> Path:
    return _settings_path()
