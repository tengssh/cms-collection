import os
import sys
import pytest
from unittest.mock import MagicMock, patch

import api_chat
from api_chat import (
    ProviderConfig,
    DEFAULT_PROVIDERS,
    list_providers,
    get_provider_config,
    register_provider,
    get_api_key,
    resolve_provider_and_model,
    unwrap_to_string,
    cms_collection_search,
    call_chat_completion,
)


# ==========================================
# 1. Tests for Provider Selection & Registry
# ==========================================

def test_list_providers():
    providers = list_providers()
    assert "openrouter" in providers
    assert "nvidia" in providers


def test_get_provider_config_valid():
    or_config = get_provider_config("openrouter")
    assert or_config.name == "openrouter"
    assert or_config.base_url == "https://openrouter.ai/api/v1"
    assert or_config.api_key_env == "OPENROUTER_API_KEY"
    assert or_config.default_model == "nvidia/nemotron-3.5-super-vl:free"
    assert or_config.client_type == "openai"

    nv_config = get_provider_config("nvidia")
    assert nv_config.name == "nvidia"
    assert nv_config.base_url == "https://integrate.api.nvidia.com/v1"
    assert nv_config.api_key_env == "NVIDIA_API_KEY"
    assert nv_config.default_model == "mistralai/mistral-large-4-1t-a52b"
    assert nv_config.client_type == "openai"


def test_get_provider_config_aliases_and_case():
    # Case insensitivity
    assert get_provider_config("NVIDIA").name == "nvidia"
    assert get_provider_config("OpenRouter").name == "openrouter"

    # Aliases
    assert get_provider_config("nim").name == "nvidia"
    assert get_provider_config("nvidia-nim").name == "nvidia"
    assert get_provider_config("open-router").name == "openrouter"
    assert get_provider_config("or").name == "openrouter"


def test_get_provider_config_unknown():
    with pytest.raises(ValueError) as excinfo:
        get_provider_config("unknown_provider_xyz")
    assert "Supported providers:" in str(excinfo.value)
    assert "openrouter" in str(excinfo.value)
    assert "nvidia" in str(excinfo.value)


def test_register_provider_extensibility():
    new_provider = ProviderConfig(
        name="custom_nim",
        base_url="https://custom.nim.local/v1",
        api_key_env="CUSTOM_NIM_KEY",
        default_model="custom/llama-3",
        aliases=["cnim"],
        description="Self-hosted custom NIM endpoint",
    )
    register_provider(new_provider)

    assert "custom_nim" in list_providers()
    retrieved = get_provider_config("cnim")
    assert retrieved.name == "custom_nim"
    assert retrieved.base_url == "https://custom.nim.local/v1"


# ==========================================
# 2. Tests for Resolving Provider and Model
# ==========================================

def test_resolve_provider_and_model_defaults():
    with patch.dict(os.environ, {}, clear=True):
        config, model = resolve_provider_and_model()
        assert config.name == "openrouter"
        assert model == config.default_model


def test_resolve_provider_and_model_explicit_args():
    config, model = resolve_provider_and_model("nvidia", "custom/model-x")
    assert config.name == "nvidia"
    assert model == "custom/model-x"


def test_resolve_provider_and_model_from_env():
    env_vars = {
        "API_PROVIDER": "nvidia",
        "API_MODEL": "meta/llama-3.1-70b-instruct",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config, model = resolve_provider_and_model()
        assert config.name == "nvidia"
        assert model == "meta/llama-3.1-70b-instruct"


# ==========================================
# 3. Tests for API Key Handling
# ==========================================

def test_get_api_key_success():
    config = get_provider_config("nvidia")
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "nv-test-key-123"}):
        assert get_api_key(config) == "nv-test-key-123"


def test_get_api_key_missing():
    config = get_provider_config("nvidia")
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            get_api_key(config)
        assert "NVIDIA_API_KEY" in str(excinfo.value)


# ==========================================
# 4. Tests for Chat Completions (Requests & OpenAI)
# ==========================================

def test_call_chat_completion_requests_fallback():
    config = get_provider_config("nvidia")
    messages = [{"role": "user", "content": "Hello NVIDIA"}]

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from NIM via requests fallback!",
                }
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "nv-dummy-key"}), \
         patch("api_chat.HAS_OPENAI", False), \
         patch("requests.post", return_value=mock_resp) as mock_post:
        res = call_chat_completion(config, "mistralai/mistral-large-4-1t-a52b", messages)

        assert res["choices"][0]["message"]["content"] == "Hello from NIM via requests fallback!"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer nv-dummy-key"
        assert call_kwargs[1]["json"]["model"] == "mistralai/mistral-large-4-1t-a52b"


def test_call_chat_completion_rate_limit_retry():
    config = get_provider_config("openrouter")
    messages = [{"role": "user", "content": "Hello"}]

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "2"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Recovered from 429!"}}]
    }
    resp_200.raise_for_status.return_value = None

    status_updates = []

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-dummy-key"}), \
         patch("api_chat.HAS_OPENAI", False), \
         patch("requests.post", side_effect=[resp_429, resp_200]) as mock_post, \
         patch("time.sleep") as mock_sleep:
        res = call_chat_completion(
            config,
            "liquid/lfm-2.5-2.6b:free",
            messages,
            on_retry=lambda msg: status_updates.append(msg),
        )

        assert res["choices"][0]["message"]["content"] == "Recovered from 429!"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(3)
        assert any("Rate limited" in s for s in status_updates)


def test_call_chat_completion_openai_client_both_providers():
    for provider_name, expected_url, key_env, dummy_key, model in [
        ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "or-dummy-key", "nvidia/nemotron-3.5-super-vl:free"),
        ("nvidia", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "nv-dummy-key", "mistralai/mistral-large-4-1t-a52b"),
    ]:
        config = get_provider_config(provider_name)
        messages = [{"role": "user", "content": f"Hello {provider_name}"}]

        mock_msg = MagicMock()
        mock_msg.role = "assistant"
        mock_msg.content = f"Hello from {provider_name}!"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict(os.environ, {key_env: dummy_key}), \
             patch("api_chat.HAS_OPENAI", True), \
             patch("api_chat.OpenAI", mock_openai_cls):
            res = call_chat_completion(config, model, messages)

            call_kwargs = mock_openai_cls.call_args[1]
            assert call_kwargs["base_url"] == expected_url
            assert call_kwargs["api_key"] == dummy_key
            assert call_kwargs["timeout"] == 60.0
            assert call_kwargs["max_retries"] == 1
            assert res["choices"][0]["message"]["content"] == f"Hello from {provider_name}!"


# ==========================================
# 5. Tests for Query Unwrapping & Chroma Search
# ==========================================

def test_unwrap_to_string_in_api_chat():
    assert unwrap_to_string("simple") == "simple"
    assert unwrap_to_string({"query": "quantum mechanics"}) == "quantum mechanics"
    assert unwrap_to_string([{"query": "first in list"}]) == "first in list"


def test_cms_collection_search_mocked():
    mock_results = {
        "ids": ["id_1"],
        "documents": ["Section: Modeling | Resource: ToolA | Description: Great tool"],
        "metadatas": [{"url": "https://example.com/toola", "tags": "sim,dft"}],
        "scores": [0.88],
        "score_name": "Distance",
    }
    with patch("api_chat.get_shared_collection") as mock_get_coll, \
         patch("api_chat.search_collection", return_value=mock_results) as mock_search:
        res = cms_collection_search("ToolA")
        assert "**ToolA**" in res
        assert "Section: Modeling" in res
        assert "URL: https://example.com/toola" in res
        assert "Tags: sim,dft" in res


def test_sanitize_messages():
    from api_chat import sanitize_messages
    raw = [
        {"role": "assistant", "content": None},
        {"role": "user", "content": "hello"},
    ]
    cleaned = sanitize_messages(raw)
    assert cleaned[0]["content"] == "No information is found in the search results."

    # Test dangling tool call repair
    dangling = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_99", "function": {"name": "test_fn"}}],
        },
        {"role": "user", "content": "next query"},
    ]
    repaired = sanitize_messages(dangling)
    roles = [m["role"] for m in repaired]
    assert roles == ["assistant", "tool", "user"]
    assert repaired[0]["content"] == ""
    assert repaired[1]["tool_call_id"] == "call_99"


def test_sanitize_empty_assistant_message_content():
    from api_chat import sanitize_messages
    msgs = [
        {"role": "user", "content": "What model are you using?"},
        {"role": "assistant", "content": ""},  # empty assistant content
        {"role": "user", "content": "Where to find 2d materials?"},
    ]
    cleaned = sanitize_messages(msgs)
    # NVIDIA NIM rejects content="" with 400 Bad Request; sanitize_messages must ensure non-empty text
    assert cleaned[1]["content"] != ""
    assert "No information is found" in cleaned[1]["content"]


# ==========================================
# 6. Tests for Listing Available Models
# ==========================================

def test_fetch_available_models_mocked():
    from api_chat import fetch_available_models, get_provider_config
    config = get_provider_config("nvidia")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"id": "mistralai/mistral-large-4-1t-a52b"},
            {"id": "01-ai/yi-large"},
        ]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        models = fetch_available_models(config)
        assert len(models) == 2
        mock_get.assert_called_once()
        assert "models" in mock_get.call_args[0][0]


def test_list_available_models_all_and_filtered():
    from api_chat import list_available_models

    mock_data = [
        {"id": "nvidia/nemotron-3.5-super-vl:free"},
        {"id": "liquid/lfm-2.5-2.6b:free"},
        {"id": "meta-llama/llama-3.3-70b-instruct"},
    ]
    with patch("api_chat.fetch_available_models", return_value=mock_data):
        # All models
        all_models = list_available_models("openrouter", stream_output=False)
        assert len(all_models) == 3
        assert "liquid/lfm-2.5-2.6b:free" in all_models

        # Filtered by "free"
        free_models = list_available_models("openrouter", filter_query="free", stream_output=False)
        assert len(free_models) == 2
        assert "meta-llama/llama-3.3-70b-instruct" not in free_models

        # Filtered by "llama"
        llama_models = list_available_models("openrouter", filter_query="llama", stream_output=False)
        assert len(llama_models) == 1
        assert llama_models[0] == "meta-llama/llama-3.3-70b-instruct"


def test_no_side_effects_on_import():
    sys.modules.pop("api_chat", None)
    with patch("requests.post") as mock_post, \
         patch("chromadb.PersistentClient") as mock_chroma:
        import api_chat
        mock_post.assert_not_called()
        mock_chroma.assert_not_called()

