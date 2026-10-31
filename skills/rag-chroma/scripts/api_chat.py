"""
API Chat Agent (Remote APIs: OpenRouter, NVIDIA NIM, etc. + Chroma)
Provides interactive RAG chat capabilities using remote LLM endpoints.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Union, Any, Optional, Dict, List
from dotenv import load_dotenv
from halo import Halo

load_dotenv()

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

import requests

from chroma_utils import (
    get_chroma_client,
    get_collection,
    search_collection,
)

# Global variables for lazy initialization
_collection = None


def get_shared_collection():
    """
    Lazily initializes and retrieves the Chroma DB collection instance.
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = get_collection(client, name="cms_collection")
    return _collection


def unwrap_to_string(val: Any) -> str:
    """
    Recursively unwraps dictionaries and lists to extract a plain string.
    """
    if isinstance(val, dict):
        for key in ["query", "type", "description", "content"]:
            if key in val:
                unwrapped = unwrap_to_string(val[key])
                if unwrapped:
                    return unwrapped
        for v in val.values():
            unwrapped = unwrap_to_string(v)
            if unwrapped:
                return unwrapped
        return ""
    elif isinstance(val, list):
        for item in val:
            unwrapped = unwrap_to_string(item)
            if unwrapped:
                return unwrapped
        return ""
    return str(val)


def cms_collection_search(query: Union[str, dict], limit: int = 3) -> str:
    """
    Search the Chroma database for relevant documents matching query.
    """
    query_str = unwrap_to_string(query)
    if not query_str.strip():
        return "No search query provided."

    collection = get_shared_collection()
    results = search_collection(collection, query_text=query_str, limit=limit)

    ids = results.get("ids", [])
    score_name = results.get("score_name", "Score")
    output = []

    for i in range(len(ids)):
        raw_document = results["documents"][i]
        metadata = results["metadatas"][i]
        score = results["scores"][i]

        url = metadata.get("url", "No URL provided")
        tags = metadata.get("tags", "No tags")
        doc_split = raw_document.split('|')
        section = doc_split[0].replace("Section:", "").strip() if len(doc_split) > 0 else "N/A"
        resource = doc_split[1].replace("Resource:", "").strip() if len(doc_split) > 1 else "N/A"
        description = doc_split[2].replace("Description:", "").strip() if len(doc_split) > 2 else ""

        output.append(
            f"**{resource}**\nSection: {section}\nDescription: {description}\nURL: {url}\nTags: {tags}\n{score_name}: {score}"
        )
    return "\n\n---\n\n".join(output) if output else "No results found."


@dataclass
class ProviderConfig:
    """Configuration for an LLM API provider."""
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    client_type: str = "openai"  # "openai" or "requests"
    endpoint_path: str = "/chat/completions"
    models_path: str = "/models"
    headers: Dict[str, str] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    description: str = ""


# Default supported providers (easily expandable)
DEFAULT_PROVIDERS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="nvidia/nemotron-3.5-super-vl:free",
        client_type="openai",
        headers={
            "HTTP-Referer": "https://github.com/cms-collection",
            "X-Title": "CMS Collection RAG",
        },
        aliases=["open-router", "or"],
        description="OpenRouter API",
    ),
    "nvidia": ProviderConfig(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="mistralai/mistral-large-4-1t-a52b",
        client_type="openai",
        aliases=["nim", "nvidia-nim"],
        description="NVIDIA NIM API",
    ),
}

_PROVIDER_REGISTRY: Dict[str, ProviderConfig] = dict(DEFAULT_PROVIDERS)


def register_provider(config: ProviderConfig) -> None:
    """Registers a new or custom provider configuration."""
    _PROVIDER_REGISTRY[config.name.lower()] = config


def list_providers() -> List[str]:
    """Returns a list of all registered provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def get_provider_config(name_or_alias: str) -> ProviderConfig:
    """
    Resolves a provider configuration by name or alias (case-insensitive).
    """
    lookup = name_or_alias.strip().lower()
    if lookup in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[lookup]

    for config in _PROVIDER_REGISTRY.values():
        if lookup in [a.lower() for a in config.aliases]:
            return config

    supported = ", ".join(list_providers())
    raise ValueError(f"Unknown provider '{name_or_alias}'. Supported providers: {supported}")


def get_api_key(config: ProviderConfig) -> str:
    """
    Retrieves the API key for a provider from the environment.
    """
    key = os.getenv(config.api_key_env, "").strip()
    if not key:
        raise ValueError(
            f"Missing API key for provider '{config.name}'. "
            f"Please set the '{config.api_key_env}' environment variable in your .env file or environment."
        )
    return key


def resolve_provider_and_model(
    provider_arg: Optional[str] = None,
    model_arg: Optional[str] = None,
) -> tuple[ProviderConfig, str]:
    """
    Resolves the provider config and model to use, checking arguments,
    environment variables (API_PROVIDER, API_MODEL), or sensible defaults.
    """
    provider_name = (
        provider_arg
        or os.getenv("API_PROVIDER")
        or "openrouter"
    )
    config = get_provider_config(provider_name)

    model = (
        model_arg
        or os.getenv("API_MODEL")
        or config.default_model
    )
    return config, model


SYSTEM_PROMPT = """You are a Research Assistant.

- For greetings: Reply with one sentence and wait for a question.
- For technical questions: You MUST use the 'cms_collection_search' tool to search the database. Pass plain text search keywords.
- When you receive results from 'cms_collection_search': Synthesize your final response immediately using the retrieved results. Format the results as a list under the header '## Search Results' including resource name, section, description, and URL.
- If no information is found in the search results, say "No information is found in the search results." Do not fabricate information.
"""

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "cms_collection_search",
        "description": "Search the CMS collection database for technical information, tools, libraries, or datasets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain text search query keywords"
                }
            },
            "required": ["query"]
        }
    }
}


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures message history adheres strictly to OpenAI / NVIDIA NIM requirements:
    1. Every message has role and valid content (string or None only with tool_calls).
    2. Any assistant message with tool_calls is followed by tool messages for each tool_call_id.
    3. Trailing or dangling tool_calls without tool messages are repaired.
    """
    sanitized: List[Dict[str, Any]] = []
    pending_tool_calls: set[str] = set()

    for msg in messages:
        m = dict(msg)
        role = m.get("role", "user")

        if role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                pending_tool_calls = {tc.get("id") for tc in tcs if tc.get("id")}
                if m.get("content") is None:
                    m["content"] = ""
            else:
                pending_tool_calls.clear()
                # Assistant turns without tool calls must contain non-empty content
                # to satisfy strict API validators (e.g. NVIDIA NIM / Mistral 400 Bad Request)
                if not m.get("content") or not str(m["content"]).strip():
                    m["content"] = "No information is found in the search results."
        elif role == "tool":
            t_id = m.get("tool_call_id")
            if t_id and t_id in pending_tool_calls:
                pending_tool_calls.remove(t_id)
        elif role in ("user", "system"):
            if pending_tool_calls:
                for missing_id in pending_tool_calls:
                    sanitized.append({
                        "role": "tool",
                        "tool_call_id": missing_id,
                        "content": "No tool response available.",
                    })
                pending_tool_calls.clear()
            if m.get("content") is None:
                m["content"] = ""

        sanitized.append(m)

    if pending_tool_calls:
        for missing_id in pending_tool_calls:
            sanitized.append({
                "role": "tool",
                "tool_call_id": missing_id,
                "content": "No tool response available.",
            })

    return sanitized


def call_chat_completion(
    provider: ProviderConfig,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 60,
    on_retry: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes a chat completion call against the given provider endpoint.
    Uses OpenAI client if client_type == 'openai' and available,
    otherwise uses requests.post.
    """
    api_key = get_api_key(provider)
    clean_messages = sanitize_messages(messages)

    if provider.client_type == "openai" and HAS_OPENAI and OpenAI is not None:
        try:
            client_kwargs: Dict[str, Any] = {
                "base_url": provider.base_url,
                "api_key": api_key,
                "timeout": float(timeout),
                "max_retries": 1,
            }
            if provider.headers:
                client_kwargs["default_headers"] = provider.headers

            client = OpenAI(**client_kwargs)
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": clean_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            result_message: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content or "",
            }
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            return {"choices": [{"message": result_message}]}
        except Exception as e:
            # If it's an explicit 401 or authentication error, do not mask it with fallback
            err_str = str(e)
            if "401" in err_str or "unauthorized" in err_str.lower() or "authentication" in err_str.lower():
                raise
            # Otherwise, fall back to requests (e.g. client library incompatibilities)
            pass

    # Fallback / direct requests HTTP client
    url = f"{provider.base_url.rstrip('/')}{provider.endpoint_path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **provider.headers,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": clean_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    # Attempt request with progressive automatic retry for rate limits (HTTP 429)
    resp = None
    max_retries = 3
    base_delays = [3, 5, 8]
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as to_err:
            raise RuntimeError(
                f"Request timed out after {timeout}s waiting for {provider.name.upper()} endpoint. "
                f"The model '{model}' appears to be unresponsive or undergoing a cold start on {provider.name.upper()}."
            ) from to_err

        if resp.status_code == 429 and attempt < max_retries - 1:
            hdr_retry = 0
            try:
                hdr_retry = int(resp.headers.get("Retry-After", 0))
            except Exception:
                pass
            retry_after = max(hdr_retry, base_delays[min(attempt, len(base_delays) - 1)])
            if on_retry:
                on_retry(f"Rate limited by provider (429). Retrying in {retry_after}s (attempt {attempt + 1}/{max_retries - 1})...")
            time.sleep(retry_after)
            if on_retry:
                on_retry("Thinking...")
            continue
        break

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        err_msg = ""
        try:
            err_data = resp.json()
            if isinstance(err_data, list):
                err_msg = "; ".join(
                    str(item.get("msg") or item.get("message") or item) if isinstance(item, dict) else str(item)
                    for item in err_data
                )
            elif isinstance(err_data, dict):
                detail = err_data.get("detail")
                if isinstance(detail, list):
                    err_msg = "; ".join(
                        str(d.get("msg") or d.get("message") or d) if isinstance(d, dict) else str(d)
                        for d in detail
                    )
                elif detail:
                    err_msg = str(detail)
                elif "error" in err_data and isinstance(err_data["error"], dict):
                    err_obj = err_data["error"]
                    base_msg = err_obj.get("message", str(err_obj))
                    meta = err_obj.get("metadata")
                    if isinstance(meta, dict) and meta.get("raw"):
                        err_msg = f"{base_msg}: {meta['raw']}"
                    elif isinstance(meta, dict) and meta.get("remedy_hint"):
                        err_msg = f"{base_msg} ({meta['remedy_hint']})"
                    else:
                        err_msg = base_msg
                else:
                    err_msg = err_data.get("message") or str(err_data)
        except Exception:
            err_msg = resp.text[:300] if resp.text else str(http_err)

        raise RuntimeError(f"API Error ({resp.status_code}): {err_msg}") from http_err

    return resp.json()


def start_chat(provider_name: Optional[str] = None, model: Optional[str] = None):
    """
    Interactive chat session connected to the selected remote API provider.
    """
    try:
        config, model_name = resolve_provider_and_model(provider_name, model)
    except Exception as e:
        print(f"Configuration error: {e}")
        return

    print("\n" + "=" * 52)
    print(f"  RAG Chat Agent (API: {config.name.upper()} + Chroma)")
    print(f"  Model: {model_name}")
    print("  Type 'exit' to quit | 'clear' to reset")
    print("=" * 52 + "\n")

    # Check API key before entering loop
    try:
        get_api_key(config)
    except ValueError as e:
        print(f"Error: {e}\n")
        return

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("System: Memory cleared.\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        spinner = Halo(text='Thinking...', spinner='dots')
        spinner.start()

        def set_status(msg: str):
            spinner.text = msg

        try:
            # 1. First step: ask model (it can reply directly or call tools)
            try:
                response = call_chat_completion(
                    config, model_name, messages, tools=[TOOL_DEFINITION], tool_choice="auto", on_retry=set_status
                )
            except Exception as call_err:
                err_s = str(call_err).lower()
                if "tools" in err_s and ("not supported" in err_s or "unexpected" in err_s):
                    response = call_chat_completion(config, model_name, messages, tools=None, on_retry=set_status)
                else:
                    raise

            choice_msg = response["choices"][0]["message"]
            tool_calls = choice_msg.get("tool_calls")

            if not tool_calls:
                content = (choice_msg.get("content") or "").strip()
                if not content:
                    content = "No information is found in the search results."
                choice_msg["content"] = content
                messages.append(choice_msg)
                spinner.stop()
                print(f"\nAssistant: {content}\n")
            else:
                # Model returned tool calls
                messages.append(choice_msg)
                collected_results: List[str] = []

                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(fn_args_raw, str):
                        try:
                            fn_args = json.loads(fn_args_raw)
                        except Exception:
                            fn_args = {"query": fn_args_raw}
                    else:
                        fn_args = fn_args_raw

                    if fn_name == "cms_collection_search":
                        search_term = unwrap_to_string(fn_args)
                        spinner.text = f"Searching collection for: '{search_term}'..."
                        tool_result = cms_collection_search(fn_args)
                    else:
                        tool_result = f"Unknown tool: {fn_name}"

                    collected_results.append(tool_result)
                    tool_msg = {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": tc.get("id", "call_1"),
                        "content": tool_result,
                    }
                    messages.append(tool_msg)

                # 2. Synthesize response: generate text answer from tool results.
                # Pass tools=None so the model generates plain text rather than another tool call.
                spinner.text = 'Synthesizing response...'
                try:
                    second_resp = call_chat_completion(
                        config, model_name, messages, tools=None, on_retry=set_status
                    )
                except Exception:
                    # Retry with tools schema in case endpoint requires it
                    second_resp = call_chat_completion(
                        config, model_name, messages, tools=[TOOL_DEFINITION], tool_choice="auto", on_retry=set_status
                    )

                final_msg = second_resp["choices"][0]["message"]
                content = (final_msg.get("content") or "").strip()
                if not content:
                    # Model returned empty text in synthesis; construct fallback from retrieved results
                    valid_results = [r for r in collected_results if r and r != "No results found."]
                    if valid_results:
                        content = f"## Search Results\n\n" + "\n\n---\n\n".join(valid_results)
                    else:
                        content = "No information is found in the search results."
                final_msg["content"] = content
                messages.append(final_msg)
                spinner.stop()
                print(f"\nAssistant: {content}\n")

        except Exception as e:
            spinner.fail("Something went wrong.")
            print(f"Error: {e}\n")


def fetch_available_models(provider: ProviderConfig) -> List[Dict[str, Any]]:
    """
    Fetches raw model records from the provider's models endpoint.
    """
    models_url = f"{provider.base_url.rstrip('/')}{provider.models_path}"
    headers = {
        "Content-Type": "application/json",
        **provider.headers,
    }
    try:
        api_key = get_api_key(provider)
        headers["Authorization"] = f"Bearer {api_key}"
    except Exception:
        pass  # Querying /models does not require auth on OpenRouter & NVIDIA NIM

    resp = requests.get(models_url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data


def list_available_models(
    provider_name: Optional[str] = None,
    filter_query: Optional[str] = None,
    stream_output: bool = True,
) -> List[str]:
    """
    Lists available model IDs for the chosen provider, optionally filtered.
    Returns the list of matching model IDs.
    """
    config, default_model = resolve_provider_and_model(provider_name, None)

    if stream_output:
        print("\n" + "=" * 56)
        print(f"  Available Models (Provider: {config.name.upper()})")
        print("=" * 56)
        print(f"  Endpoint: {config.base_url}{config.models_path}")
        print(f"  Default : {default_model}")
        if filter_query:
            print(f"  Filter  : '{filter_query}'")
        print("=" * 56 + "\n")

    try:
        raw_models = fetch_available_models(config)
    except Exception as e:
        if stream_output:
            print(f"Error fetching models from {config.name}: {e}\n")
        return []

    model_ids = [m.get("id", "") for m in raw_models if m.get("id")]
    model_ids.sort()

    if filter_query:
        q = filter_query.lower()
        model_ids = [mid for mid in model_ids if q in mid.lower()]

    if not model_ids:
        if stream_output:
            print(f"No models found matching '{filter_query}'.\n")
        return []

    free_models = [mid for mid in model_ids if ":free" in mid]
    other_models = [mid for mid in model_ids if ":free" not in mid]

    if stream_output:
        if free_models:
            print(f"--- Free Models ({len(free_models)}) ---")
            for mid in free_models:
                marker = " (default)" if mid == default_model else ""
                print(f"  • {mid}{marker}")
            print()

        if other_models:
            section_title = (
                f"--- Standard Models ({len(other_models)}) ---"
                if free_models
                else f"--- Models ({len(other_models)}) ---"
            )
            print(section_title)
            display_limit = 50 if (not filter_query and len(other_models) > 70) else len(other_models)
            for mid in other_models[:display_limit]:
                marker = " (default)" if mid == default_model else ""
                print(f"  • {mid}{marker}")
            if display_limit < len(other_models):
                print(f"  ... and {len(other_models) - display_limit} more models.")
                print(f"\nTip: Filter with `pixi run chat-api {config.name} list <keyword>`")
            print()

        print(f"Total: {len(model_ids)} model(s)")
        print(f"Run chat: pixi run chat-api {config.name} <model_id>\n")

    return model_ids


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    # Check for 'list' subcommand
    if any(a.lower() == "list" for a in args):
        non_list_args = [a for a in args if a.lower() != "list"]
        target_provider = non_list_args[0] if len(non_list_args) > 0 else None
        filter_term = non_list_args[1] if len(non_list_args) > 1 else None

        # If user did `chat-api list glm`, target_provider might be a filter query if not a known provider
        if target_provider:
            known_providers = set(_PROVIDER_REGISTRY.keys())
            for cfg in _PROVIDER_REGISTRY.values():
                known_providers.update(a.lower() for a in cfg.aliases)
            if target_provider.lower() not in known_providers:
                filter_term = target_provider
                target_provider = None

        list_available_models(target_provider, filter_term)
    else:
        cli_provider = args[0] if len(args) > 0 else None
        cli_model = args[1] if len(args) > 1 else None
        start_chat(cli_provider, cli_model)

