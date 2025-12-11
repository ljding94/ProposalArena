import os
import requests
import json


def get_llm_response(user_prompt, system_prompt=None, model="google/gemini-2.5-flash-preview-09-2025"):
    """
    Get response from LLM via OpenRouter using native requests.

    Args:
        user_prompt (str): The user prompt to send to the LLM
        system_prompt (str): The system prompt
        model (str): Model to use
    Returns:
        tuple: (content, usage_dict) where usage_dict has input_tokens, output_tokens, total_tokens
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "provider": {
                    "order": ["google-vertex", "groq"],  # preferred provider(s)
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
            }
        ),
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        usage_dict = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        return content, usage_dict
    else:
        raise Exception(f"API error: {response.status_code} {response.text}")


def get_embedding_vector(text, model="qwen/qwen3-embedding-8b", provider="deepinfra"):
    """
    Get embedding vector from LLM via OpenRouter using native requests.

    Args:
        text (str): The text to get embedding for
        model (str): Model to use for embedding
        provider (str): Preferred provider for the model
    Returns:
        list: The embedding vector as a list of floats
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        url="https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "provider": {
                    "order": [provider],
                },
                "input": [text],
            }
        ),
    )

    if response.status_code == 200:
        result = response.json()
        embedding = result["data"][0]["embedding"]
        return embedding
    else:
        raise Exception(f"API error: {response.status_code} {response.text}")
