"""Gemini REST adapter; API keys are sent in headers, never URLs."""
import re
import httpx
from .providers import ProviderError


def generate_gemini(messages, config):
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', config.gemini_model):
        raise ProviderError('Set GEMINI_MODEL to a valid model ID from Google AI Studio.')
    contents = []
    for message in messages[1:]:
        contents.append({'role': 'model' if message['role'] == 'assistant' else 'user',
                         'parts': [{'text': message['content']}]})
    try:
        with httpx.Client(timeout=45) as client:
            response = client.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent',
                headers={'x-goog-api-key': config.gemini_api_key},
                json={'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
                      'contents': contents,
                      'generationConfig': {'temperature': 0, 'maxOutputTokens': 2048}},
            )
            if response.status_code == 429:
                raise ProviderError('The Gemini demo quota is temporarily reached. Showing source excerpts instead; try generated answers again later.')
            response.raise_for_status()
            data = response.json()
        parts = data['candidates'][0]['content']['parts']
        answer = '\n'.join(part['text'] for part in parts if part.get('text') and not part.get('thought'))
        if not answer.strip():
            raise ValueError('No answer')
        return answer
    except ProviderError:
        raise
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError('Gemini could not generate an answer. Check the key, selected model, and quota. Showing source excerpts instead.') from exc
