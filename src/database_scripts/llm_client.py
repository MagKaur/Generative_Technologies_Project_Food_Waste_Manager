import logging

from typing import Any, Dict, List, Optional
from openai import AzureOpenAI
from config.config import AZURE_OPENAI_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION


class LLMClient:
    """Client for communication with Azure OpenAI"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION
        )
        self._deployment = AZURE_OPENAI_MODEL
        self._logger = logger or logging.getLogger(__name__)

    def parse_ingredients(self, ingredients: List[str]) -> List[Dict]:
        """
        Calls model with list of ingredients and returns structured json data.
        """
        system_prompt = (
            "You are an assistant that extracts structured ingredient data from "
            "recipe ingredient lines. Return ONLY valid JSON."
        )

        user_prompt = (
            "Extract ingredients from the following list. "
            "For each item, return: name (string, normalized), amount (float or null), "
            "unit (string or null), comment (string or null). "
            "Respond in JSON with the shape:\n"
            "{ \"ingredients\": [ {\"name\": ..., \"amount\": ..., \"unit\": ..., \"comment\": ...}, ... ] }\n\n"
            f"Input list:\n{ingredients}"
        )

        response = self._client.chat.completions.create(
            model=self._deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content

        #self._logger.debug(content)

        import json
        data = json.loads(content)
        return data.get("ingredients", [])
