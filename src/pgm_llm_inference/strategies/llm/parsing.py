import ast
import json
import re


def extract_last_json_object(raw: str) -> dict:
    """Extract the last balanced JSON-like object from an LLM response."""
    if not raw or not raw.strip():
        raise ValueError("Empty LLM output")

    stack = []
    start = None
    results = []

    for index, character in enumerate(raw):
        if character == "{":
            if not stack:
                start = index
            stack.append(character)
        elif character == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                results.append(raw[start : index + 1])

    if not results:
        raise ValueError(f"No JSON object found. Raw: {raw}")

    candidate = results[-1].strip()
    candidate = re.sub(r"(?<![:/])//.*", "", candidate)

    def replace_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n").replace("\r", "\\r")

    candidate = re.sub(r'"(.*?)"', replace_newlines, candidate, flags=re.DOTALL)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            clean_candidate = re.sub(r"[\x00-\x1F]+", " ", candidate)
            return json.loads(clean_candidate)
        except Exception:
            raise ValueError(
                f"Failed to parse JSON. Error: {error}. Candidate: {candidate}"
            ) from error
