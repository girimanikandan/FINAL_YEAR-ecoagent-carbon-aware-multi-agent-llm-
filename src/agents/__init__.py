"""Fixed agent roles. Prompts live in configs/prompts/<version>/*.txt (never embedded in code)."""
import re

from src.config import prompt_template

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """First fenced code block; falls back to the raw text if it looks like code; else empty."""
    m = _FENCE.findall(text or "")
    if m:
        return m[0].strip("\n")
    t = (text or "").strip()
    return t if ("def " in t or "import " in t) else ""


class Agent:
    ROLE = ""
    STAGE = ""      # graph stage this role belongs to (repair counts as part of the coder stage)

    def __init__(self, cfg: dict):
        self.template = prompt_template(cfg, self.ROLE)

    def build_prompt(self, **ctx) -> str:
        return self.template.format(**ctx)
