import re

def clean_text(content: str) -> str:
    """Remove references that usually appear as [1], [2], etc."""
    return re.sub(r'\[\d+\]', '', content)
