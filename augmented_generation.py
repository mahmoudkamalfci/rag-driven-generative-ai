def augment_prompt(user_prompt: str, context: str) -> str:
    return f"{user_prompt}\n\nContext:\n{context}"
