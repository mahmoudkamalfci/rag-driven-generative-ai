from augmented_generation import augment_prompt

def test_augment_prompt():
    user_prompt = "What is Mars?"
    context = "Mars is a red planet."
    expected = "What is Mars?\n\nContext:\nMars is a red planet."
    assert augment_prompt(user_prompt, context) == expected
