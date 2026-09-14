from transformers import MarianTokenizer, MarianMTModel


MODEL_NAME = "Helsinki-NLP/opus-mt-en-ar"

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate_abstract(text):

    # Validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Text must be a non-empty string."
        )

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate translation
    outputs = model.generate(
        **inputs,
        max_length=512
    )

    # Decode translated text
    translation = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translation