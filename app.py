import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


id2label = {0: "Negative", 1: "Positive"}

def classify(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    logits = model(inputs).logits
    predictions = torch.argmax(logits)

    return id2label[predictions.tolist()]


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("./model/v3")
    tokenizer = AutoTokenizer.from_pretrained("./model/v3")

    classification_model = AutoModelForCausalLM.from_pretrained("./model/classification_v2")
    classification_tokenizer = AutoTokenizer.from_pretrained("./model/classification_v2")

    st.title("Steam Review Generator for Baldur's Gate 3")

    # User input prompt
    prompt = st.text_area("Enter a prompt:", "I think that")

    if st.button("Generate"):
        # Tokenize and generate text
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated_text = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200)

        # Decode and display the generated text
        st.subheader("Generated Text:")
        st.write(tokenizer.decode(generated_text[0], skip_special_tokens=True))

        # Display the review's polarity
        st.subheader("Review Type:")
        st.write(classify(generated_text))
