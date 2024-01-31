import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your pre-trained language model
model = AutoModelForCausalLM.from_pretrained("./model/v3")
tokenizer = AutoTokenizer.from_pretrained("./model/v3")

# Streamlit app
st.title("Steam Review Generator for Baldur's Gate 3")

# User input prompt
prompt = st.text_area("Enter a prompt:", "I think that...")

if st.button("Generate"):
    # Tokenize and generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_text = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200)

    # Decode and display the generated text
    st.subheader("Generated Text:")
    st.write(tokenizer.decode(generated_text[0], skip_special_tokens=True))

