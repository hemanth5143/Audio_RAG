from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from stt import record_audio, transcribe_audio  # Import necessary functions from stt.py

model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3"
)

st.title("Data analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": model})

    # Add option for input method after CSV upload
    input_method = st.radio("Choose input method", ('Type prompt', 'Speak prompt'))

    prompt = ""
    if input_method == 'Type prompt':
        prompt = st.text_area("Enter your prompt:")
    elif input_method == 'Speak prompt':
        if st.button("Record Audio"):
            with st.spinner("Listening..."):
                filename = "recorded_audio.wav"
                if record_audio(filename):  # Record audio with silence detection
                    prompt = transcribe_audio(filename)  # Get the transcription after recording
            if prompt:
                st.write(f"Transcription: {prompt}")
                # Automatically generate the response after recording
                with st.spinner("Generating response..."):
                    st.write(df.chat(prompt))
            else:
                st.error("Failed to transcribe audio.")
