import streamlit as st
from utils import load_yaml
from pipline import Pipline
import matplotlib.pylab as plt

def plot_data(data, figsize=(12, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
        
    return fig
        
if __name__ == "__main__":
    cfg_path = 'config.yml'
    config = load_yaml(cfg_path)
    
    pipline = Pipline(
        config=config
    )
    st.title("Expressive Text To Speech (Demo)")

    text = st.text_input("Enter text")
    neutral = st.slider('Neutral', 0.0, 1.0, 0.0)
    happy = st.slider('Happy', 0.0, 1.0, 0.0)
    sad = st.slider('Sad', 0.0, 1.0, 0.0)
    angry = st.slider('Angry', 0.0, 1.0, 0.0)

    display_log = st.checkbox("Print Log")

    if st.button("run"):
        output_path, text, normed_text, phoneme, mel_spec, mel_outputs_postnet, alignments =  \
            pipline.infer(text, neutral=neutral, happy=happy, angry=angry, sad=sad)
            
        audio_file = open(f'{output_path}', "rb")
        audio_bytes = audio_file.read()
        st.markdown(f"## Audio : ")
        st.audio(audio_bytes, format="audio/wav", start_time=0)
        
        print(mel_spec)
        fig = plot_data((mel_spec, mel_outputs_postnet, alignments))
        
        st.pyplot(fig)
        if display_log:
            st.markdown(f'## Log : ')
            st.write(f'- text : " {text} "')
            st.write(f'- normed_text : " {normed_text} "')
            st.write(f"- phoneme_text: {phoneme}")