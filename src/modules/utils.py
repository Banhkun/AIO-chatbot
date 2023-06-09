import os
import pandas as pd
import streamlit as st
import pdfplumber
from PIL import Image

from modules.chatbot import Chatbot
from modules.chatbot import Chatbot_no_file

from modules.embedder import Embedder
from langchain.callbacks import get_openai_callback

class Utilities:

    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or 
        from the user's input and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded from .env", icon="🚀")
        else:
            # user_api_key = st.sidebar.text_input(
            #     label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            # )
            #No need to input apikey anymore
            user_api_key = ""
            # if user_api_key:
            #     st.sidebar.success("API key loaded", icon="🚀")

        return user_api_key
    
    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type=["csv", "pdf", "txt","png","jpg"], label_visibility="collapsed")
        if uploaded_file is not None:

            def show_csv_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                uploaded_file.seek(0)
                shows = pd.read_csv(uploaded_file)
                file_container.write(shows)
            def show_png_file(uploaded_file):
                file_container = st.expander("Your PNG file:")
                image = Image.open(uploaded_file)
                file_container.image(image, caption='Uploaded PNG', use_column_width=True)
            
            def show_pdf_file(uploaded_file):
                file_container = st.expander("Your PDF file :")
                with pdfplumber.open(uploaded_file) as pdf:
                    pdf_text = ""
                    for page in pdf.pages:
                        pdf_text += page.extract_text() + "\n\n"
                file_container.write(pdf_text)

            
            def get_file_extension(uploaded_file):
                return os.path.splitext(uploaded_file)[1].lower()
            
            file_extension = get_file_extension(uploaded_file.name)

            # Show the contents of the file based on its extension
            if file_extension == ".csv" :
                show_csv_file(uploaded_file)
            elif file_extension == ".png" :
                show_png_file(uploaded_file)
            elif file_extension == ".jpg" :
                show_png_file(uploaded_file)    
            elif file_extension== ".pdf" : 
                show_pdf_file(uploaded_file)

        else:
            st.sidebar.info(
                "👆 Upload your CSV file here"
            )
            st.session_state["reset_chat"] = False

        #print(uploaded_file)
        return uploaded_file

    
    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature, chain_type):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        if uploaded_file and os.path.splitext(uploaded_file.name)[1].lower() == ".csv":
            chatbot = Chatbot(model_name=model, temperature=temperature,uploaded_file=uploaded_file)
            return chatbot
        elif uploaded_file and (os.path.splitext(uploaded_file.name)[1].lower() == ".png" or os.path.splitext(uploaded_file.name)[1].lower() == ".jpg"):
            chatbot = Chatbot(model_name=model, temperature=temperature,image=uploaded_file)
            return chatbot
        else:
            embeds = Embedder()

            with st.spinner("Processing..."):
                uploaded_file.seek(0)
                file = uploaded_file.read()
                # Get the document embeddings for the uploaded file
                vectors = embeds.getDocEmbeds(file, uploaded_file.name)

                # Create a Chatbot instance with the specified model and temperature
                chatbot = Chatbot(model, temperature,vectors, chain_type)
            st.session_state["ready"] = True

            return chatbot
    @staticmethod
    def setup_chatbot_no_file(model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """

        with st.spinner("Processing..."):
            # Create a Chatbot instance with the specified model and temperature
            chatbot = Chatbot_no_file(model, temperature)
        st.session_state["ready"] = True

        return chatbot
    def count_tokens_agent(agent, query):
        """
        Count the tokens used by the CSV Agent
        """
        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens')

        return result
    
