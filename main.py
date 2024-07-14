# LLM App to summarize any content from the text file.

import streamlit as st
from enum import Enum
from io import StringIO
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

CREATIVITY=0


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e


class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def validate_api_key(self, key:str):
        if not key:
            st.warning("Please enter your API Key")
            # st.stop()
        else:    
            if (key.startswith("sk-") or key.startswith("gsk_")):
                st.success("Received valid API Key!")
            else:
                st.error("Invalid API Key!")

    def get_api_key(self):
        
        # Get the API Key to query the model
        input_text = st.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )

        # Validate the API key
        self.validate_api_key(input_text)
        return input_text
    
    def create(self):
        try:
            # Set the page title for blog post
            st.set_page_config(page_title="AI Long Text Summarizer")
            st.markdown("<h1 style='text-align: center;'>AI Long Text Summarizer</h1>", unsafe_allow_html=True)
            
            # Intro: instructions
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("Summarize your long content from text file.")

            with col2:
                st.write("Contact Hiren Kelaiya to build your AI Projects")

            # Select the model provider
            st.markdown("## Which model provider you want to choose?")
            option_model_provider = st.selectbox(
                    'Select the model provider',
                    ('GroqCloud', 'OpenAI')
                )

            # Input API Key for model to query
            st.markdown(f"## Enter Your {option_model_provider} API Key")
            # Input API Key for model to query
            api_key = self.get_api_key()

            # Input
            st.markdown("## Upload the text file you want to summarize")
            uploaded_file = st.file_uploader("Choose a file", type="txt")

            # Output
            st.markdown("### Here is your Summary:")
            if uploaded_file is not None:
                # To read file as bytes:
                # bytes_data = uploaded_file.getvalue()
                #st.write(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                # st.write(stringio)

                # To read file as string:
                file_input = stringio.read()
                # st.write(string_data)

                # Can be used wherever a "file-like" object is accepted:
                #dataframe = pd.read_csv(uploaded_file)
                #st.write(dataframe)
                if len(file_input.split(" ")) > 20000:
                    st.write("Please enter a shorter file. The maximum length is 20000 words.")
                    st.stop()

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n"], 
                    chunk_size=5000, 
                    chunk_overlap=350
                )
                splitted_documents = text_splitter.create_documents([file_input])
                
                # Load the LLM model
                llm_model = LLMModel(model_provider=option_model_provider)
                llm = llm_model.load(api_key=api_key)

                summarize_chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce"
                )

                summary_output = summarize_chain.invoke(splitted_documents)
                st.write(summary_output["output_text"])

        except Exception as e:
            st.error(str(e), icon=":material/error:")



def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()