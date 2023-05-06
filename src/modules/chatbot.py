import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
import numpy
import pandas 
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

class Chatbot:

    def __init__(self, model_name, temperature, vectors, chain_type):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.chain_type = chain_type

    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow-up entry: {question}
        Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are a friendly conversational assistant named Người yêu của Bảnh, designed to answer questions and chat with the user from a contextual file.
        You receive data from a user's file and a question, you must help the user find the information they need. 
        Your answers must be user-friendly and respond to the user in the language they speak to you.
        question: {question}
        =========
        context: {context}
        ======="""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        question_generator = LLMChain(llm=llm, prompt=self.CONDENSE_QUESTION_PROMPT,verbose=True)

        doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.QA_PROMPT, verbose=True)

        chain = ConversationalRetrievalChain(
            retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True)


        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

class Chatbot_no_file:

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    # _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    #     Chat History:
    #     {chat_history}
    #     Follow-up entry: {question}
    #     Standalone question:"""
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # qa_template = """You are a friendly conversational assistant named Người yêu của Bảnh, designed to answer questions and chat with the user from a contextual file.
    #     You receive data from a user's file and a question, you must help the user find the information they need. 
    #     Your answers must be user-friendly and respond to the user in the language they speak to you.
    #     question: {question}
    #     =========
    #     context: {context}
    #     ======="""
    # QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])


    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], 
            template=template
        )


        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0), 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationBufferWindowMemory(k=2),
        )

        output = chatgpt_chain.predict(human_input=query)

        # retriever = self.vectors.as_retriever()

        # question_generator = LLMChain(llm=llm, prompt=self.CONDENSE_QUESTION_PROMPT,verbose=True)

        # doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=self.QA_PROMPT, verbose=True)

        # chain = ConversationalRetrievalChain(
        #     retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, verbose=True)


        # chain_input = {"question": query, "chat_history": st.session_state["history"]}
        # result = chain(chain_input)

        # st.session_state["history"].append((query, result["answer"]))
        # #count_tokens_chain(chain, chain_input)
        # return result["answer"]
        return output

    
def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
