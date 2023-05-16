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
import re
from io import BytesIO
import sys
from io import StringIO
from langchain.agents import create_csv_agent
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
from PIL import Image
from langchain.document_loaders import YoutubeLoader
import openai
import requests
import os
class Chatbot:

    def __init__(self, model_name, temperature, vectors=False, chain_type=False,uploaded_file=False,image=False):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.chain_type = chain_type
        self.uploaded_file= uploaded_file
        self.image = image
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow-up entry: {question}
        Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Assistant will give a full answer about the problem

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

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
        if ("generate" in query.lower() and ("image" in query.lower() or "picture" in query.lower())) or ("tạo cho tôi một" in query.lower() and ("bức ảnh" in query.lower() or "tấm ảnh" in query.lower()or "bức tranh" in query.lower())):

            template_for_Dalle= """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: Help me write a prompt for Dall-e to {human_input}
Assistant:"""
            openai.api_key = ""
            prompt = PromptTemplate(
                input_variables=["history", "human_input"], 
                template=template_for_Dalle
            )
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0.7,max_tokens=3500), 
                prompt=prompt, 
                verbose=True, 
                memory=ConversationBufferWindowMemory(k=2),
            )

            output = chatgpt_chain.predict(human_input=query)
            response = openai.Image.create(prompt=output,    n=1,   size="256x256",)
            
            print(response["data"][0]["url"])
            image_container = st.expander("Display Generated PNG")
            response = requests.get(response["data"][0]["url"])
            img = Image.open(BytesIO(response.content))

            image_container.image(img, caption='Generated PNG', use_column_width=True)
            #return [output,img]
            return [output.strip(),img]
            
        if self.image:
            try:
                loader = ImageCaptionLoader(path_images=self.image)
                # Image.open(requests.get(list_image_urls[0], stream=True).raw).convert('RGB')
                index = VectorstoreIndexCreator().from_loaders([loader])
                return  index.query(query)
            except:
                prompt = PromptTemplate(
                    input_variables=["history", "human_input"], 
                    template=self.template
                )
                chatgpt_chain = LLMChain(
                    llm=OpenAI(temperature=0.7,max_tokens=3500), 
                    prompt=prompt, 
                    verbose=True, 
                    memory=ConversationBufferWindowMemory(k=5),
                    
                )
                output = chatgpt_chain.predict(human_input=query)
                return output.strip()
        if self.uploaded_file:
            try:
                            # format the CSV file for the agent
                uploaded_file_content = BytesIO(self.uploaded_file.getvalue())

                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                # Create and run the CSV agent with the user's query
                agent = create_csv_agent(ChatOpenAI(temperature=0), uploaded_file_content, verbose=True, max_iterations=4)
                agent.run(query)
                # agent.run(query)
                sys.stdout = old_stdout

                # Clean up the agent's thoughts to remove unwanted characters
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                
                # Display the agent's thoughts
                with st.expander("Display the agent's thoughts"):
                    st.write(cleaned_thoughts)

                thoughts = []
                final_answer = None

                lines = cleaned_thoughts.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if line.startswith("Thought:"):
                        thought = line[len("Thought:"):]
                        thought_actions = []
                        j = i + 1
                        while j < len(lines) and lines[j].startswith("Action:"):
                            action_input = lines[j+1][len("Action Input:"):].strip()
                            thought_actions.append(action_input)
                            j += 2
                        thoughts.append({"thought": thought, "actions": thought_actions})
                        i = j - 1
                    elif line.startswith("Final Answer:"):
                        final_answer = line[len("Final Answer:"):]
                    i += 1

                result = ""
                for i in range(len(thoughts)):
                    thought = thoughts[i]["thought"]
                    result += f"{thought}\n"
                    for j in range(len(thoughts[i]["actions"])):
                        action = thoughts[i]["actions"][j]
                        result += f"  \n{action}\n"
                    result += "\n"

                result += f"{final_answer}"
                return result
            except:
                prompt = PromptTemplate(
                    input_variables=["history", "human_input"], 
                    template=self.template
                )
                chatgpt_chain = LLMChain(
                    llm=OpenAI(temperature=0.7,max_tokens=3500), 
                    prompt=prompt, 
                    verbose=True, 
                    memory=ConversationBufferWindowMemory(k=5),
                    
                )
                output = chatgpt_chain.predict(human_input=query)
                return output.strip()
        
        if self.vectors:
            try:        
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
            except:
                prompt = PromptTemplate(
                    input_variables=["history", "human_input"], 
                    template=self.template
                )
                chatgpt_chain = LLMChain(
                    llm=OpenAI(temperature=0.7,max_tokens=3500), 
                    prompt=prompt, 
                    verbose=True, 
                    memory=ConversationBufferWindowMemory(k=5),
                    
                )
                output = chatgpt_chain.predict(human_input=query)
                return output.strip()

class Chatbot_no_file:

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        template_for_Dalle= """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: Help me write a prompt for Dall-e to {human_input}
Assistant:"""
        template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Assistant will give a full answer about the problem

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""
        if ("generate" in query.lower() and ("image" in query.lower() or "picture" in query.lower())) or ("tạo cho tôi một" in query.lower() and ("bức ảnh" in query.lower() or "tấm ảnh" in query.lower()or "bức tranh" in query.lower())):
            

            openai.api_key = ""
            
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

            prompt = PromptTemplate(
                input_variables=["history", "human_input"], 
                template=template_for_Dalle
            )
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0.7, max_tokens=3500), 
                prompt=prompt, 
                verbose=True, 
                memory=ConversationBufferWindowMemory(k=2),
            )

            output = chatgpt_chain.predict(human_input=query)
            response = openai.Image.create(prompt=output,    n=1,   size="256x256",)
            image_container = st.expander("Display Generated PNG")
            response = requests.get(response["data"][0]["url"])
            img = Image.open(BytesIO(response.content))

            image_container.image(img, caption='Generated PNG', use_column_width=True)
            #return [output,img]
            return [output.strip(),img]


        else:
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

            prompt = PromptTemplate(
                input_variables=["history", "human_input"], 
                template=template
            )
            chatgpt_chain = LLMChain(
                llm=OpenAI(temperature=0.7,max_tokens=3500), 
                prompt=prompt, 
                verbose=True, 
                memory=ConversationBufferWindowMemory(k=5),
                
            )
            output = chatgpt_chain.predict(human_input=query)
            return output.strip()

    
def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
