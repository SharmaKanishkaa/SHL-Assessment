import os
import streamlit as st


# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  
        temperature=0.5,
        max_length=512,  
        # model_kwargs={
        #     "max_new_tokens": 512  
        # }
    )
    return llm

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


HF_TOKEN ="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"

def main():
    st.title("SHL Assessment Recommendation System")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                You are an intelligent recommendation assistant that helps hiring managers choose the most relevant SHL individual assessments.

                Use only the information provided in the context to recommend SHL assessments based on the user's input. 
                Do NOT hallucinate or assume any information that is not in the context. 
                Give main focus on job role, test type and duration.
                Each recommendation must include:
                - Assessment Name
                - URL (must be from SHL product catalog)
                - Remote Testing Support (Yes/No)
                - Adaptive/IRT Support (Yes/No)
                - Duration
                - Test Type
                Context: {context}
                Question: {question}
                Recommend only 10 lising.  

                Start your answer directly. No introductions or small talk.
                """
        # Return results in sorted order of matching.
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            # result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")
if __name__ == "__main__":
    main()