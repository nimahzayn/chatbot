import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np
genai.configure(api_key="AIzaSyCOlqTKvlOSHqV9r91ahNhfkXmmSiFZRhE")
model = genai.GenerativeModel("MECCo")
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
def chat_with_MECCo(prompt):
    response = model.generate_content(prompt)
    return response.text
print("You got anything to ask?(Y/N)")
while True:
    user_input = input("Message")
    if user_input.upper() == "N":
        print("Happy to help!")
        break
    response = chat_with_MECCo(user_input)
    print(response)
