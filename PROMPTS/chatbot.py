from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

memory = ConversationBufferMemory(
    memory_key="chat_history",  
    return_messages=True         
)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Answer only in max 10 words."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory
)

print(chain.invoke({"input": "Hi, can you tell me about LangChain?"}))
print(chain.invoke({"input": "And who created it?"}))
print(chain.invoke({"input": "Remind me what you said about it earlier."}))
