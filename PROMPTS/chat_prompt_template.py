from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

chat_template = ChatPromptTemplate(
    messages = [
        ('system', 'you are a helpful {domain} expert'),
        ('human', 'Explain in simple terms about {topic}')

    ]
)

chain = chat_template | model

response = chain.invoke({
    'domain': 'science',
    'topic': 'black holes'
})

print(response.content)
