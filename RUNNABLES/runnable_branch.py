from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 150, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

print(final_chain.invoke({'topic':'Adidas vs Nike'}))
