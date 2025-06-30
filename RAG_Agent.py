from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ========== Load Environment ==========
load_dotenv()

# ========== Set Up Models ==========
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ========== Load PDF ==========
pdf_path = "Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
print(f"PDF has been loaded and has {len(pages)} pages")

# ========== Split Text ==========
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)

# ========== Build FAISS Vector Store ==========
vectorstore = FAISS.from_documents(pages_split, embeddings)
print("FAISS vector store created")

# ========== Set Up Retriever ==========
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ========== Define Tool ==========
@tool
def retriever_tool(query: str) -> str:
    """Searches and returns info from the Stock Market PDF."""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "unknown")
        preview = doc.page_content[:100].strip().replace("\n", " ")  # Optional: show a snippet
        #print(preview)
        results.append(f"Page {page + 1} (Chunk {i+1}):\n{doc.page_content}")
    return "\n\n".join(results)


tools = [retriever_tool]
llm = llm.bind_tools(tools)

# ========== LangGraph Setup ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {t.name: t for t in tools}

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    return {'messages': [response]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} → Query: {t['args'].get('query', 'No query')}")
        if t['name'] not in tools_dict:
            result = "Invalid tool name."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    return {'messages': results}

# ========== Build LangGraph ==========
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# ========== Run CLI Agent ==========
def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()