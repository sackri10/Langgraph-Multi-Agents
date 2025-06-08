import os
import streamlit as st
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
tvly_api_key = os.getenv("TAVILY_API_KEY", "")

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Load and process documents
@st.cache_resource
def get_retriever():
    # Load documents
    loader = DirectoryLoader("../data", glob="**/consoles.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    
    # Create vector store
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    generation: str
    validation_passed: bool

# Define nodes
def llm_node(state: AgentState):
    """A node that calls an LLM for a general purpose answer."""
    st.info("Calling LLM Node")
    model = get_llm()
    response = model.invoke(state["messages"])
    return {"messages": [response], "generation": response.content}

def rag_node(state: AgentState):
    """A node that performs RAG on the Sony PlayStation technical documents."""
    st.info("Calling RAG Node on the Sony PlayStation technical documents")
    question = state["messages"][-1].content
    retriever = get_retriever()
    docs = retriever.invoke(question)
    rag_content = "\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering based on provided context.
        Only use the following context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Context: {context}
        Question: {question}
        Answer:
        """
    )
    model = get_llm()
    chain = prompt | model
    response = chain.invoke({"context": rag_content, "question": question})
    return {"messages": [response], "generation": response.content}

def web_search_node(state: AgentState):
    """A node that performs a web search using Tavily for real time information."""
    st.info("Calling Web Search Node")
    question = state["messages"][-1].content
    web_search_tool = TavilySearchResults(max_results=2, api_key=tvly_api_key)
    st.write("Web Search Tool initialized")
    search_results = web_search_tool.invoke({"query": question})
    generation = "\n".join([res["content"] for res in search_results])
    response_message = AIMessage(content=generation)
    return {"messages": [response_message], "generation": generation}

def validation_node(state: AgentState):
    """A node that validates the generated output. 
    It uses an LLM to check if the generation is relevant to the question asked and fully answers it."""
    st.info("Calling Validation Node")
    question = state["messages"][0].content
    generation = state["generation"]
    validation_prompt = ChatPromptTemplate.from_template(
        """Given the original user question and generated answer, please validate the following
        1. Does the answer directly address the user's question?
        2. Is the answer accurate and not hallucinated?
        3. Is the answer complete and detailed enough?

        Original Question: {question}
        Generated Answer: {generation}

        Please answer with a single word 'yes' or 'no' only.
         If the answer is valid, respond with only the word "VALID".
    If the answer is invalid, respond with "INVALID" followed by a brief, constructive critique on how to improve it for the next attempt.
    For example: "INVALID: The answer is too generic. The user asked for a specific number."
            """
    )
    model = get_llm()
    validation_chain = validation_prompt | model
    validation_response = validation_chain.invoke({"question": question, "generation": generation})
    validation_text = validation_response.content

    if "VALID" in validation_text:
        st.success("VALIDATION PASSED")
        return {"validation_passed": True}
    else:
        st.error("VALIDATION FAILED")
        # Add the critique to the message history so the supervisor can use it
        critique_message = HumanMessage(content=f"Critique from validator: {validation_text}")
        return {"validation_passed": False, "messages": [critique_message]}

def supervisor_node(state: AgentState):
    """A node that supervises the conversation and decides which node to call next."""
    st.info("Calling Supervisor Node")
    context = "\n".join([msg.pretty_repr() for msg in state["messages"][-3:]])
    
    prompt = f"""You are a supervisor in a multi-agent system. Your job is to decide the next action based on the conversation history.
    The user's request is the first message. Subsequent messages might be previous attempts or critiques from a validator.

    Choose the best tool for the next step:
    - 'llm': For Electronic gadgets and their specification available with LLM Trained data.
    - 'rag': To answer questions about Sony Play Station, based on an internal knowledge base.
    - 'web_search': For questions requiring real-time information on the new electronic devices which are recently released.

    Conversation History:
    {context}

    Based on the history, what is the best next action? Respond with only one of the following: 'llm', 'rag', 'web_search'.
    """
    model = get_llm()
    response = model.invoke(prompt)
    next_action = response.content.strip().lower()
    st.write(f"Supervisor decided to call: {next_action}")
    return {"next_action": next_action}

def router_function(state: AgentState):
    """A router that decides which node to call next based on the next action."""
    return state["next_action"]

def validation_router(state: AgentState):
    """A router that decides which node to call next based on the validation result."""
    if state["validation_passed"]:
        return END
    else:
        return "supervisor"

# Create the graph
@st.cache_resource
def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("validation", validation_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        router_function,
        {
            "llm": "llm",
            "rag": "rag",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("llm", "validation")
    workflow.add_edge("rag", "validation")
    workflow.add_edge("web_search", "validation")

    workflow.add_conditional_edges(
        "validation",
        validation_router,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    return workflow.compile()

# Streamlit app
def main():
    st.title("LangGraph Multi-Agent System")
    st.write("This app demonstrates a multi-agent system using LangGraph that can answer questions using different tools.")
    
    # Create a sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses a multi-agent system built with LangGraph to answer questions.
        
        The system includes:
        - A supervisor agent that decides which tool to use
        - An LLM agent for general knowledge
        - A RAG agent for PlayStation information
        - A web search agent for real-time information
        - A validation agent to ensure quality answers
        """)
        
        st.header("Available Tools")
        st.write("- LLM: For general knowledge about electronic gadgets")
        st.write("- RAG: For Sony PlayStation specific information")
        st.write("- Web Search: For real-time information about new devices")
    
    # User input
    query = st.text_input("Ask a question:", "Tell me about iPhone 13")
    
    if st.button("Submit"):
        with st.spinner("Processing your question..."):
            # Initialize the workflow
            app = create_workflow()
            
            # Create initial state
            initial_state = {"messages": [HumanMessage(content=query)]}
            
            # Create a container for the processing steps
            process_container = st.container()
            
            # Create a container for the final answer
            answer_container = st.container()
            
            with process_container:
                st.subheader("Processing Steps:")
                # Stream the output to see the steps
                for output in app.stream(initial_state, config={"recursion_limit": 10}):
                    pass  # Steps are displayed via st.info/st.success/st.error in the node functions
            
            # Get the final answer
            final_state = app.invoke(initial_state, config={"recursion_limit": 10})
            
            with answer_container:
                st.subheader("Final Answer:")
                st.write(final_state["generation"])

if __name__ == "__main__":
    main()
