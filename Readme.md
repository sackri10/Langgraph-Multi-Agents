# LangGraph Multi-Agent Q&A System

This project demonstrates an advanced multi-agent question-answering system built using Python, LangGraph, Streamlit, and various LangChain components. The system can intelligently route user queries to different specialized "agents" (LLM, RAG, Web Search) based on the nature of the question, and then validate the generated answer for quality and relevance.

## Features

-   **Multi-Agent Architecture:** Utilizes LangGraph to create a stateful graph of interconnected agents.
-   **Intelligent Routing (Supervisor Agent):** A supervisor agent uses an LLM to analyze the query and conversation history to decide which specialized tool/agent to employ next.
-   **Specialized Tool Agents:**
    -   **LLM Agent:** Handles general knowledge questions using Google's Gemini model.
    -   **RAG Agent:** Performs Retrieval Augmented Generation on a local knowledge base (e.g., Sony PlayStation technical documents) for specific queries.
    -   **Web Search Agent:** Uses Tavily Search API for real-time information and answers requiring up-to-date knowledge.
-   **Validation Agent:** An LLM-based agent that critiques the generated answer for accuracy, relevance, and completeness.
-   **Self-Correction Loop:** If validation fails, the system can loop back to the supervisor with the critique, allowing for a revised attempt.
-   **Streamlit Web Interface:** Provides a user-friendly UI to interact with the agent system.
-   **State Management:** Leverages LangGraph's `AgentState` to maintain and pass information (like message history, generated answers, validation status) between agents.

## Core LangGraph Concepts Demonstrated

-   **`AgentState`:** Defines the shared memory and data structure for the graph.
-   **`Nodes`:** Python functions representing individual agents or processing steps.
-   **`Edges`:** Connections defining the flow between nodes.
-   **`Conditional Edges`:** Enables dynamic routing based on the current `AgentState`, crucial for the supervisor's decision-making and the validation loop.
-   **`StateGraph`:** The primary class for building these cyclical, stateful agentic workflows.
-   **`END`:** A special marker to terminate graph execution.

## Project Structure
.
├── app.py # Main Streamlit application and LangGraph logic
├── data/ # Directory for local knowledge base files
│ └── consoles.txt # Example knowledge base file for the RAG agent
├── .env.example # Example environment file (rename to .env)
├── requirements.txt # Python dependencies
└── README.md # This file

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.8 or higher:** [Download Python](https://www.python.org/downloads/)
2.  **Google Cloud Project & API Key:**
    -   For using Google Generative AI models (Gemini).
    -   Enable the "Vertex AI API".
    -   Obtain your **Google API Key**.
3.  **Hugging Face Account & Token (Optional but Recommended for Embeddings):**
    -   For using the `BAAI/bge-small-en` embedding model.
    -   Obtain your **Hugging Face Token** (read access is usually sufficient).
4.  **Tavily AI Account & API Key:**
    -   For the Web Search agent.
    -   Sign up at [tavily.com](https://tavily.com/).
    -   Obtain your **Tavily API Key**.
5.  **(Optional) LangSmith Account:**
    -   For tracing and debugging LangGraph executions.
    -   Obtain your **LangChain API Key**.

## Setup Instructions

1.  **Clone the Repository (or create your project folder):**
    If this code is in a Git repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    Otherwise, create a project directory and place `app.py` inside it.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    -   Windows: `.\venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    langchain
    langchain-core
    langchain-google-genai
    langchain-huggingface
    langchain-community
    langgraph
    python-dotenv
    faiss-cpu # or faiss-gpu if you have CUDA installed
    tavily-python
    # Ensure transformers and sentence-transformers are compatible with langchain-huggingface
    transformers
    sentence-transformers
    ```
    Then, install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of your project directory. You can copy `.env.example` if provided, or create it from scratch:
    ```env
    # .env

    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    HF_TOKEN="YOUR_HUGGINGFACE_TOKEN" # Optional, for certain HuggingFace models
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

    # LangSmith Configuration (Optional)
    # LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY"
    # LANGCHAIN_TRACING_V2="true" # Already set in app.py
    # LANGCHAIN_PROJECT="LangGraph-Multi-Agent" # Already set in app.py, customize if needed
    ```
    Replace the placeholder values with your actual API keys.

5.  **Prepare RAG Data (Optional but Recommended for Full Functionality):**
    -   Create a directory named `data` in your project root.
    -   Inside the `data` directory, create a text file named `consoles.txt`. Populate this file with information about Sony PlayStation consoles (or any other topic you want the RAG agent to be an expert on). The `DirectoryLoader` in `app.py` is configured to look for this file.
    Example `data/consoles.txt`:
    ```txt
    The PlayStation 5 (PS5) is a home video game console developed by Sony Interactive Entertainment.
    It was announced as the successor to the PlayStation 4 in April 2019, was launched on November 12, 2020, in Australia, Japan, New Zealand, North America, and South Korea, and November 19, 2020, for most other regions.
    The PS5 features a custom AMD Zen 2 CPU and an RDNA 2 GPU, providing significant performance improvements over its predecessor.
    It supports ray tracing, 4K resolution at 120Hz, and has a very fast custom SSD for near-instant load times.
    The DualSense controller offers haptic feedback and adaptive triggers.
    ```

## Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to your project directory** in the terminal.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  The application should open in your web browser, typically at `http://localhost:8501`.

## How to Use

1.  The application interface will load, showing a title and an input field.
2.  The sidebar provides information about the system and the available tools.
3.  **Ask a Question:** Type your question into the input field. Try questions that would logically map to different agents:
    *   **LLM Agent:** "What are the benefits of an AMOLED screen?"
    *   **RAG Agent:** "Tell me about the PS5 controller." (if `consoles.txt` is set up)
    *   **Web Search Agent:** "What are the latest announcements from Apple?"
4.  **Submit:** Click the "Submit" button.
5.  **Processing:** The application will show "Processing your question..." and display informational messages in the main area indicating which node (Supervisor, LLM, RAG, Web Search, Validation) is currently active.
6.  **View Answer:** Once the processing is complete (either the graph reaches `END` after successful validation or hits a recursion limit), the final generated answer will be displayed. If validation failed and the system looped, you might see evidence of the critique influencing subsequent attempts in the processing steps.

## Code Overview (`app.py`)

The `app.py` script contains the entire logic:

1.  **Imports and Setup:** Standard library imports, LangChain/LangGraph components, and environment variable loading.
2.  **LLM and Embeddings Initialization:** Functions `get_llm()` and `get_embeddings()` initialize and cache these models.
3.  **Retriever Setup (`get_retriever()`):** Loads documents from the `../data` directory, splits them, creates embeddings, and sets up a FAISS vector store retriever for the RAG agent.
4.  **`AgentState` Definition:** Defines the shared state class for the graph.
5.  **Node Definitions:**
    -   `llm_node()`: Calls the general-purpose LLM.
    -   `rag_node()`: Implements the RAG pipeline.
    -   `web_search_node()`: Performs web searches using Tavily.
    -   `validation_node()`: Validates the output of the tool nodes using an LLM.
    -   `supervisor_node()`: The core routing logic; uses an LLM to decide the next tool based on conversation history and tool descriptions.
6.  **Router Functions:**
    -   `router_function()`: Reads `state["next_action"]` for supervisor's conditional routing.
    -   `validation_router()`: Reads `state["validation_passed"]` for validation node's conditional routing.
7.  **Graph Creation (`create_workflow()`):**
    -   Initializes `StateGraph(AgentState)`.
    -   Adds all defined functions as nodes.
    -   Sets the entry point to the `supervisor` node.
    -   Defines conditional edges from the `supervisor` to tool nodes (based on `router_function`).
    -   Defines direct edges from tool nodes to the `validation` node.
    -   Defines conditional edges from the `validation` node (either to `END` or back to `supervisor`, based on `validation_router`).
    -   Compiles the graph.
8.  **Streamlit Application (`main()`):**
    -   Sets up the UI (title, sidebar, input field).
    -   On submission, initializes the LangGraph `app`.
    -   Creates the `initial_state` with the user's query.
    -   Invokes the graph using `app.stream()` (to show intermediate steps via `st.info` in nodes) and `app.invoke()` to get the final state.
    -   Displays the final answer.

## Customization and Troubleshooting

-   **Knowledge Base:** Modify the `DirectoryLoader` path and `glob` pattern in `get_retriever()` to point to your own documents for the RAG agent.
-   **Supervisor Prompts:** The prompt in `supervisor_node()` is crucial. You can refine the tool descriptions or the instructions to the supervisor LLM to improve routing accuracy.
-   **Validation Prompts:** The prompt in `validation_node()` determines the quality criteria. Adjust it to match your specific needs.
-   **Tool Agents:** Add, remove, or modify tool agents (nodes) as required. Remember to update the supervisor's prompt and the graph definition accordingly.
-   **Recursion Limit:** The `app.stream()` and `app.invoke()` calls have a `config={"recursion_limit": 10}`. This prevents infinite loops if, for example, validation consistently fails. You might need to adjust this.
-   **API Key Errors:** Double-check your `.env` file for correct API keys and ensure it's being loaded.
-   **LangSmith:** If you have LangSmith set up, it will be invaluable for visualizing the graph execution flow and debugging issues.

This project provides a solid foundation for building more complex, intelligent, and robust AI applications with LangGraph.
