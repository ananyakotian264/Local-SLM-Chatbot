import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama

# ==========================================
# 1. Build the LangGraph Workflow
# ==========================================
# We use @st.cache_resource so Streamlit doesn't rebuild the graph on every interaction
@st.cache_resource
def build_graph():
    # Initialize the local SLM via Ollama
    # If you downloaded a different model (e.g., "phi3" or "qwen2.5:0.5b"), change it here.
    llm = ChatOllama(model="llama3", temperature=0.7)

    # Define the node function that calls our model
    def call_model(state: MessagesState):
        # The state contains a list of messages. We pass them directly to the LLM.
        response = llm.invoke(state["messages"])
        # Return the new message to be appended to the state
        return {"messages": [response]}

    # Define the StateGraph using the built-in MessagesState
    workflow = StateGraph(MessagesState)
    
    # Add our single node
    workflow.add_node("assistant", call_model)
    
    # Define the flow (Start -> Assistant -> End)
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)
    
    # Compile the graph into a runnable application
    app = workflow.compile()
    return app

# Initialize the graph
graph_app = build_graph()

# ==========================================
# 2. Build the Streamlit User Interface
# ==========================================
st.set_page_config(page_title="Local LangGraph Agent", page_icon="🤖")
st.title("🤖 Local SLM Chatbot")
st.markdown("Powered by **Ollama**, **LangGraph**, and **Streamlit**.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat messages on the screen
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Capture user input
if prompt := st.chat_input("Ask me anything..."):
    
    # 1. Append user message to Streamlit UI state
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Pass the entire conversation history to LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Running locally)..."):
            
            # LangGraph expects a dictionary with the 'messages' key
            inputs = {"messages": st.session_state.messages}
            
            # Invoke the graph
            result = graph_app.invoke(inputs)
            
            # The updated state contains all messages; grab the last one (the AI's response)
            ai_response = result["messages"][-1]
            
            # Display the result
            st.markdown(ai_response.content)
            
            # 3. Save the AI's response back to Streamlit's session state memory
            st.session_state.messages.append(ai_response)