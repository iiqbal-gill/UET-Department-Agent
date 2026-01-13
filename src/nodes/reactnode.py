"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.agent_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
# Import standard libraries
import uuid

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def guardrail(self, state: RAGState) -> RAGState:
        """
        Strictly checks if the question is department-related.
        """
        print(f"--- GUARDRAIL CHECKING: {state.question} ---") # DEBUG PRINT
        
        system_prompt = (
            "You are a strict filter for a University Department chatbot. "
            "Your ONLY job is to classify if the question is about the University, "
            "Computer Science Department, Admissions, Faculty, Courses, or the Prospectus.\n"
            "Rules:\n"
            "1. Greetings (Hi, Hello) -> Reply 'NO'\n"
            "2. General Knowledge (What is Python?, Who is Obama?) -> Reply 'NO'\n"
            "3. Department Questions (Who is the HOD?, Fee structure?) -> Reply 'YES'\n\n"
            "Reply strictly with 'YES' or 'NO'."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.question)
        ]
        
        response = self.llm.invoke(messages)
        decision = response.content.strip().upper()
        
        print(f"--- GUARDRAIL DECISION: {decision} ---") # DEBUG PRINT
        
        # Check for NO (handling punctuation like NO.)
        if "NO" in decision:
            return RAGState(
                question=state.question,
                answer="I only answer department information."
            )
        
        # If YES, pass through (answer is empty)
        return RAGState(question=state.question, answer="")

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        print("--- RETRIEVING DOCUMENTS ---") # DEBUG PRINT
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    # ... (Keep _build_tools, _build_agent, generate_answer unchanged) ...
    def _build_tools(self) -> List[Tool]:
        # (Paste your existing _build_tools code here)
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )
        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        # (Paste your existing _build_agent code here)
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs only department related questions; Don't use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools,prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        # (Paste your existing generate_answer code here)
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )