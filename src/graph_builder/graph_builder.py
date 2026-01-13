"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.agent_state import RAGState
from src.nodes.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        """
        builder = StateGraph(RAGState)
        
        # 1. Add All Nodes
        builder.add_node("guardrail", self.nodes.guardrail)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # 2. Set Entry Point (CRITICAL: Must be guardrail)
        builder.set_entry_point("guardrail")
        
        # 3. Define Conditional Logic
        def check_guardrail(state: RAGState):
            # Check if the specific refusal message is present
            if state.answer and "I only answer department information" in state.answer:
                return "end"
            return "continue"

        # 4. Add Conditional Edges
        builder.add_conditional_edges(
            "guardrail",
            check_guardrail,
            {
                "end": END,           # Stop immediately if irrelevant
                "continue": "retriever" # Proceed to retrieval if relevant
            }
        )
        
        # 5. Connect the rest of the graph
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)