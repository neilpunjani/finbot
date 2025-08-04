from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.prompts import PromptTemplate
import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PDFDocumentAgent:
    def __init__(self, vector_store=None):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=False
        )
        
        self.vector_store = vector_store
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_documents",
                description="Search through PDF documents for relevant information about policies, procedures, safety protocols, environmental compliance, or operational best practices. Use this to find specific information from mining policy documents.",
                func=self._search_documents
            ),
            Tool(
                name="get_document_summary",
                description="Get a summary of a specific PDF document's contents and main topics. Useful for understanding what information is available in each document.",
                func=self._get_document_summary
            ),
            Tool(
                name="list_available_documents",
                description="List all available PDF documents and their main topic areas. Use this to understand what policy documents are available for analysis.",
                func=self._list_available_documents
            )
        ]
    
    def _create_agent(self):
        # Create a custom ReAct prompt for PDF document analysis
        template = """You are a specialized PDF document analysis agent focused on mining industry policies and procedures.

Your capabilities include:
- Searching through PDF documents for specific information
- Analyzing policy documents for compliance requirements
- Extracting safety protocols and procedures
- Finding environmental regulations and guidelines
- Identifying operational best practices

Available document types:
- Health and Safety Policies
- Environmental Compliance & Sustainability
- Operational Best Practices & Equipment Usage

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- Always use the exact tool names: search_documents, get_document_summary, list_available_documents
- Action Input should be just the parameter value, not the function call format
- Example: Action: search_documents, Action Input: drilling procedures

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _search_documents(self, query: str) -> str:
        """Search through PDF documents for relevant information"""
        try:
            if not self.vector_store:
                return "No vector store available for document search."
            
            # Perform vector search
            results = self.vector_store.similarity_search_with_score(query, k=5)
            
            if not results:
                return "No relevant information found in the PDF documents."
            
            # Filter for PDF results and format
            formatted_results = []
            pdf_results = []
            
            for doc, score in results:
                metadata = doc.metadata
                if metadata.get('source_type') == 'pdf':
                    pdf_results.append((doc, score))
            
            if not pdf_results:
                return "No relevant information found in the PDF documents."
            
            for i, (doc, score) in enumerate(pdf_results[:5], 1):
                metadata = doc.metadata
                source_file = metadata.get('source_file', metadata.get('file_path', 'Unknown'))
                page = metadata.get('page', 'Unknown')
                
                formatted_results.append(
                    f"Result {i} (Relevance: {1-score:.3f}):\n"
                    f"Source: {os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'}\n"
                    f"Page: {page}\n"
                    f"Content: {doc.page_content}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _get_document_summary(self, document_name: str = "") -> str:
        """Get summary of available documents or a specific document"""
        try:
            if not self.vector_store:
                return "No vector store available for document summary."
            
            # Get all documents and filter for PDFs
            all_docs = self.vector_store.similarity_search("", k=100)  # Get many docs to filter
            pdf_docs = [doc for doc in all_docs if doc.metadata.get('source_type') == 'pdf']
            
            if not pdf_docs:
                return "No PDF documents found in the database."
            
            # Group by source file
            docs_info = {}
            for doc in pdf_docs:
                metadata = doc.metadata
                source = metadata.get('source_file', metadata.get('file_path', 'Unknown'))
                if source not in docs_info:
                    docs_info[source] = {
                        'schema': metadata.get('schema', {}),
                        'topics': metadata.get('topics', []),
                        'domain': metadata.get('domain', 'Unknown'),
                        'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
            
            if document_name:
                # Return specific document info
                for source, info in docs_info.items():
                    if document_name.lower() in source.lower():
                        schema = info['schema']
                        return f"Document: {os.path.basename(source)}\n" \
                               f"Domain: {info['domain']}\n" \
                               f"Main Topics: {', '.join(info['topics']) if info['topics'] else 'General content'}\n" \
                               f"Content Preview: {info['content_preview']}"
                return f"Document '{document_name}' not found."
            
            # Return all documents summary
            summary = "Available PDF Documents:\n\n"
            for source, info in docs_info.items():
                summary += f"• {os.path.basename(source)}\n"
                summary += f"  Domain: {info['domain']}\n"
                summary += f"  Topics: {', '.join(info['topics']) if info['topics'] else 'General content'}\n\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return f"Error getting document summary: {str(e)}"
    
    def _list_available_documents(self, query: str = "") -> str:
        """List all available PDF documents"""
        return self._get_document_summary()
    
    def query(self, question: str) -> str:
        """Process a question about PDF documents"""
        try:
            logger.info(f"PDF agent processing query: {question}")
            result = self.agent.invoke({"input": question})
            return result.get("output", "No response generated")
        except Exception as e:
            logger.error(f"Error in PDF agent query: {e}")
            return f"Error processing query: {str(e)}"