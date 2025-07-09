from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import os

from .router_agent import RouterAgent
from .sql_agent import SQLAgent
from .excel_agent import ExcelAgent
from .csv_agent import CSVAgent
from .email_agent import EmailAgent

class ChatbotState(TypedDict):
    query: str
    routing_decision: Dict[str, Any]
    agent_response: str
    error_message: str
    attempt_count: int
    fallback_attempted: bool

class ChatbotWorkflow:
    def __init__(self):
        self.router = RouterAgent()
        
        # Initialize agents (with error handling)
        self.agents = {}
        self._initialize_agents()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _initialize_agents(self):
        """Initialize all agents with error handling"""
        try:
            self.agents['sql'] = SQLAgent()
        except Exception as e:
            print(f"Warning: Could not initialize SQL agent: {e}")
            self.agents['sql'] = None
        
        try:
            self.agents['excel'] = ExcelAgent()
        except Exception as e:
            print(f"Warning: Could not initialize Excel agent: {e}")
            self.agents['excel'] = None
        
        try:
            self.agents['csv'] = CSVAgent()
        except Exception as e:
            print(f"Warning: Could not initialize CSV agent: {e}")
            self.agents['csv'] = None
        
        try:
            self.agents['email'] = EmailAgent()
        except Exception as e:
            print(f"Warning: Could not initialize Email agent: {e}")
            self.agents['email'] = None
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("try_fallback", self._try_fallback)
        
        # Define edges
        workflow.set_entry_point("route_query")
        
        workflow.add_edge("route_query", "execute_query")
        
        # Conditional edges from execute_query
        workflow.add_conditional_edges(
            "execute_query",
            self._should_retry,
            {
                "success": END,
                "error": "handle_error",
                "fallback": "try_fallback"
            }
        )
        
        workflow.add_edge("handle_error", END)
        workflow.add_edge("try_fallback", "execute_query")
        
        return workflow
    
    def _route_query(self, state: ChatbotState) -> ChatbotState:
        """Route the query to determine which agent should handle it"""
        try:
            routing_decision = self.router.route_query(state["query"])
            state["routing_decision"] = routing_decision
            state["attempt_count"] = 0
            state["fallback_attempted"] = False
            state["error_message"] = ""
            
        except Exception as e:
            state["error_message"] = f"Error in routing: {str(e)}"
            state["routing_decision"] = {
                "data_source": "sql",  # Default fallback
                "confidence": "low",
                "reasoning": "Routing failed, using default"
            }
        
        return state
    
    def _execute_query(self, state: ChatbotState) -> ChatbotState:
        """Execute the query using the selected agent"""
        try:
            data_source = state["routing_decision"]["data_source"]
            query = state["query"]
            
            # Check if agent is available
            if data_source not in self.agents or self.agents[data_source] is None:
                state["error_message"] = f"{data_source.upper()} agent is not available"
                return state
            
            # Execute query
            agent = self.agents[data_source]
            response = agent.query(query)
            
            # Enhanced response with mandatory source attribution
            enhanced_response = self._enhance_response_with_source(response, state["routing_decision"], data_source)
            state["agent_response"] = enhanced_response
            state["error_message"] = ""
            
        except Exception as e:
            state["error_message"] = f"Error executing query with {data_source} agent: {str(e)}"
            state["attempt_count"] += 1
        
        return state
    
    def _handle_error(self, state: ChatbotState) -> ChatbotState:
        """Handle errors and provide fallback response"""
        error_msg = state["error_message"]
        
        # Try to provide helpful error message
        if "not found" in error_msg.lower():
            state["agent_response"] = f"I couldn't find the required data source. Please check your configuration. Error: {error_msg}"
        elif "connection" in error_msg.lower():
            state["agent_response"] = f"There was a connection issue. Please check your network and credentials. Error: {error_msg}"
        else:
            state["agent_response"] = f"I encountered an error while processing your query. Error: {error_msg}"
        
        return state
    
    def _try_fallback(self, state: ChatbotState) -> ChatbotState:
        """Try alternative agents if primary agent fails"""
        if state["fallback_attempted"]:
            return state
        
        # Try alternative sources
        alternative_sources = state["routing_decision"].get("alternative_sources", [])
        
        for alt_source in alternative_sources:
            if alt_source in self.agents and self.agents[alt_source] is not None:
                state["routing_decision"]["data_source"] = alt_source
                state["routing_decision"]["reasoning"] = f"Fallback to {alt_source} after primary source failed"
                state["fallback_attempted"] = True
                return state
        
        # If no alternatives, try any available agent
        available_agents = [name for name, agent in self.agents.items() if agent is not None]
        if available_agents:
            fallback_agent = available_agents[0]
            state["routing_decision"]["data_source"] = fallback_agent
            state["routing_decision"]["reasoning"] = f"Fallback to {fallback_agent} (only available agent)"
            state["fallback_attempted"] = True
        
        return state
    
    def _should_retry(self, state: ChatbotState) -> str:
        """Determine if we should retry, use fallback, or end"""
        if state["error_message"]:
            if state["attempt_count"] < 2 and not state["fallback_attempted"]:
                return "fallback"
            else:
                return "error"
        else:
            return "success"
    
    def _enhance_response_with_source(self, response: str, routing_decision: Dict[str, Any], data_source: str) -> str:
        """Enhance the response with mandatory source attribution"""
        # Extract source information from the response if available
        source_info = self._extract_source_info(response, data_source)
        
        # Create header with source attribution
        header = f"📊 **DATA SOURCE**: {source_info}\n"
        header += f"🎯 **QUERY TYPE**: {routing_decision.get('query_type', 'Analysis').title()}\n"
        header += f"✅ **CONFIDENCE**: {routing_decision.get('confidence', 'Medium').title()}\n\n"
        
        # Clean response (remove any existing source info to avoid duplication)
        clean_response = self._clean_response(response)
        
        enhanced = header + clean_response
        
        # Add confidence note if needed
        if routing_decision.get('confidence') == 'low':
            enhanced += "\n\n⚠️ **Note**: This response has low confidence. Consider rephrasing your query for better results."
        
        return enhanced
    
    def _extract_source_info(self, response: str, data_source: str) -> str:
        """Extract specific source information from agent response"""
        data_source_upper = data_source.upper()
        
        if data_source == 'excel':
            # Try to extract worksheet information from Excel agent response
            if 'worksheet' in response.lower() or 'sheet' in response.lower():
                # Look for patterns like "Sales_Summary worksheet" or "from Budget_Analysis"
                import re
                sheet_patterns = [
                    r'(\w+_\w+)\s+worksheet',
                    r'(\w+_\w+)\s+sheet',
                    r'from\s+(\w+_\w+)',
                    r'Sheet\s+[\'"](\w+_\w+)[\'"]',
                    r'(Sales_Summary|Budget_Analysis|Product_Performance|Employee_Data|Regional_Sales)'
                ]
                
                for pattern in sheet_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    if matches:
                        sheets = list(set(matches))  # Remove duplicates
                        if len(sheets) == 1:
                            return f"{data_source_upper} File → {sheets[0]} Sheet"
                        else:
                            return f"{data_source_upper} File → Multiple Sheets: {', '.join(sheets)}"
            
            return f"{data_source_upper} File → Multiple Worksheets"
        
        elif data_source == 'csv':
            # Try to extract CSV file information
            if 'sales' in response.lower() and 'csv' in response.lower():
                return f"{data_source_upper} File → sales_data.csv"
            elif 'customer' in response.lower() and 'csv' in response.lower():
                return f"{data_source_upper} File → customer_data.csv"
            else:
                return f"{data_source_upper} File → CSV Dataset"
        
        elif data_source == 'sql':
            # Try to extract table information
            if 'table' in response.lower():
                import re
                table_matches = re.findall(r'table\s+[\'"]?(\w+)[\'"]?', response, re.IGNORECASE)
                if table_matches:
                    return f"{data_source_upper} Database → {table_matches[0]} Table"
            return f"{data_source_upper} Database → SQL Tables"
        
        elif data_source == 'email':
            return f"{data_source_upper} → Outlook/Exchange Mailbox"
        
        return f"{data_source_upper} Data Source"
    
    def _clean_response(self, response: str) -> str:
        """Clean response by removing existing source attribution"""
        # Remove existing patterns like [CSV Agent - Analysis]
        import re
        patterns_to_remove = [
            r'\[\w+\s+Agent[^\]]*\]\s*\n*',
            r'\[\w+\s+CSV\s+Analysis\]\s*\n*',
            r'\[\w+\s+Data\s+Source[^\]]*\]\s*\n*'
        ]
        
        cleaned = response
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def process_query(self, query: str) -> str:
        """Main method to process a user query"""
        try:
            # Initialize state
            initial_state = {
                "query": query,
                "routing_decision": {},
                "agent_response": "",
                "error_message": "",
                "attempt_count": 0,
                "fallback_attempted": False
            }
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            return result["agent_response"]
            
        except Exception as e:
            return f"Workflow error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get the status of all agents"""
        status = "System Status:\n\n"
        
        for agent_name, agent in self.agents.items():
            if agent is not None:
                # Special check for email agent connectivity
                if agent_name == 'email' and hasattr(agent, 'account') and agent.account is None:
                    status += f"❌ {agent_name.upper()} Agent: Not Available (not connected)\n"
                else:
                    status += f"✅ {agent_name.upper()} Agent: Available\n"
            else:
                status += f"❌ {agent_name.upper()} Agent: Not Available\n"
        
        status += f"\n🔄 Router Agent: Available"
        
        return status
    
    def get_available_commands(self) -> str:
        """Get list of available commands and examples"""
        commands = """
Available Commands and Examples:

SQL Database Queries:
- "Show me the total sales for last month"
- "Count the number of customers in the database"
- "What are the top 5 products by revenue?"

Excel File Analysis:
- "Analyze the budget spreadsheet"
- "Compare data across different worksheets"
- "What's the total in the financial report?"

CSV File Queries:
- "Show me the summary of the sales data"
- "What are the trends in the customer data?"
- "Calculate the average from the CSV file"

Email Analysis:
- "Find emails about budget from last week"
- "Show me all emails from John Smith"
- "What are the main topics in my recent emails?"

System Commands:
- "What data sources are available?"
- "Show system status"
- "Help" or "What can you do?"
        """
        
        return commands