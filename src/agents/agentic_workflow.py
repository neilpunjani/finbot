from typing import Dict, Any, List
from typing_extensions import TypedDict
import os

from .react_orchestrator import ReActOrchestrator
from .agentic_csv_agent import AgenticCSVAgent
from .agentic_excel_agent import AgenticExcelAgent
from .sql_agent import SQLAgent
from .email_agent import EmailAgent

class AgenticChatbotWorkflow:
    def __init__(self):
        # Initialize agentic agents
        self.agents = {}
        self._initialize_agents()
        
        # Create the ReAct orchestrator
        self.orchestrator = ReActOrchestrator(
            csv_agent=self.agents.get('csv'),
            excel_agent=self.agents.get('excel'),
            sql_agent=self.agents.get('sql'),
            email_agent=self.agents.get('email')
        )
        
        print("🤖 Agentic AI System initialized with reasoning capabilities")
    
    def _initialize_agents(self):
        """Initialize all agentic agents with error handling"""
        
        try:
            self.agents['csv'] = AgenticCSVAgent()
            print("✅ Agentic CSV agent initialized")
        except Exception as e:
            print(f"❌ Could not initialize Agentic CSV agent: {e}")
            self.agents['csv'] = None
        
        try:
            self.agents['excel'] = AgenticExcelAgent()
            print("✅ Agentic Excel agent initialized")
        except Exception as e:
            print(f"❌ Could not initialize Agentic Excel agent: {e}")
            self.agents['excel'] = None
        
        try:
            self.agents['sql'] = SQLAgent()
            print("✅ SQL agent initialized")
        except Exception as e:
            print(f"❌ Could not initialize SQL agent: {e}")
            self.agents['sql'] = None
        
        try:
            self.agents['email'] = EmailAgent()
            print("✅ Email agent initialized")
        except Exception as e:
            print(f"❌ Could not initialize Email agent: {e}")
            self.agents['email'] = None
    
    def process_query(self, query: str) -> str:
        """Process query using agentic reasoning"""
        try:
            # Use the ReAct orchestrator for intelligent reasoning
            response = self.orchestrator.process_query(query)
            
            # Format the response with reasoning steps
            formatted_response = self._format_agentic_response(response)
            
            return formatted_response
            
        except Exception as e:
            return f"Agentic reasoning error: {str(e)}"
    
    def _format_agentic_response(self, agent_response) -> str:
        """Format the agentic response with reasoning transparency"""
        
        # Main response
        response_text = f"🤖 **Agentic AI Analysis**\n\n"
        response_text += f"{agent_response.content}\n\n"
        
        # Sources used
        if agent_response.sources_used:
            response_text += f"📊 **Data Sources Used**: {', '.join(agent_response.sources_used).upper()}\n\n"
        
        # Reasoning transparency (optional - can be toggled)
        if hasattr(agent_response, 'reasoning_steps') and agent_response.reasoning_steps:
            response_text += f"🧠 **Reasoning Process**:\n"
            for i, step in enumerate(agent_response.reasoning_steps, 1):
                response_text += f"{i}. **Thought**: {step.thought[:100]}...\n"
                response_text += f"   **Action**: {step.action}\n"
                response_text += f"   **Result**: {step.observation[:100]}...\n\n"
        
        # Confidence level
        confidence_emoji = {
            "high": "🟢",
            "medium": "🟡", 
            "low": "🔴"
        }
        emoji = confidence_emoji.get(agent_response.confidence.value, "🟡")
        response_text += f"{emoji} **Confidence**: {agent_response.confidence.value.title()}\n"
        
        # Additional data suggestion
        if agent_response.needs_more_data:
            response_text += f"\n💡 **Suggestion**: This analysis could be enhanced with additional data sources or more specific queries.\n"
        
        return response_text
    
    def get_system_status(self) -> str:
        """Get the status of the agentic system"""
        status = "🤖 Agentic AI System Status:\n\n"
        
        # Agent status
        for agent_name, agent in self.agents.items():
            if agent is not None:
                if agent_name in ['csv', 'excel']:
                    status += f"✅ {agent_name.upper()} Agent: Agentic reasoning enabled\n"
                else:
                    status += f"✅ {agent_name.upper()} Agent: Available\n"
            else:
                status += f"❌ {agent_name.upper()} Agent: Not Available\n"
        
        # Orchestrator status
        status += f"\n🧠 ReAct Orchestrator: Active\n"
        status += f"🔄 Cross-source reasoning: Enabled\n"
        status += f"📊 Confidence scoring: Enabled\n"
        status += f"🎯 Multi-step planning: Enabled\n"
        
        return status
    
    def get_available_commands(self) -> str:
        """Get list of available commands with agentic capabilities"""
        commands = """
🤖 **Agentic AI Capabilities**:

**Complex Financial Analysis**:
- "Calculate profit margin for Nova Scotia in 2023"
- "Compare revenue growth across all entities"
- "Analyze cash flow trends and provide recommendations"
- "What's the ROI for our mining operations?"

**Multi-Source Reasoning**:
- "How does gold production correlate with financial performance?"
- "Compare operational efficiency with environmental impact"
- "Analyze workforce productivity against safety metrics"

**Intelligent Data Discovery**:
- "Find all revenue sources and categorize them"
- "Identify cost centers and their impact on profitability"
- "Discover trends in our operational data"

**Business Intelligence**:
- "What are the key performance indicators I should monitor?"
- "Identify areas for operational improvement"
- "Analyze our competitive position based on available data"

**Self-Reflective Analysis**:
- The system will indicate confidence levels
- It can suggest additional data sources if needed
- It provides reasoning transparency for complex queries
- It can identify when it needs more information

**System Commands**:
- "What data sources are available?"
- "Show system status"
- "Help" or "What can you do?"
        """
        
        return commands
    
    def get_data_info(self) -> str:
        """Get comprehensive information about available data"""
        info = "📊 **Available Data Sources for Agentic Analysis**:\n\n"
        
        # CSV data info
        if self.agents['csv']:
            info += "**CSV Data (Operational Intelligence)**:\n"
            info += self.agents['csv'].get_data_info()
            info += "\n"
        
        # Excel data info
        if self.agents['excel']:
            info += "**Excel Data (Financial Intelligence)**:\n"
            info += self.agents['excel'].get_data_info()
            info += "\n"
        
        # Other agents info
        if self.agents['sql']:
            info += "**SQL Database**: Available for structured queries\n"
        if self.agents['email']:
            info += "**Email Analysis**: Available for communication insights\n"
        
        info += "\n🧠 **Agentic Capabilities**: The system can reason across all these sources to provide comprehensive insights.\n"
        
        return info