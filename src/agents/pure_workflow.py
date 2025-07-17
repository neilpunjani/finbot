from typing import Dict, Any, List
import os
from .focused_agent import FocusedAgenticWorkflow

class PureAgenticWorkflow:
    """
    Pure agentic workflow using intelligent reasoning agents.
    No keywords, no hardcoded rules - just pure AI reasoning.
    """
    
    def __init__(self):
        print("🚀 Initializing Focused Two-Phase Agent...")
        
        # Use focused two-phase workflow for optimal performance
        self.focused_workflow = FocusedAgenticWorkflow()
        
        # Verify data sources are available
        self.data_sources = self._verify_data_sources()
        
        print("✅ Focused Two-Phase Agent initialized successfully!")
        print(f"📊 Discovered {len(self.data_sources)} data sources")
        print("🎯 Agent features: Discovery → Analysis, No sheet bouncing, Concrete answers")
    
    def _verify_data_sources(self) -> Dict[str, bool]:
        """Verify which data sources are available"""
        sources = {}
        
        # Check CSV data
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            sources['csv'] = len(csv_files) > 0
            print(f"📄 CSV: {len(csv_files)} files found")
        else:
            sources['csv'] = False
            print("❌ CSV: No data directory found")
        
        # Check Excel data
        excel_path = os.getenv("EXCEL_FILE_PATH")
        if excel_path and os.path.exists(excel_path):
            sources['excel'] = True
            print(f"📊 Excel: File found at {excel_path}")
        else:
            sources['excel'] = False
            print("❌ Excel: No file found")
        
        # Check SQL database
        if os.getenv("DATABASE_URL"):
            sources['sql'] = True
            print("🗄️ SQL: Database connection available")
        else:
            sources['sql'] = False
            print("❌ SQL: No database connection")
        
        # Check email
        if os.getenv("EMAIL_ADDRESS") and os.getenv("EMAIL_PASSWORD"):
            sources['email'] = True
            print("📧 Email: Connection available")
        else:
            sources['email'] = False
            print("❌ Email: No credentials found")
        
        return sources
    
    def process_query(self, query: str) -> str:
        """Process query using focused two-phase approach"""
        print(f"🎯 Focused Two-Phase Agent processing query: {query}")
        
        try:
            # Use the focused two-phase workflow for optimal performance
            response = self.focused_workflow.process_query(query)
            
            print("✅ Focused Agent completed two-phase analysis")
            return response
            
        except Exception as e:
            error_msg = f"🎯 Focused Agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_system_status(self) -> str:
        """Get the status of the focused two-phase system"""
        # Use the focused workflow's status method
        return self.focused_workflow.get_system_status() + f"\n\n**Data Sources Verified**: {len(self.data_sources)} sources discovered"
    
    def get_available_commands(self) -> str:
        """Get information about what the focused two-phase system can do"""
        # Use the focused workflow's commands method
        return self.focused_workflow.get_available_commands()
    
    def get_data_info(self) -> str:
        """Get information about available data (discovered dynamically)"""
        info = "🎯 **Focused Discovery Report**\n\n"
        info += "The agent uses two-phase approach for optimal performance:\n"
        info += "Phase 1: Quick relevance discovery | Phase 2: Focused analysis\n\n"
        
        # Available sources
        available_sources = [source for source, available in self.data_sources.items() if available]
        
        if available_sources:
            info += f"**Available for Focused Discovery**: {', '.join(available_sources).upper()}\n\n"
            
            info += "**Two-Phase Discovery Process**:\n"
            info += "🔍 **Phase 1 - Discovery**:\n"
            info += "   • Scans all data sources quickly\n"
            info += "   • Scores relevance for the query\n"
            info += "   • Selects top 2 most relevant sources\n\n"
            info += "🎯 **Phase 2 - Analysis**:\n"
            info += "   • Analyzes only pre-selected sources\n"
            info += "   • Focused analysis with clear objectives\n"
            info += "   • Provides concrete answers\n\n"
            
            info += "**Performance Benefits**:\n"
            info += "• No time wasted on irrelevant data\n"
            info += "• Fast discovery phase for source selection\n"
            info += "• Focused analysis for accuracy\n"
            info += "• No bouncing between multiple sources\n"
            info += "• Concrete answers, not endless loops\n"
        else:
            info += "❌ No data sources are currently available.\n"
            info += "Please check your environment configuration.\n"
        
        return info