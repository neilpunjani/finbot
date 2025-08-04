from typing import Dict, Any, List
import os
import asyncio
from .rag_enhanced_workflow import RAGEnhancedWorkflow

class PureAgenticWorkflow:
    """
    Pure agentic workflow using RAG-enhanced discovery + intelligent analysis.
    Fast vector-based discovery with expert domain analysis.
    """
    
    def __init__(self, use_rag: bool = True, rebuild_index: bool = False):
        if use_rag:
            print("Initializing RAG-Enhanced Agentic Workflow...")
            
            # Use RAG-enhanced workflow for maximum performance
            self.rag_workflow = RAGEnhancedWorkflow(rebuild_index=rebuild_index)
            self.use_rag = True
            
            print("RAG-Enhanced Workflow initialized successfully!")
            print("Flow: Vector Discovery -> Schema Selection -> Expert Analysis")
        else:
            print("🚀 Initializing Legacy Focused Two-Phase Agent...")
            
            # Fallback to old system if needed
            from .focused_agent import FocusedAgenticWorkflow
            self.focused_workflow = FocusedAgenticWorkflow()
            self.use_rag = False
            
            print("✅ Legacy Focused Agent initialized!")
        
        # Verify data sources are available
        self.data_sources = self._verify_data_sources()
    
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
        """Process query using RAG-enhanced or legacy approach"""
        
        try:
            if self.use_rag:
                # Use RAG-enhanced workflow for maximum performance
                response = self.rag_workflow.process_query(query)
                print("✅ RAG-Enhanced Agent completed analysis")
                return response
            else:
                # Fallback to legacy focused workflow
                print(f"🎯 Legacy Focused Agent processing query: {query}")
                response = self.focused_workflow.process_query(query)
                print("✅ Legacy Focused Agent completed analysis")
                return response
            
        except Exception as e:
            system_type = "RAG-Enhanced" if self.use_rag else "Legacy Focused"
            error_msg = f"❌ {system_type} Agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def process_query_async(self, query: str) -> str:
        """PERFORMANCE OPTIMIZED: Async query processing to avoid blocking operations"""
        
        try:
            if self.use_rag:
                # Use RAG-enhanced workflow for maximum performance (run in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.rag_workflow.process_query, query)
                print("✅ RAG-Enhanced Agent completed async analysis")
                return response
            else:
                # Fallback to legacy focused workflow (run in thread pool to avoid blocking)
                print(f"🎯 Legacy Focused Agent processing async query: {query}")
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.focused_workflow.process_query, query)
                print("✅ Legacy Focused Agent completed async analysis")
                return response
            
        except Exception as e:
            system_type = "RAG-Enhanced" if self.use_rag else "Legacy Focused"
            error_msg = f"❌ {system_type} Async Agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_system_status(self) -> str:
        """Get the status of the current system"""
        if self.use_rag:
            # Use RAG-enhanced workflow status
            return self.rag_workflow.get_system_status() + f"\n\n**Environment Sources Verified**: {len(self.data_sources)} source types available"
        else:
            # Use legacy workflow status
            return self.focused_workflow.get_system_status() + f"\n\n**Data Sources Verified**: {len(self.data_sources)} sources discovered"
    
    def get_available_commands(self) -> str:
        """Get information about what the current system can do"""
        if self.use_rag:
            # Use RAG-enhanced workflow commands
            return self.rag_workflow.get_available_commands()
        else:
            # Use legacy workflow commands
            return self.focused_workflow.get_available_commands()
    
    def rebuild_rag_index(self) -> str:
        """Rebuild RAG index (only available in RAG mode)"""
        if self.use_rag:
            return self.rag_workflow.rebuild_index()
        else:
            return "❌ RAG index rebuild only available in RAG mode. Initialize with use_rag=True."
    
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