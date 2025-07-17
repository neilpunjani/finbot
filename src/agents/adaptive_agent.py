import os
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import serialization function
def serialize_for_json(obj):
    """Convert dataclass objects to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                result[key] = serialize_for_json(value)
            else:
                result[key] = value
        return result
    else:
        return obj

class AdaptiveIntelligentAgent:
    """
    An adaptive agent that learns data structure automatically.
    Like ChatGPT - it figures out the data structure by itself.
    No manual structure hints needed.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Agent system prompt for adaptive intelligence
        self.system_prompt = """
        You are an extremely intelligent data analysis agent with adaptive learning capabilities.
        
        CORE INTELLIGENCE PRINCIPLES:
        1. AUTONOMOUS LEARNING: You figure out data structure by yourself, like ChatGPT would
        2. PATTERN RECOGNITION: You identify patterns in data without being told
        3. ADAPTIVE REASONING: You adjust your approach based on what you discover
        4. STRUCTURE DISCOVERY: You learn how data is organized through exploration
        5. INTELLIGENT INFERENCE: You make smart assumptions about data meaning
        
        YOUR APPROACH:
        1. EXPLORE: Examine the data structure intelligently
        2. DISCOVER: Find patterns and relationships autonomously
        3. INFER: Make intelligent assumptions about data organization
        4. ADAPT: Adjust your strategy based on discoveries
        5. SOLVE: Answer the query using your learned understanding
        
        You are as smart as ChatGPT at understanding data structures.
        You don't need to be told how data is organized - you figure it out.
        You learn by examining, not by being instructed.
        """
    
    def solve_adaptively(self, df: pd.DataFrame, query: str, sheet_name: str = "") -> str:
        """
        Solve query by adaptively learning the data structure
        """
        print(f"🧠 Adaptive Agent: Learning data structure autonomously...")
        
        # Create an adaptive pandas agent that learns the structure
        adaptive_prompt = f"""
        You are analyzing a dataset to answer this query: "{query}"
        
        Dataset info:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Worksheet: {sheet_name}
        
        BE SMART AND ADAPTIVE:
        
        1. FIRST: Intelligently explore the data structure
           - Examine column names and types
           - Look at sample data to understand patterns
           - Identify what type of data this is (financial, operational, etc.)
        
        2. LEARN: Figure out how this data is organized
           - Is it row-based or column-based?
           - Where is the information I need stored?
           - What are the patterns and relationships?
        
        3. DISCOVER: Find the relevant data
           - Use your intelligence to locate what the query asks for
           - Don't assume structure - discover it
           - Be like ChatGPT - figure it out yourself
        
        4. ADAPT: Adjust your approach based on what you learned
           - If it's row-based, search row values
           - If it's column-based, search columns
           - Use the structure you discovered
        
        5. SOLVE: Answer the query using your learned understanding
        
        IMPORTANT: Don't make assumptions about structure.
        Explore and learn like an intelligent human would.
        Show your learning process and discoveries.
        """
        
        # Create pandas agent with adaptive intelligence
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True,
            prefix=adaptive_prompt
        )
        
        # Let the agent learn and solve adaptively
        result = agent.run(f"Learn the data structure and answer: {query}")
        
        return result

class AdaptiveAgenticWorkflow:
    """
    Workflow that uses adaptive intelligence for any data structure
    """
    
    def __init__(self):
        print("🧠 Initializing Adaptive Intelligent Agent...")
        self.agent = AdaptiveIntelligentAgent()
        
        # Discover tools
        self.tools = self._discover_tools()
        print("✅ Adaptive Intelligence ready!")
        print("🎯 Agent will learn data structure autonomously")
    
    def _discover_tools(self) -> Dict[str, Any]:
        """Discover available tools"""
        tools = {}
        
        # Excel tools
        excel_path = os.getenv("EXCEL_FILE_PATH")
        if excel_path and os.path.exists(excel_path):
            tools['excel'] = excel_path
        
        # CSV tools
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                tools['csv'] = {'directory': csv_dir, 'files': csv_files}
        
        return tools
    
    def process_query(self, query: str) -> str:
        """Process query with adaptive intelligence"""
        print(f"🧠 Adaptive Agent processing: {query}")
        
        try:
            # Intelligent data source selection
            best_result = None
            best_confidence = 0
            
            # Try Excel data with adaptive learning
            if 'excel' in self.tools:
                print("🧠 Trying Excel data with adaptive learning...")
                excel_result = self._try_excel_adaptively(query)
                if excel_result and "not found" not in excel_result.lower():
                    confidence = self._assess_result_confidence(excel_result, query)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = f"📊 Excel Analysis:\n{excel_result}"
            
            # Try CSV data with adaptive learning
            if 'csv' in self.tools and (best_confidence < 0.8):
                print("🧠 Trying CSV data with adaptive learning...")
                csv_result = self._try_csv_adaptively(query)
                if csv_result and "not found" not in csv_result.lower():
                    confidence = self._assess_result_confidence(csv_result, query)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = f"📄 CSV Analysis:\n{csv_result}"
            
            if best_result:
                return f"{best_result}\n\n🧠 **Adaptive Intelligence**: Learned data structure autonomously (confidence: {best_confidence:.1f})"
            else:
                return "🤔 Adaptive learning in progress - data structure requires more exploration"
                
        except Exception as e:
            return f"🧠 Adaptive learning error: {str(e)}"
    
    def _try_excel_adaptively(self, query: str) -> Optional[str]:
        """Try Excel data with adaptive learning"""
        try:
            excel_path = self.tools['excel']
            xls = pd.ExcelFile(excel_path)
            
            # Try worksheets intelligently
            best_result = None
            best_score = 0
            
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    
                    if df.shape[0] < 10:  # Skip small datasets
                        continue
                    
                    # Let the adaptive agent learn and solve
                    result = self.agent.solve_adaptively(df, query, sheet_name)
                    
                    # Score the result
                    score = self._score_result(result, query)
                    
                    if score > best_score:
                        best_score = score
                        best_result = f"Worksheet: {sheet_name}\n{result}"
                
                except Exception as e:
                    print(f"   ⚠️ Error with {sheet_name}: {e}")
                    continue
            
            return best_result
            
        except Exception as e:
            print(f"❌ Excel adaptive learning error: {e}")
            return None
    
    def _try_csv_adaptively(self, query: str) -> Optional[str]:
        """Try CSV data with adaptive learning"""
        try:
            csv_info = self.tools['csv']
            csv_dir = csv_info['directory']
            csv_files = csv_info['files']
            
            # Try CSV files intelligently
            best_result = None
            best_score = 0
            
            for csv_file in csv_files[:3]:  # Try first 3 CSV files
                try:
                    csv_path = os.path.join(csv_dir, csv_file)
                    df = pd.read_csv(csv_path)
                    
                    if df.shape[0] < 10:  # Skip small datasets
                        continue
                    
                    # Let the adaptive agent learn and solve
                    result = self.agent.solve_adaptively(df, query, csv_file)
                    
                    # Score the result
                    score = self._score_result(result, query)
                    
                    if score > best_score:
                        best_score = score
                        best_result = f"File: {csv_file}\n{result}"
                
                except Exception as e:
                    print(f"   ⚠️ Error with {csv_file}: {e}")
                    continue
            
            return best_result
            
        except Exception as e:
            print(f"❌ CSV adaptive learning error: {e}")
            return None
    
    def _score_result(self, result: str, query: str) -> float:
        """Score how good a result is"""
        if not result:
            return 0.0
        
        result_lower = result.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Check if result contains numbers (good for financial queries)
        if any(char.isdigit() for char in result):
            score += 0.3
        
        # Check if result contains currency symbols
        if any(symbol in result for symbol in ['$', '€', '£', 'USD', 'CAD']):
            score += 0.2
        
        # Check if result contains query keywords
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in result_lower:
                score += 0.1
        
        # Check for calculation indicators
        if any(word in result_lower for word in ['total', 'sum', 'calculated', 'analysis']):
            score += 0.2
        
        # Penalty for "not found" type messages
        if any(phrase in result_lower for phrase in ['not found', 'no data', 'cannot find', 'unable to']):
            score -= 0.5
        
        return min(score, 1.0)
    
    def _assess_result_confidence(self, result: str, query: str) -> float:
        """Assess confidence in the result"""
        # Use the same scoring logic for now
        return self._score_result(result, query)
    
    def get_system_status(self) -> str:
        """Get system status"""
        return """🧠 **Adaptive Intelligent Agent Status**

**System Type**: Adaptive Intelligence with Autonomous Learning
**Agent Model**: GPT-4o with Adaptive Data Structure Discovery
**Learning Method**: Autonomous pattern recognition (like ChatGPT)

**Adaptive Capabilities**:
🧠 Autonomous data structure learning
🔍 Pattern recognition without hints
📊 Intelligent data organization discovery
🎯 Adaptive strategy adjustment
🔄 Self-learning and exploration
✨ ChatGPT-level intelligence for data understanding

**Learning Process**:
1. **Explore**: Examines data structure intelligently
2. **Discover**: Finds patterns autonomously
3. **Infer**: Makes smart assumptions about organization
4. **Adapt**: Adjusts approach based on discoveries
5. **Solve**: Answers using learned understanding

**No Manual Structure Hints Needed**:
✅ Learns row-based vs column-based automatically
✅ Discovers financial data patterns
✅ Identifies entity and time relationships
✅ Adapts to any data organization
✅ Works like ChatGPT - figures it out itself

**Available Data Sources**: """ + f"{len(self.tools)} sources discovered"
    
    def get_available_commands(self) -> str:
        """Get available commands"""
        return """🧠 **Adaptive Intelligent Agent Capabilities**

**Autonomous Learning Examples**:
- "What was revenue in 2023 for Ontario?"
  → Agent learns data structure and finds answer
- "Calculate profit margin for any region"
  → Agent discovers how profit data is organized
- "Find the highest performing entity"
  → Agent learns performance metrics organization

**Adaptive Intelligence Features**:
🧠 **Autonomous Structure Discovery**: No hints needed about data organization
🔍 **Pattern Recognition**: Finds relationships like ChatGPT would
📊 **Intelligent Exploration**: Examines data structure systematically
🎯 **Smart Adaptation**: Adjusts approach based on discoveries
✨ **ChatGPT-level Understanding**: Figures out data meaning independently

**Learning Capabilities**:
- Row-based financial data (Level1/Level2/Level3 structure)
- Column-based tabular data (traditional format)
- Hierarchical account structures (account codes + categories)
- Time-series data (dates, periods, years)
- Entity-based data (regions, offices, departments)
- Multi-dimensional data (any complex structure)

**How It Works**:
1. **Explores** data structure intelligently
2. **Discovers** patterns and relationships
3. **Learns** data organization autonomously
4. **Adapts** strategy based on structure
5. **Solves** query using learned understanding

**Intelligence Level**: Like ChatGPT - no training needed!
**Structure Hints**: None required - agent learns everything
**Adaptation**: Automatic for any data format

Just ask your question - the agent will figure out your data!
"""