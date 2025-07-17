import os
import json
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class FastAgentResult:
    answer: str
    confidence: float
    data_sources_used: List[str]
    reasoning_steps: List[str]
    execution_time: float

class FastPureAgent:
    """
    Optimized pure agentic AI that minimizes API calls for faster responses.
    Combines planning and execution into single efficient calls.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use faster, cheaper model
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Discover available tools once
        self.available_tools = self._discover_tools()
        
        # Streamlined system prompt
        self.system_prompt = """
        You are an intelligent data analysis agent. Your goal is to provide fast, accurate answers.
        
        PROCESS:
        1. Quickly assess what data you need
        2. Choose the most relevant data source
        3. Analyze the data efficiently
        4. Provide a direct answer with confidence level
        
        PRINCIPLES:
        - Be fast and efficient
        - Provide direct answers, not lengthy plans
        - Use the most relevant data source
        - State your confidence level (High/Medium/Low)
        """
    
    def _discover_tools(self) -> Dict[str, Any]:
        """Quickly discover available tools"""
        tools = {}
        
        # CSV Data
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                tools['csv'] = csv_files
        
        # Excel Data
        excel_path = os.getenv("EXCEL_FILE_PATH")
        if excel_path and os.path.exists(excel_path):
            tools['excel'] = excel_path
        
        return tools
    
    def solve_fast(self, query: str) -> str:
        """Fast problem solving with minimal API calls"""
        start_time = datetime.now()
        
        # Single comprehensive prompt that handles planning and execution
        fast_prompt = PromptTemplate.from_template("""
        Answer this query quickly and efficiently using available data sources.
        
        QUERY: {query}
        
        AVAILABLE DATA SOURCES:
        {tools}
        
        INSTRUCTIONS:
        1. Determine what data you need to answer the query
        2. Choose the most relevant data source
        3. If you need to analyze data, provide the specific approach
        4. Give a direct answer with confidence level
        
        For financial queries like profit margins:
        - Look for revenue and cost data
        - Calculate: (Revenue - Costs) / Revenue * 100
        - Provide the percentage with supporting numbers
        
        Respond in this format:
        **ANSWER:** [Direct answer to the query]
        
        **DATA SOURCE:** [Which source you would use: CSV/Excel/etc]
        
        **APPROACH:** [Brief 1-2 sentence approach]
        
        **CONFIDENCE:** [High/Medium/Low with reasoning]
        
        **CALCULATION:** [If applicable, show the calculation]
        """)
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=fast_prompt.format(
                query=query,
                tools=json.dumps(self.available_tools, indent=2)
            ))
        ])
        
        # If we have actual data sources, do a quick data analysis
        if self.available_tools and any("profit" in query.lower() and year in query for year in ["2023", "2022", "2024"]):
            analysis_result = self._quick_data_analysis(query)
            if analysis_result:
                response.content += f"\n\n**ACTUAL DATA ANALYSIS:**\n{analysis_result}"
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Add performance info
        response.content += f"\n\n⚡ **Response Time:** {execution_time:.2f} seconds"
        response.content += f"\n🤖 **Agent Type:** Fast Pure Agent (Optimized)"
        
        return response.content
    
    def _quick_data_analysis(self, query: str) -> Optional[str]:
        """Quick data analysis for financial queries"""
        try:
            # Try Excel first for financial data
            if 'excel' in self.available_tools:
                excel_path = self.available_tools['excel']
                
                # Quick Excel analysis
                try:
                    # Load Excel file and explore worksheets
                    xls = pd.ExcelFile(excel_path)
                    
                    # Try to find the most relevant worksheet
                    target_sheet = None
                    for sheet_name in xls.sheet_names:
                        if any(keyword in sheet_name.lower() for keyword in ['financial', 'revenue', 'summary', 'data']):
                            target_sheet = sheet_name
                            break
                    
                    # Use first sheet if no obvious financial sheet found
                    if target_sheet is None:
                        target_sheet = xls.sheet_names[0]
                    
                    # Load the data
                    df = pd.read_excel(excel_path, sheet_name=target_sheet)
                    
                    # Look for specific regions and years in the query
                    regions_to_find = []
                    years_to_find = []
                    
                    # Extract regions from query
                    region_keywords = ['ontario', 'nova scotia', 'alberta', 'british columbia', 'quebec', 'manitoba', 'saskatchewan']
                    for region in region_keywords:
                        if region in query.lower():
                            regions_to_find.append(region)
                    
                    # Extract years from query
                    year_keywords = ['2023', '2022', '2024', '2021']
                    for year in year_keywords:
                        if year in query:
                            years_to_find.append(year)
                    
                    # Look for revenue/financial data
                    revenue_keywords = ['revenue', 'sales', 'income', 'earnings']
                    cost_keywords = ['cost', 'expense', 'expenditure']
                    
                    # Try to find relevant data
                    results = []
                    
                    # Check if we can find region and year data
                    for region in regions_to_find:
                        for year in years_to_find:
                            # Look for rows that match region and year
                            region_mask = df.astype(str).apply(
                                lambda x: x.str.contains(region, case=False, na=False)
                            ).any(axis=1)
                            
                            year_mask = df.astype(str).apply(
                                lambda x: x.str.contains(year, case=False, na=False)
                            ).any(axis=1)
                            
                            matching_rows = df[region_mask & year_mask]
                            
                            if not matching_rows.empty:
                                # Look for revenue columns
                                revenue_cols = []
                                for col in df.columns:
                                    if any(keyword in col.lower() for keyword in revenue_keywords):
                                        revenue_cols.append(col)
                                
                                if revenue_cols:
                                    for rev_col in revenue_cols:
                                        revenue_values = matching_rows[rev_col].dropna()
                                        if not revenue_values.empty:
                                            total_revenue = revenue_values.sum()
                                            results.append(f"{region.title()} {year} revenue: ${total_revenue:,.0f}")
                    
                    if results:
                        return f"Found actual data in {target_sheet}: " + "; ".join(results)
                    else:
                        # If no specific matches, provide general info about the data structure
                        sample_data = df.head(3).to_dict('records')
                        columns = list(df.columns)
                        return f"Excel file contains {len(df)} rows with columns: {columns[:5]}... Data structure available for analysis."
                
                except Exception as e:
                    return f"Excel analysis error: {str(e)}"
            
            # Try CSV as fallback
            if 'csv' in self.available_tools:
                csv_files = self.available_tools['csv']
                if csv_files:
                    try:
                        # Quick scan of CSV files
                        for csv_file in csv_files[:2]:  # Check first 2 CSV files
                            csv_path = os.path.join(os.getenv("CSV_DIRECTORY", "data/csv"), csv_file)
                            df = pd.read_csv(csv_path, nrows=10)
                            
                            # Check if this CSV might have relevant data
                            if any(keyword in query.lower() for keyword in ['revenue', 'financial']):
                                revenue_cols = [col for col in df.columns if 'revenue' in col.lower()]
                                if revenue_cols:
                                    return f"Found revenue data in CSV {csv_file}: columns {revenue_cols}"
                    
                    except Exception as e:
                        return f"CSV analysis error: {str(e)}"
            
            return None
            
        except Exception as e:
            return f"Data analysis error: {str(e)}"

class FastAgenticWorkflow:
    """
    Fast workflow wrapper that uses the optimized agent
    """
    
    def __init__(self):
        print("🚀 Initializing Fast Pure Agentic AI System...")
        self.agent = FastPureAgent()
        print("✅ Fast Pure Agentic AI System ready!")
    
    def process_query(self, query: str) -> str:
        """Process query with fast agent"""
        print(f"⚡ Fast agent processing: {query}")
        
        try:
            response = self.agent.solve_fast(query)
            print("✅ Fast agent completed")
            return response
        except Exception as e:
            error_msg = f"⚡ Fast agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_system_status(self) -> str:
        """Get fast system status"""
        return """⚡ **Fast Pure Agentic AI System**

**System Type**: Optimized Pure Agentic AI
**Model**: GPT-4o-mini (faster, efficient)
**Response Time**: ~2-5 seconds
**Approach**: Single-call optimization

**Capabilities**:
🚀 Fast financial analysis
⚡ Quick data discovery
📊 Direct answer generation
🎯 Confidence assessment

**Available Data Sources**:
""" + "\n".join([f"✅ {source}: {info}" for source, info in self.agent.available_tools.items()])
    
    def get_available_commands(self) -> str:
        """Get fast system commands"""
        return """⚡ **Fast Pure Agentic AI Capabilities**

**Financial Queries** (2-5 seconds):
- "What was the profit margin for Nova Scotia in 2023?"
- "Calculate ROI for our mining operations"
- "Show revenue trends by region"

**Data Analysis** (2-3 seconds):
- "Analyze operational efficiency"
- "Find cost optimization opportunities"
- "Compare regional performance"

**Business Intelligence** (3-5 seconds):
- "Identify key performance indicators"
- "Analyze market trends"
- "Provide strategic recommendations"

**Speed Optimizations**:
✅ Single API call per query
✅ GPT-4o-mini for speed
✅ Streamlined reasoning
✅ Direct data access
✅ Efficient prompting

Type your question for fast analysis!
"""