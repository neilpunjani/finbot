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

@dataclass
class SheetAnalysis:
    name: str
    relevance_score: float
    data_summary: str
    recommended: bool
    reason: str

class DataDiscoveryAgent:
    """
    PHASE 1: Quick data discovery agent that decides which sheets to analyze
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use faster model for discovery
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def discover_relevant_sheets(self, excel_path: str, query: str) -> List[SheetAnalysis]:
        """Quickly discover which sheets are relevant for the query"""
        print(f"🔍 PHASE 1: Discovering relevant sheets for query: {query}")
        
        xls = pd.ExcelFile(excel_path)
        sheet_analyses = []
        
        # Quick scan of all sheets
        for sheet_name in xls.sheet_names:
            try:
                # Load only first few rows for speed
                df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=10)
                
                if df.empty or df.shape[1] < 2:
                    continue
                
                # Quick relevance analysis
                analysis = self._analyze_sheet_relevance(sheet_name, df, query)
                sheet_analyses.append(analysis)
                
                print(f"   📄 {sheet_name}: Score {analysis.relevance_score:.1f} - {analysis.reason}")
                
            except Exception as e:
                print(f"   ❌ Error reading {sheet_name}: {e}")
                continue
        
        # Sort by relevance and return top candidates
        sheet_analyses.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Mark top sheets as recommended
        for i, analysis in enumerate(sheet_analyses):
            analysis.recommended = i < 2 and analysis.relevance_score > 3.0  # Top 2 with score > 3
        
        recommended_count = sum(1 for a in sheet_analyses if a.recommended)
        print(f"🎯 DISCOVERY COMPLETE: {recommended_count} sheets recommended for analysis")
        
        return sheet_analyses
    
    def _analyze_sheet_relevance(self, sheet_name: str, df: pd.DataFrame, query: str) -> SheetAnalysis:
        """Analyze how relevant a sheet is for the query"""
        
        score = 0.0
        reasons = []
        
        # Analyze sheet name
        sheet_lower = sheet_name.lower()
        query_lower = query.lower()
        
        # Sheet name relevance
        if any(word in sheet_lower for word in ['financial', 'revenue', 'profit', 'income', 'sales']):
            score += 2.0
            reasons.append("financial sheet name")
        
        if any(word in sheet_lower for word in ['data', 'summary', 'main', 'primary']):
            score += 1.0
            reasons.append("main data sheet")
        
        # Column analysis
        columns_text = ' '.join(str(col).lower() for col in df.columns)
        
        # Financial columns
        financial_keywords = ['revenue', 'sales', 'income', 'profit', 'cost', 'expense', 'amount']
        financial_score = sum(1 for kw in financial_keywords if kw in columns_text)
        score += financial_score * 0.5
        if financial_score > 0:
            reasons.append(f"{financial_score} financial columns")
        
        # Location/Entity columns
        location_keywords = ['entity', 'region', 'province', 'office', 'location', 'area']
        location_score = sum(1 for kw in location_keywords if kw in columns_text)
        score += location_score * 0.5
        if location_score > 0:
            reasons.append(f"{location_score} location columns")
        
        # Time columns
        time_keywords = ['year', 'date', 'period', 'time', 'month']
        time_score = sum(1 for kw in time_keywords if kw in columns_text)
        score += time_score * 0.5
        if time_score > 0:
            reasons.append(f"{time_score} time columns")
        
        # Query-specific keywords
        query_words = [word for word in query_lower.split() if len(word) > 3]
        query_score = sum(1 for word in query_words if word in columns_text)
        score += query_score * 1.0
        if query_score > 0:
            reasons.append(f"matches query terms")
        
        # Data size bonus
        if df.shape[0] > 50:
            score += 1.0
            reasons.append("substantial data")
        
        # Data quality check
        numeric_cols = df.select_dtypes(include=['number']).shape[1]
        if numeric_cols > 2:
            score += 1.0
            reasons.append(f"{numeric_cols} numeric columns")
        
        reason = ", ".join(reasons) if reasons else "no specific indicators"
        
        data_summary = f"{df.shape[0]} rows, {df.shape[1]} cols. Columns: {list(df.columns)[:3]}..."
        
        return SheetAnalysis(
            name=sheet_name,
            relevance_score=score,
            data_summary=data_summary,
            recommended=False,  # Will be set later
            reason=reason
        )

class FocusedAnalysisAgent:
    """
    PHASE 2: Focused analysis agent that only analyzes pre-selected sheets
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze_selected_sheets(self, excel_path: str, sheet_analyses: List[SheetAnalysis], query: str) -> str:
        """Analyze only the recommended sheets with detailed calculation transparency"""
        recommended_sheets = [a for a in sheet_analyses if a.recommended]
        
        if not recommended_sheets:
            return "No relevant sheets found for analysis"
        
        print(f"🎯 PHASE 2: Analyzing {len(recommended_sheets)} recommended sheets")
        
        analysis_report = "🔍 **CALCULATION TRANSPARENCY REPORT**\n"
        analysis_report += "=" * 60 + "\n\n"
        
        best_result = None
        best_score = 0
        best_calculation_details = None
        
        for sheet_analysis in recommended_sheets:
            print(f"   📊 Analyzing {sheet_analysis.name}...")
            analysis_report += f"**Sheet: {sheet_analysis.name}**\n"
            analysis_report += f"Selected because: {sheet_analysis.reason}\n"
            analysis_report += f"Data summary: {sheet_analysis.data_summary}\n\n"
            
            try:
                # Load full sheet data
                df = pd.read_excel(excel_path, sheet_name=sheet_analysis.name)
                
                # Show data structure for transparency
                data_preview = self._generate_data_preview(df)
                analysis_report += f"Data structure preview:\n{data_preview}\n\n"
                
                # Focused analysis on this specific sheet with calculation details
                result, calculation_details = self._analyze_sheet_focused_with_details(df, sheet_analysis, query)
                
                analysis_report += f"Calculation process:\n{calculation_details}\n\n"
                
                if result:
                    # Score the result
                    result_score = self._score_analysis_result(result, query)
                    analysis_report += f"Result: {result}\n"
                    analysis_report += f"Confidence score: {result_score:.1f}\n\n"
                    
                    if result_score > best_score:
                        best_score = result_score
                        best_result = result
                        best_calculation_details = calculation_details
                        print(f"   ✅ Good result (score: {result_score:.1f})")
                    else:
                        print(f"   📋 Result found (score: {result_score:.1f})")
                else:
                    analysis_report += "Result: No relevant data found\n\n"
                    print(f"   ❌ No relevant data found")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                analysis_report += f"{error_msg}\n\n"
                print(f"   ❌ Error analyzing {sheet_analysis.name}: {e}")
                continue
            
            analysis_report += "-" * 40 + "\n\n"
        
        # Final result with complete transparency
        if best_result:
            # Get the best sheet for verification
            best_sheet = None
            for sheet_analysis in recommended_sheets:
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet_analysis.name)
                    result, _ = self._analyze_sheet_focused_with_details(df, sheet_analysis, query)
                    if result and self._score_analysis_result(result, query) == best_score:
                        best_sheet = sheet_analysis
                        best_df = df
                        break
                except:
                    continue
            
            final_report = f"✅ **FINAL ANSWER**: {best_result}\n\n"
            final_report += f"🧮 **HOW THIS WAS CALCULATED**:\n{best_calculation_details}\n\n"
            
            # Add verification report if we found the best sheet
            if best_sheet:
                verification_report = self._create_verification_report(best_df, query, best_result, best_sheet.name)
                final_report += f"🔍 **VERIFICATION REPORT**:\n{verification_report}\n\n"
            
            final_report += f"🎯 **Analysis Method**: Focused two-phase approach (Discovery → Analysis)\n\n"
            final_report += f"📊 **DETAILED CALCULATION REPORT**:\n{analysis_report}"
            return final_report
        else:
            return f"❌ No relevant data found in the recommended sheets\n\n📊 **ANALYSIS REPORT**:\n{analysis_report}"
    
    def _generate_data_preview(self, df: pd.DataFrame) -> str:
        """Generate a preview of the data structure for transparency"""
        preview = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        preview += f"Columns: {list(df.columns)}\n"
        preview += f"Data types: {dict(df.dtypes)}\n"
        
        # Show first few rows
        preview += f"First 3 rows:\n{df.head(3).to_string()}\n"
        
        # Show sample of numeric data if available
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            preview += f"Numeric columns summary:\n{df[numeric_cols].describe().to_string()}"
        
        return preview
    
    def _analyze_sheet_focused_with_details(self, df: pd.DataFrame, sheet_analysis: SheetAnalysis, query: str) -> tuple[Optional[str], str]:
        """Perform focused analysis and return both result and calculation details"""
        
        calculation_log = []
        calculation_log.append(f"Starting analysis on sheet: {sheet_analysis.name}")
        calculation_log.append(f"Query: {query}")
        calculation_log.append(f"Data shape: {df.shape}")
        calculation_log.append(f"Available columns: {list(df.columns)}")
        
        # Capture detailed pandas operations for transparency
        pandas_operations = self._capture_pandas_operations(df, query)
        calculation_log.append("Detailed data analysis:")
        calculation_log.append(pandas_operations)
        
        # Enhanced focused prompt with calculation logging
        focused_prompt = f"""
        You are analyzing the sheet "{sheet_analysis.name}" which was selected because: {sheet_analysis.reason}
        
        This sheet contains: {sheet_analysis.data_summary}
        
        Your task: {query}
        
        CALCULATION TRANSPARENCY INSTRUCTIONS:
        1. SHOW YOUR WORK: Explain every step of your calculation
        2. SHOW PANDAS CODE: Show the exact pandas code you're running
        3. SHOW DATA FILTERS: Show exactly what data you're filtering/selecting
        4. SHOW FORMULAS: Show the exact calculations you're performing
        5. SHOW INTERMEDIATE RESULTS: Show subtotals and intermediate calculations
        6. VERIFY YOUR LOGIC: Double-check your approach and calculations
        7. SHOW ROW COUNTS: Show how many rows match your filters
        
        REQUIRED OUTPUT FORMAT:
        1. Data exploration: What columns and data structure you found
        2. Pandas code: The exact pandas code you're executing
        3. Filtering logic: Exactly what filters you applied and why
        4. Row verification: How many rows match your filters
        5. Calculation steps: Step-by-step calculation process with actual values
        6. Verification: How you verified the result is correct
        7. Final answer: The final numerical result with units/context
        
        Example format:
        "Step 1: Data Exploration
         - Found columns: ['Entity', 'Level2', 'Amount', 'Year']
         - Data shape: (1000, 4)
         
         Step 2: Pandas Code Executed
         - Code: filtered_data = df[(df['Entity'] == 'Ontario') & (df['Year'] == 2023) & (df['Level2'] == 'Revenue')]
         - Row count after filter: 3 rows
         
         Step 3: Filtering Logic
         - Filter 1: Entity == 'Ontario' (looking for Ontario data)
         - Filter 2: Year == 2023 (looking for 2023 data)
         - Filter 3: Level2 == 'Revenue' (looking for revenue entries)
         
         Step 4: Calculation Process
         - Code: result = filtered_data['Amount'].sum()
         - Individual values: [1000000, 2000000, 1200000]
         - Sum calculation: 1000000 + 2000000 + 1200000 = 4200000
         
         Step 5: Verification
         - Verified 3 rows matched the filters
         - Verified all amounts are numeric
         - Cross-checked entity names match exactly
         
         Final Answer: Ontario 2023 revenue = $4,200,000"
        
        CRITICAL: Always show the exact pandas code you execute and the row counts after each filter.
        """
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,  # Enable verbose for calculation transparency
            allow_dangerous_code=True,
            prefix=focused_prompt
        )
        
        try:
            calculation_log.append("Starting pandas agent analysis...")
            result = agent.run(query)
            calculation_log.append(f"Agent completed analysis")
            calculation_log.append(f"Raw result: {result}")
            
            # Parse and enhance the calculation details
            calculation_details = "\n".join(calculation_log)
            calculation_details += f"\n\nAgent Response:\n{result}"
            
            return result, calculation_details
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}"
            calculation_log.append(error_details)
            calculation_details = "\n".join(calculation_log)
            print(f"     ⚠️ Analysis error: {e}")
            return None, calculation_details

    def _analyze_sheet_focused(self, df: pd.DataFrame, sheet_analysis: SheetAnalysis, query: str) -> Optional[str]:
        """Perform focused analysis on a specific sheet"""
        
        # Create focused pandas agent
        focused_prompt = f"""
        You are analyzing the sheet "{sheet_analysis.name}" which was selected because: {sheet_analysis.reason}
        
        This sheet contains: {sheet_analysis.data_summary}
        
        Your task: {query}
        
        FOCUSED ANALYSIS INSTRUCTIONS:
        1. This sheet was pre-selected as relevant - focus on finding the answer here
        2. Examine the data structure quickly but thoroughly
        3. Look for the specific information requested in the query
        4. If you find relevant data, provide a direct answer with numbers
        5. If this sheet doesn't have the exact data, say so clearly
        
        Be direct and specific. Don't explore other possibilities - just analyze this sheet.
        """
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=False,  # Reduce verbosity for speed
            allow_dangerous_code=True,
            prefix=focused_prompt
        )
        
        try:
            result = agent.run(query)
            return result
        except Exception as e:
            print(f"     ⚠️ Analysis error: {e}")
            return None
    
    def _score_analysis_result(self, result: str, query: str) -> float:
        """Score how good an analysis result is"""
        if not result:
            return 0.0
        
        score = 0.0
        result_lower = result.lower()
        
        # Check for specific numbers
        if any(char.isdigit() for char in result):
            score += 2.0
        
        # Check for currency or financial indicators
        if any(symbol in result for symbol in ['$', '€', '£', ',000', 'million', 'billion']):
            score += 2.0
        
        # Check for direct answers
        if any(phrase in result_lower for phrase in ['total', 'revenue', 'profit', 'amount', 'sum']):
            score += 1.0
        
        # Penalty for "not found" messages
        if any(phrase in result_lower for phrase in ['not found', 'no data', 'cannot find', 'does not contain']):
            score -= 3.0
        
        # Check for query keyword matches
        query_words = [w for w in query.lower().split() if len(w) > 3]
        matches = sum(1 for word in query_words if word in result_lower)
        score += matches * 0.5
        
        return max(score, 0.0)
    
    def _capture_pandas_operations(self, df: pd.DataFrame, query: str) -> str:
        """Capture and show actual pandas operations for maximum transparency"""
        operations_log = []
        operations_log.append("=== PANDAS OPERATIONS LOG ===")
        operations_log.append(f"Original DataFrame shape: {df.shape}")
        operations_log.append(f"Columns available: {list(df.columns)}")
        
        # Show key data samples for reference
        operations_log.append("\nSample data for reference:")
        operations_log.append(df.head(5).to_string())
        
        # Analyze what operations would be needed for this query
        query_lower = query.lower()
        operations_log.append(f"\nQuery analysis: '{query}'")
        
        # Try to identify key terms in the query
        key_terms = []
        for col in df.columns:
            if any(term in str(col).lower() for term in query_lower.split()):
                key_terms.append(col)
        
        operations_log.append(f"Potentially relevant columns: {key_terms}")
        
        # Show data types for clarity
        operations_log.append(f"\nColumn data types:")
        for col, dtype in df.dtypes.items():
            operations_log.append(f"  {col}: {dtype}")
        
        # Show unique values for key categorical columns (first 10)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:3]:  # Limit to first 3 to avoid spam
            unique_vals = df[col].unique()[:10]
            operations_log.append(f"\nUnique values in '{col}': {unique_vals}")
            if len(df[col].unique()) > 10:
                operations_log.append(f"  ... and {len(df[col].unique()) - 10} more")
        
        return "\n".join(operations_log)
    
    def _create_verification_report(self, df: pd.DataFrame, query: str, result: str, sheet_name: str) -> str:
        """Create a detailed verification report to validate the calculation"""
        verification = []
        verification.append("🔍 **CALCULATION VERIFICATION REPORT**")
        verification.append("=" * 50)
        verification.append(f"Sheet: {sheet_name}")
        verification.append(f"Query: {query}")
        verification.append(f"Agent Result: {result}")
        verification.append("")
        
        # Try to extract key numbers from the result
        import re
        numbers_in_result = re.findall(r'[\d,]+\.?\d*', result)
        if numbers_in_result:
            verification.append(f"Numbers found in result: {numbers_in_result}")
        
        # Show manual verification steps
        verification.append("**Manual Verification Steps:**")
        verification.append("1. Check the data structure and column names")
        verification.append("2. Identify the exact filters that should be applied")
        verification.append("3. Manually perform the calculation")
        verification.append("4. Compare with agent result")
        verification.append("")
        
        # Show data structure
        verification.append("**Data Structure Check:**")
        verification.append(f"Total rows: {len(df)}")
        verification.append(f"Total columns: {len(df.columns)}")
        verification.append(f"Column names: {list(df.columns)}")
        verification.append("")
        
        # Show sample calculations based on query terms
        query_lower = query.lower()
        verification.append("**Sample Manual Calculations:**")
        
        # Try to identify potential filter columns
        potential_entity_cols = [col for col in df.columns if any(term in col.lower() for term in ['entity', 'company', 'location', 'region', 'province', 'office'])]
        potential_value_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'revenue', 'sales', 'income', 'profit', 'cost'])]
        potential_time_cols = [col for col in df.columns if any(term in col.lower() for term in ['year', 'date', 'period', 'time'])]
        potential_category_cols = [col for col in df.columns if any(term in col.lower() for term in ['level', 'type', 'category', 'class'])]
        
        verification.append(f"Potential entity columns: {potential_entity_cols}")
        verification.append(f"Potential value columns: {potential_value_cols}")
        verification.append(f"Potential time columns: {potential_time_cols}")
        verification.append(f"Potential category columns: {potential_category_cols}")
        verification.append("")
        
        # Show unique values in key columns for debugging
        verification.append("**Key Column Values (for debugging filters):**")
        for col in potential_entity_cols + potential_category_cols:
            if col in df.columns:
                unique_vals = df[col].unique()[:5]  # Show first 5
                verification.append(f"'{col}' values: {unique_vals}")
                if len(df[col].unique()) > 5:
                    verification.append(f"  ... and {len(df[col].unique()) - 5} more")
        
        verification.append("")
        verification.append("**How to Manually Verify:**")
        verification.append("1. Load the Excel sheet manually")
        verification.append("2. Apply the same filters mentioned in the agent's response")
        verification.append("3. Sum/calculate the values in the target column")
        verification.append("4. Compare with the agent's result")
        
        return "\n".join(verification)

class FocusedAgenticWorkflow:
    """
    Two-phase focused workflow: Discovery → Analysis
    """
    
    def __init__(self):
        print("🚀 Initializing Focused Two-Phase Agent...")
        self.discovery_agent = DataDiscoveryAgent()
        self.analysis_agent = FocusedAnalysisAgent()
        
        # Discover tools
        self.tools = self._discover_tools()
        print("✅ Focused Agent ready!")
        print("🎯 Phase 1: Discovery | Phase 2: Focused Analysis")
    
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
        """Process query with focused two-phase approach"""
        print(f"🎯 Focused Agent processing: {query}")
        
        try:
            if 'excel' in self.tools:
                excel_path = self.tools['excel']
                
                # PHASE 1: Discovery
                sheet_analyses = self.discovery_agent.discover_relevant_sheets(excel_path, query)
                
                if not any(a.recommended for a in sheet_analyses):
                    return "🔍 Discovery Phase: No relevant sheets found for this query"
                
                # PHASE 2: Focused Analysis
                result = self.analysis_agent.analyze_selected_sheets(excel_path, sheet_analyses, query)
                
                return result
            else:
                return "No Excel data source available"
                
        except Exception as e:
            return f"🎯 Focused Agent error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status"""
        return """🎯 **Focused Two-Phase Agent with Calculation Transparency**

**System Type**: Focused Discovery → Analysis with Full Transparency
**Agent Model**: GPT-4o-mini (Discovery) + GPT-4o (Analysis)
**Strategy**: Pre-select relevant data, then focused analysis with step-by-step calculation details

**Two-Phase Process**:
🔍 **Phase 1 - Discovery Agent**:
   • Quickly scans ALL sheets/data sources
   • Scores relevance for the specific query
   • Selects top 2 most relevant sheets
   • Uses fast GPT-4o-mini for speed

🎯 **Phase 2 - Analysis Agent with Transparency**:
   • Only analyzes pre-selected relevant sheets
   • Shows complete data structure and column analysis
   • Displays exact pandas code executed
   • Shows step-by-step calculation process
   • Provides detailed verification reports
   • Uses GPT-4o for detailed analysis

**Calculation Transparency Features**:
🔍 Data structure preview with column types
🧮 Step-by-step calculation process
📊 Exact pandas code execution details
✅ Row count verification after each filter
🔍 Manual verification instructions
📋 Detailed calculation reports

**Performance Benefits**:
✅ No time wasted on irrelevant sheets
✅ Fast discovery phase for sheet selection
✅ Focused analysis phase for accuracy
✅ Complete calculation transparency
✅ Easy debugging of incorrect results
✅ Concrete answers with verification

**Available Data Sources**: """ + f"{len(self.tools)} sources discovered"
    
    def get_available_commands(self) -> str:
        """Get available commands"""
        return """🎯 **Focused Two-Phase Agent Capabilities**

**How It Works**:
1. 🔍 **Discovery Phase**: Quickly scans all sheets/data
2. 🎯 **Selection Phase**: Picks top 2 most relevant
3. 📊 **Analysis Phase**: Focused analysis on selected data
4. ✅ **Result Phase**: Direct answer with concrete numbers

**Query Examples**:
- "What was revenue in 2023 for Ontario?"
  → Phase 1: Finds financial sheets
  → Phase 2: Analyzes only those sheets
  → Result: "Ontario 2023 revenue: $4,200,000"

**Performance Features**:
🚀 **Fast Discovery**: Quick sheet scanning
🎯 **Smart Selection**: Relevance-based filtering
📊 **Focused Analysis**: No wandering between sheets
✅ **Concrete Results**: Direct answers, not loops

**Focused Process**:
1. **Scans** all available data sources rapidly
2. **Scores** each source for query relevance
3. **Selects** top candidates automatically
4. **Analyzes** only the relevant data
5. **Provides** direct answer with supporting data

**No More**:
❌ Bouncing between multiple sheets
❌ Analyzing irrelevant data
❌ Getting stuck in analysis loops
❌ Vague or incomplete answers

**Instead**:
✅ Targeted data selection
✅ Focused analysis approach
✅ Clear decision making
✅ Concrete results

Just ask your question - the agent will find the right data and give you a direct answer!
"""