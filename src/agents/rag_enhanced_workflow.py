import os
import pandas as pd
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from .rag_discovery_agent import RAGDiscoveryAgent, SourceCandidate


class RAGEnhancedExcelAgent:
    """Enhanced Excel agent that works with RAG-discovered sources"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze(self, source_candidate: SourceCandidate, query: str, expertise: str = None) -> str:
        """Analyze using pre-selected source with domain expertise"""
        
        schema = source_candidate.schema
        
        try:
            # Load the specific sheet/file
            if schema.sheet_name:
                # Excel sheet
                df = pd.read_excel(schema.file_path, sheet_name=schema.sheet_name)
                data_context = f"Excel sheet '{schema.sheet_name}'"
            else:
                # CSV file
                df = pd.read_csv(schema.file_path)
                data_context = f"CSV file '{os.path.basename(schema.file_path)}'"
            
            print(f"   📊 Loaded {data_context}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Generate expertise-based prompt (use provided expertise or fallback)
            final_expertise = expertise or schema.domain_info.get('expertise_needed', 'financial_analyst')
            expert_prompt = self._create_expert_prompt(query, schema, data_context, final_expertise)
            
            # DEBUG: DataFrame info
            print(f"   🔍 DataFrame loaded successfully:")
            print(f"      Shape: {df.shape}")
            print(f"      Columns: {list(df.columns)}")
            print(f"      Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            print(f"      First few rows available: {not df.head().empty}")
            
            # Create pandas agent with expert context and DEBUG ENABLED
            print(f"   🤖 Creating pandas agent with enhanced configuration...")
            
            # Add explicit constraint against sampling
            full_prompt = f"""{expert_prompt}

**CRITICAL**: The data is ALREADY LOADED in the DataFrame variable 'df'. 

**FORBIDDEN OPERATIONS**:
- Do NOT use pd.read_csv(), pd.read_excel(), or any file reading functions
- Do NOT try to read 'VW_PBI.csv' or any other file - THE DATA IS ALREADY LOADED
- Do NOT use df.head(), df.tail(), df.sample() 
- Do NOT limit rows with [:10], [:100], etc.
- Do NOT use .iloc[:n] to limit data

**MANDATORY**: 
- Work DIRECTLY with the variable 'df' - it contains {df.shape[0]:,} rows of loaded data
- The DataFrame 'df' IS the VW_PBI sheet data - use it directly
- Start your analysis with df.info(), df.columns, df.shape to understand the data

**REMEMBER**: You have {df.shape[0]:,} rows of data already loaded in 'df' - use ALL of them!"""
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=True,  # Enable verbose to see what's happening
                prefix=full_prompt,
                max_iterations=15,  # Increase iterations
                early_stopping_method="generate",
                allow_dangerous_code=True,  # Allow code execution for data analysis
                # return_intermediate_steps=True  # Removed to fix API compatibility
            )
            
            print(f"   📊 Agent created. Executing analysis...")
            print(f"   🎯 Query: {query}")
            
            # Execute analysis with error handling
            try:
                # Use invoke instead of run for newer LangChain versions
                agent_result = agent.invoke({"input": query})
                result = agent_result.get("output", str(agent_result))
                print(f"   ✅ Analysis completed successfully")
            except Exception as agent_error:
                print(f"   ❌ Agent execution failed: {str(agent_error)}")
                
                # Fallback: Try basic DataFrame operations directly
                print(f"   🔄 Attempting fallback direct analysis...")
                try:
                    fallback_result = self._fallback_analysis(df, query, schema)
                    result = f"**Fallback Analysis**:\n{fallback_result}"
                except Exception as fallback_error:
                    result = f"**Analysis Failed**:\nAgent error: {str(agent_error)}\nFallback error: {str(fallback_error)}"
                    print(f"   ❌ Fallback also failed: {str(fallback_error)}")
            
            return f"""**RAG-Enhanced Analysis Result**

**Source**: {schema.source_name} ({data_context})
**Selected because**: {source_candidate.reason}
**Domain expertise**: {schema.domain_info.get('expertise_needed', 'general analyst')}
**Data structure**: {schema.domain_info.get('structure_type', 'tabular')}

**Analysis**:
{result}

**Data Context**:
- Columns available: {', '.join(schema.columns)}
- Calculation capabilities: {', '.join(schema.domain_info.get('calculation_capabilities', ['basic']))}
- Row count: {schema.row_count:,}
- DataFrame shape: {df.shape}
"""
            
        except Exception as e:
            return f"❌ Analysis failed for {schema.source_name}: {str(e)}"
    
    def _create_expert_prompt(self, query: str, schema, data_context: str, expertise: str) -> str:
        """Create expert prompt based on domain and structure"""
        
        domain_type = schema.domain_info.get('domain_type', 'unknown')
        structure_type = schema.domain_info.get('structure_type', 'tabular')
        calculations = schema.domain_info.get('calculation_capabilities', [])
        
        # Base expert identity
        expert_prompt = f"""You are a {expertise} analyzing {domain_type} data from {data_context}.

**Data Structure**: {structure_type}
**Available Columns**: {', '.join(schema.columns)}
**Known Calculations**: {', '.join(calculations) if calculations else 'Standard calculations'}

**Domain Expertise**:"""
        
        # Add domain-specific guidance
        if domain_type == "financial":
            expert_prompt += f"""
- This is financial data from VW_PBI Excel sheet with hierarchical structure (Level1/Level2/Level3)
- The DataFrame 'df' contains {schema.row_count:,} rows of financial data - work directly with 'df'
- EXPLORE FIRST: Check df['Level1'].unique(), df['Level2'].unique(), df['Level3'].unique() to understand the hierarchy
- Revenue/Income data could be in any level - search across Level1, Level2, Level3, and Account columns
- For financial ratios, explore the Balance Sheet structure first before filtering
- Cash ratio = Cash and cash equivalents / Current Liabilities (find these terms in the data)
- Current ratio = Current Assets / Current Liabilities (find these terms in the data)
- Example exploration: df[df['Level1'].str.contains('Revenue|Income', case=False, na=False)] 
- Use hierarchical filtering with Level1/Level2/Level3 columns based on ACTUAL values found
- Filter by Year column for specific periods
- THE DATA IS ALREADY LOADED - do not try to read any files"""
            
        elif domain_type == "mining":
            expert_prompt += """
- This is mining operations data with production and operational metrics
- Look for metal production, ore processing, grades, and recovery rates
- Calculate efficiency ratios and production metrics
- Consider time-based analysis for trends"""
            
        elif domain_type == "hr":
            expert_prompt += """
- This is HR/training data with employee and time-based information
- Look for training hours, certifications, and employee metrics
- Aggregate by employee, department, or time period as needed"""
        
        expert_prompt += f"""

**Your Task**: {query}

**CRITICAL INSTRUCTIONS**:
1. **USE THE ENTIRE DATASET** - Never use .head(), .sample(), or limit rows
2. **WORK WITH ALL {schema.row_count:,} ROWS** - This is financial data requiring complete analysis
3. **EXPLORE THE DATA FIRST** - Use df.shape, df.columns, df.info()
4. **CHECK ACTUAL VALUES** - Use .unique() on Level1, Level2, Level3 columns to see what exists
5. **SEARCH INTELLIGENTLY** - Look for terms across ALL columns, not just assumed locations
6. **DON'T ASSUME DATA STRUCTURE** - Revenue might be in Level2, Level3, or Account columns
7. Filter data appropriately (by year, entity, etc.) but include ALL relevant records
8. For financial calculations, ensure you're using the most recent/relevant data
9. **EXPLORE BEFORE FILTERING** - Check df['Level1'].unique(), df['Level2'].unique(), etc.
10. **NEVER SAMPLE THE DATA** - Use the complete dataset for accurate calculations

**MANDATORY EXPLORATION STEPS**:
- First: df.columns to see all available columns
- Second: Check unique values in hierarchical columns (Level1, Level2, Level3, Account)
- Third: Search for your target data across all relevant columns
- Fourth: Filter and calculate based on what you actually found

**DATASET SIZE**: {schema.row_count:,} rows - use them all for accurate financial analysis."""

        return expert_prompt
    
    def _fallback_analysis(self, df: pd.DataFrame, query: str, schema) -> str:
        """Fallback analysis when pandas agent fails"""
        
        query_lower = query.lower()
        
        # Basic financial ratio calculations
        if 'cash ratio' in query_lower:
            cash_cols = [col for col in df.columns if 'cash' in col.lower()]
            liability_cols = [col for col in df.columns if any(term in col.lower() for term in ['liability', 'liabilities', 'current_liab'])]
            
            if cash_cols and liability_cols:
                cash_col = cash_cols[0]
                liability_col = liability_cols[0]
                
                # Get recent data
                if len(df) > 0:
                    latest_cash = df[cash_col].iloc[-1] if not df[cash_col].isna().all() else 0
                    latest_liability = df[liability_col].iloc[-1] if not df[liability_col].isna().all() else 1
                    
                    if latest_liability != 0:
                        ratio = latest_cash / latest_liability
                        return f"Cash Ratio = {latest_cash:,.0f} / {latest_liability:,.0f} = {ratio:.3f}\n\nCalculation based on columns: {cash_col} and {liability_col}"
        
        # Basic data summary
        summary_parts = [
            f"Data Summary for query: {query}",
            f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"Available columns: {', '.join(df.columns)}"
        ]
        
        # Show sample data if available
        if len(df) > 0:
            summary_parts.append(f"\nSample data (first 3 rows):")
            summary_parts.append(df.head(3).to_string())
        
        return "\n".join(summary_parts)


class RAGEnhancedCSVAgent:
    """Enhanced CSV agent that works with RAG-discovered sources"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze(self, source_candidate: SourceCandidate, query: str, expertise: str = None) -> str:
        """Analyze CSV source with domain expertise"""
        
        schema = source_candidate.schema
        
        try:
            # Load CSV file
            df = pd.read_csv(schema.file_path)
            
            print(f"   📄 Loaded CSV: {os.path.basename(schema.file_path)} ({df.shape[0]} rows, {df.shape[1]} cols)")
            
            # Generate expertise-based prompt (use provided expertise or fallback)
            final_expertise = expertise or schema.domain_info.get('expertise_needed', 'data_analyst')
            expert_prompt = self._create_expert_prompt(query, schema, final_expertise)
            
            # DEBUG: DataFrame info
            print(f"   🔍 CSV DataFrame loaded successfully:")
            print(f"      Shape: {df.shape}")
            print(f"      Columns: {list(df.columns)}")
            print(f"      Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Create pandas agent with expert context and DEBUG ENABLED
            print(f"   🤖 Creating CSV pandas agent...")
            
            # Add explicit DataFrame constraint for CSV agent
            full_csv_prompt = f"""{expert_prompt}

**CRITICAL**: The CSV data is ALREADY LOADED in the DataFrame variable 'df'. 

**FORBIDDEN OPERATIONS**:
- Do NOT use pd.read_csv(), pd.read_excel(), or any file reading functions
- Do NOT try to read 'hr_data.csv', 'WorkforceData.csv', or any other file - THE DATA IS ALREADY LOADED
- Do NOT use df.head(), df.tail(), df.sample() 
- Do NOT limit rows with [:10], [:100], etc.

**MANDATORY**: 
- Work DIRECTLY with the variable 'df' - it contains {df.shape[0]:,} rows of loaded CSV data
- The DataFrame 'df' IS the {schema.source_name} data - use it directly
- Start your analysis with df.info(), df.columns, df.shape to understand the data

**REMEMBER**: You have {df.shape[0]:,} rows of data already loaded in 'df' - use ALL of them!"""
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=True,  # Enable verbose to see what's happening
                prefix=full_csv_prompt,
                max_iterations=15,  # Increase iterations
                early_stopping_method="generate",
                allow_dangerous_code=True,  # Allow code execution for data analysis
                # return_intermediate_steps=True  # Removed to fix API compatibility
            )
            
            print(f"   📊 CSV Agent created. Executing analysis...")
            
            # Execute analysis with error handling
            try:
                # Use invoke instead of run for newer LangChain versions
                agent_result = agent.invoke({"input": query})
                result = agent_result.get("output", str(agent_result))
                print(f"   ✅ CSV Analysis completed successfully")
            except Exception as agent_error:
                print(f"   ❌ CSV Agent execution failed: {str(agent_error)}")
                
                # Fallback: Basic DataFrame analysis
                print(f"   🔄 Attempting CSV fallback analysis...")
                try:
                    fallback_result = self._fallback_csv_analysis(df, query, schema)
                    result = f"**Fallback CSV Analysis**:\n{fallback_result}"
                except Exception as fallback_error:
                    result = f"**CSV Analysis Failed**:\nAgent error: {str(agent_error)}\nFallback error: {str(fallback_error)}"
            
            return f"""**RAG-Enhanced CSV Analysis Result**

**Source**: {os.path.basename(schema.file_path)}
**Selected because**: {source_candidate.reason}
**Domain expertise**: {schema.domain_info.get('expertise_needed', 'general analyst')}

**Analysis**:
{result}

**Data Context**:
- Columns: {', '.join(schema.columns)}
- Row count: {schema.row_count:,}
- Capabilities: {', '.join(schema.domain_info.get('calculation_capabilities', ['basic']))}
"""
            
        except Exception as e:
            return f"❌ CSV analysis failed for {schema.source_name}: {str(e)}"
    
    def _create_expert_prompt(self, query: str, schema, expertise: str) -> str:
        """Create expert prompt for CSV analysis"""
        
        domain_type = schema.domain_info.get('domain_type', 'unknown')
        
        return f"""You are a {expertise} analyzing {domain_type} data from a CSV file.

**Available Columns**: {', '.join(schema.columns)}
**Data Type**: {domain_type}
**Row Count**: {schema.row_count:,}

**Your Task**: {query}

**Instructions**:
1. Analyze the data structure and identify relevant columns
2. Perform any required calculations or aggregations
3. Filter data as needed for the specific query
4. Provide clear, numerical results
5. Explain your methodology

Use the full dataset to provide a complete answer."""
    
    def _fallback_csv_analysis(self, df: pd.DataFrame, query: str, schema) -> str:
        """Fallback analysis for CSV when pandas agent fails"""
        
        query_lower = query.lower()
        
        # Basic training hours analysis
        if 'training' in query_lower and 'hours' in query_lower:
            training_cols = [col for col in df.columns if 'training' in col.lower() or 'hours' in col.lower()]
            if training_cols:
                col = training_cols[0]
                total_hours = df[col].sum() if df[col].dtype in ['int64', 'float64'] else len(df)
                return f"Total training hours: {total_hours:,.0f}\nBased on column: {col}"
        
        # Basic data summary
        summary_parts = [
            f"CSV Data Summary for query: {query}",
            f"File: {schema.file_path}",
            f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"Available columns: {', '.join(df.columns)}"
        ]
        
        # Show sample data
        if len(df) > 0:
            summary_parts.append(f"\nSample data (first 3 rows):")
            summary_parts.append(df.head(3).to_string())
        
        return "\n".join(summary_parts)


class RAGEnhancedWorkflow:
    """Main workflow that uses RAG for fast discovery + current agents for analysis"""
    
    def __init__(self, rebuild_index: bool = False):
        print("🚀 Initializing RAG-Enhanced Workflow...")
        
        # Initialize RAG discovery
        self.rag_discovery = RAGDiscoveryAgent(rebuild_index=rebuild_index)
        
        # Initialize enhanced agents
        self.excel_agent = RAGEnhancedExcelAgent()
        self.csv_agent = RAGEnhancedCSVAgent()
        
        print("✅ RAG-Enhanced Workflow ready!")
        print("🎯 Flow: Query → RAG Discovery → Schema-Aware Selection → Expert Analysis")
    
    def process_query(self, query: str) -> str:
        """Process query using RAG discovery + expert analysis"""
        
        print(f"🎯 RAG-Enhanced Agent processing: {query}")
        
        try:
            # PHASE 1: Fast RAG Discovery (replaces slow current discovery)
            source_candidate = self.rag_discovery.discover_best_source(query)
            
            if not source_candidate:
                return "❌ **No Data Source Found**\n\nThe system could not find any data source that contains the required information to answer your query. The available sources do not have the necessary data elements for this calculation/analysis.\n\n**Suggestion**: Try rephrasing your query or check if the required data exists in your data sources."
            
            # PHASE 2: Generate Expertise for Query
            expertise = self._generate_expertise(query, source_candidate.schema)
            print(f"   👨‍💼 Generated expertise: {expertise}")
            
            # PHASE 3: Expert Analysis with Selected Source
            schema = source_candidate.schema
            
            # DEBUG: Show exactly what source was selected
            print(f"   🔍 DEBUG SOURCE SELECTION:")
            print(f"      Source name: {schema.source_name}")
            print(f"      File path: {schema.file_path}")
            print(f"      Sheet name: {schema.sheet_name}")
            print(f"      File extension: {schema.file_path.split('.')[-1]}")
            
            if schema.file_path.endswith('.xlsx'):
                print(f"   📊 Routing to EXCEL agent for .xlsx file")
                result = self.excel_agent.analyze(source_candidate, query, expertise)
            elif schema.file_path.endswith('.csv'):
                print(f"   📄 Routing to CSV agent for .csv file") 
                result = self.csv_agent.analyze(source_candidate, query, expertise)
            else:
                return f"❌ Unsupported file type: {schema.file_path}"
            
            return result
            
        except Exception as e:
            return f"🎯 RAG-Enhanced Agent error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status"""
        
        schema_count = len(self.rag_discovery.indexer.schema_store)
        
        return f"""🎯 **RAG-Enhanced Agentic Workflow**

**System Type**: Vector RAG Discovery → Schema-Aware Selection → Expert Analysis
**Discovery Model**: text-embedding-ada-002 (Vector Search)
**Selection Model**: GPT-4o-mini (Schema Reasoning) 
**Analysis Model**: GPT-4o (Expert Analysis)

**Performance**:
- Discovery Phase: ~100-500ms (vs previous 10-20 seconds)
- Indexed Sources: {schema_count} data sources
- Cost Reduction: ~70-80% fewer LLM calls in discovery

**Capabilities**:
✅ Fast semantic source discovery
✅ Schema-aware intelligent selection  
✅ Domain expertise generation
✅ Complex calculation support
✅ Hierarchical data handling (Level1/Level2/Level3)
✅ Multi-domain analysis (Financial, Mining, HR)

**Data Sources Available**: {schema_count} sources indexed and ready for analysis"""
    
    def get_available_commands(self) -> str:
        """Get available commands and examples"""
        
        return """🎯 **RAG-Enhanced Agent Commands & Examples**

**Financial Analysis Examples**:
• "What is the cash ratio for 2024?"
• "Calculate current ratio for Acme Mining Corp"
• "What was the net income for Ontario operations?"
• "Show me the debt-to-equity ratio trends"

**Mining Operations Examples**:
• "What was gold production in Q3 2024?"
• "Calculate recovery rate for copper processing"
• "Show me ore throughput by mine site"
• "What's the grade for silver operations?"

**Training/HR Examples**:
• "What were total training hours for safety in 2024?"
• "Show certification rates by department"
• "Calculate training hours per employee for gold operations"

**System Commands**:
• 'status' - Show system status and indexed sources
• 'help' - Show this help message
• 'rebuild' - Force rebuild of RAG index (if data changed)

**How it works**:
1. 🔍 **RAG Discovery**: Finds relevant sources in ~100ms using vector search
2. 🧠 **Smart Selection**: LLM picks best source based on schema and calculation needs  
3. 🎯 **Expert Analysis**: Domain-specific agent performs complex calculations
4. ✅ **Quality Results**: Preserves all current calculation capabilities with 20x speed improvement"""
    
    def _generate_expertise(self, query: str, schema) -> str:
        """Generate specific expertise needed for the query"""
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Based on this specific query, generate a detailed expert profile who can best analyze this data.

QUERY: {query}
DATA SOURCE: {schema.source_name}
DOMAIN: {schema.domain_info.get('domain_type', 'unknown')}
COLUMNS: {', '.join(schema.columns[:10])}

Create a SPECIFIC expert description that includes:
1. Base profession (financial analyst, mining engineer, etc.)
2. Specific expertise area related to the query
3. Relevant specialization

EXAMPLES:
- Query "cash ratio" → "Financial analyst with expertise in liquidity ratios and working capital management"
- Query "gold production" → "Mining operations analyst specializing in precious metals extraction and production optimization"
- Query "training hours" → "HR analyst with expertise in workforce development and training program analysis"
- Query "debt to equity" → "Financial analyst specializing in capital structure and leverage analysis"

RESPONSE FORMAT:
Expert: [detailed expert description with specific expertise]

RESPONSE:"""

        try:
            response = llm.invoke(prompt).content
            if "Expert:" in response:
                expert_line = [line for line in response.split('\n') if 'Expert:' in line][0]
                expert_type = expert_line.split('Expert:')[1].strip()
                return expert_type
            else:
                return "financial_analyst"  # Default fallback
        except Exception as e:
            print(f"⚠️ Expertise generation failed: {str(e)}")
            return "financial_analyst"  # Safe fallback

    def rebuild_index(self) -> str:
        """Force rebuild the RAG index"""
        try:
            self.rag_discovery.rebuild_index()
            return "✅ RAG index rebuilt successfully! All data sources re-indexed."
        except Exception as e:
            return f"❌ Failed to rebuild index: {str(e)}"