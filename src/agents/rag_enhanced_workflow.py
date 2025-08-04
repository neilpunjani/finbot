import os
import pandas as pd
import time
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from .rag_discovery_agent import RAGDiscoveryAgent, SourceCandidate
from .pdf_document_agent import PDFDocumentAgent


class DataFrameCacheManager:
    """Shared DataFrame cache manager with optimized timestamp checking"""
    
    def __init__(self):
        self._dataframe_cache = {}  # Store DataFrames in memory
        self._file_timestamps = {}  # Track when files were last modified
        self._last_timestamp_check = {}  # Track when we last checked file timestamps (performance optimization)
    
    def get_cached_dataframe(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Get DataFrame from cache or load fresh if file was modified (optimized with smart timestamp checking)"""
        
        cache_key = f"{file_path}:{sheet_name or 'default'}"
        
        try:
            # PERFORMANCE OPTIMIZATION: Only check file modification time every 5 seconds to avoid excessive I/O
            current_time = time.time()
            last_check = self._last_timestamp_check.get(cache_key, 0)
            
            # If we haven't checked the timestamp recently, check it now
            if current_time - last_check > 5.0:  # Check at most every 5 seconds
                current_mtime = os.path.getmtime(file_path)
                self._file_timestamps[cache_key] = current_mtime
                self._last_timestamp_check[cache_key] = current_time
            
            cached_mtime = self._file_timestamps.get(cache_key, 0)
            
            # Load fresh data if file is new or was modified
            if cache_key not in self._dataframe_cache or cached_mtime > self._file_timestamps.get(f"{cache_key}_loaded", 0):
                print(f"   📁 Loading fresh data from {os.path.basename(file_path)} (file modified or first load)")
                
                if sheet_name:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    data_context = f"Excel sheet '{sheet_name}'"
                else:
                    df = pd.read_csv(file_path)
                    data_context = f"CSV file '{os.path.basename(file_path)}'"
                
                # Cache the DataFrame and mark when it was loaded
                self._dataframe_cache[cache_key] = df
                self._file_timestamps[f"{cache_key}_loaded"] = cached_mtime
                
                print(f"   💾 Cached {data_context}: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                print(f"   ⚡ Using cached data for {os.path.basename(file_path)} (no changes detected)")
            
            return self._dataframe_cache[cache_key]
            
        except Exception as e:
            print(f"   ❌ Cache error for {file_path}: {str(e)}")
            # Fallback to direct loading
            if sheet_name:
                return pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                return pd.read_csv(file_path)
    
    def clear_cache(self):
        """Manually clear the DataFrame cache"""
        self._dataframe_cache.clear()
        self._file_timestamps.clear()
        self._last_timestamp_check.clear()
        print("   🔄 DataFrame cache cleared")


# Shared cache manager instance
_shared_cache_manager = DataFrameCacheManager()


class RAGEnhancedExcelAgent:
    """Enhanced Excel agent that works with RAG-discovered sources"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using faster model for pandas analysis
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Use shared cache manager for performance
        self.cache_manager = _shared_cache_manager
    
    def _get_cached_dataframe(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Get DataFrame from cache or load fresh if file was modified (uses shared optimized cache)"""
        return self.cache_manager.get_cached_dataframe(file_path, sheet_name)
    
    def clear_cache(self):
        """Manually clear the DataFrame cache"""
        self.cache_manager.clear_cache()
    
    def analyze(self, source_candidate: SourceCandidate, query: str, expertise: str = None) -> str:
        """Analyze using pre-selected source with domain expertise"""
        
        schema = source_candidate.schema
        
        try:
            # Use cached DataFrame (loads fresh only if file was modified)
            df = self._get_cached_dataframe(schema.file_path, schema.sheet_name)
            
            if schema.sheet_name:
                data_context = f"Excel sheet '{schema.sheet_name}'"
            else:
                data_context = f"CSV file '{os.path.basename(schema.file_path)}'"
            
            print(f"   📊 Using {data_context}: {df.shape[0]} rows, {df.shape[1]} columns")
            
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
            
            # Create pandas agent with expert context
            full_prompt = expert_prompt
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=True,  # Enable verbose to see what's happening
                prefix=full_prompt,
                max_iterations=7,  # Balanced for performance and completion
                # early_stopping_method="generate",  # Removed - unsupported in current LangChain version
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

**FINANCIAL ANALYSIS PRINCIPLES**:
1. **ALWAYS EXPLORE FIRST**: Check df['Level1'].unique(), df['Level2'].unique(), df['Level3'].unique()
2. **COMPREHENSIVE CALCULATIONS**: 
   - Total Debt = Current Liabilities + Non-Current Liabilities + Long-term Debt
   - Total Assets = Current Assets + Non-Current Assets (or Fixed Assets)
   - Total Equity = Share Capital + Retained Earnings + Other Equity components
   - Revenue = All revenue streams (Operating Revenue + Other Revenue)
   - Total Expenses = All expense categories combined

**STANDARD FINANCIAL RATIOS** (find ALL required components):
- Debt-to-Equity = Total Debt / Total Equity
- Current Ratio = Current Assets / Current Liabilities  
- Cash Ratio = Cash & Cash Equivalents / Current Liabilities
- ROA = Net Income / Total Assets
- ROE = Net Income / Total Equity
- Gross Profit Margin = (Revenue - COGS) / Revenue

**SEARCH STRATEGY**:
- Search across Level1, Level2, Level3, Account columns for financial terms
- Use partial matching: .str.contains('Debt|Liability|Payable', case=False, na=False)
- Include ALL relevant line items (don't miss Non-Current, Long-term, etc.)
- Verify completeness by checking what you found vs what should exist

**ADAPTIVE CALCULATION STRATEGY**:
1. **FIRST**: Try to find the exact value directly (e.g., search for "Net Income", "Total Debt", etc.)
2. **IF NOT FOUND**: Calculate from components using the hierarchical structure
3. **EXAMPLE**: 
   - Net Income: Look for direct "Net Income" line item
   - If not found: Revenue - All Expenses - Interest - Tax
   - Total Debt: Look for "Total Debt" line item  
   - If not found: Current Liabilities + Non-Current Liabilities + Long-term Debt
4. **ALWAYS**: Show both what you found directly AND what you calculated
5. **VERIFY**: Cross-check calculated values against any direct line items if both exist

**MANDATORY**: For any ratio, identify and include ALL components, not just partial amounts.

**NET INCOME CALCULATION**: When asked for Net Income, CALCULATE it from components (Revenue - Expenses - Interest - Tax) rather than just looking for a "Net Income" line item. Show both the calculation breakdown and final result.

THE DATA IS ALREADY LOADED - do not try to read any files"""
            
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
            model="gpt-4o-mini",  # Using faster model for pandas analysis  
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Use shared cache manager for performance
        self.cache_manager = _shared_cache_manager
    
    def _get_cached_dataframe(self, file_path: str) -> pd.DataFrame:
        """Get CSV DataFrame from cache or load fresh if file was modified (uses shared optimized cache)"""
        return self.cache_manager.get_cached_dataframe(file_path, None)  # None for CSV (no sheet name)
    
    def clear_cache(self):
        """Manually clear the CSV DataFrame cache"""
        self.cache_manager.clear_cache()
    
    def analyze(self, source_candidate: SourceCandidate, query: str, expertise: str = None) -> str:
        """Analyze CSV source with domain expertise"""
        
        schema = source_candidate.schema
        
        try:
            # Use cached CSV DataFrame
            df = self._get_cached_dataframe(schema.file_path)
            
            print(f"   📄 Using CSV: {os.path.basename(schema.file_path)} ({df.shape[0]} rows, {df.shape[1]} cols)")
            
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
            
            # Create CSV pandas agent with expert context
            full_csv_prompt = expert_prompt
            
            agent = create_pandas_dataframe_agent(
                self.llm,
                df,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=True,  # Enable verbose to see what's happening
                prefix=full_csv_prompt,
                max_iterations=7,  # Balanced for performance and completion
                # early_stopping_method="generate",  # Removed - unsupported in current LangChain version
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
        
        # Track loading state
        self.is_loading = True
        self.loading_status = "Initializing system..."
        
        # Initialize RAG discovery
        self.loading_status = "Building vector index..."
        self.rag_discovery = RAGDiscoveryAgent(rebuild_index=rebuild_index)
        
        # Initialize enhanced agents
        self.loading_status = "Preparing AI agents..."
        self.excel_agent = RAGEnhancedExcelAgent()
        self.csv_agent = RAGEnhancedCSVAgent()
        self.pdf_agent = PDFDocumentAgent(vector_store=self.rag_discovery.indexer.vector_store)
        
        # Preload common data sources at startup
        self.loading_status = "Preloading data sources..."
        print("📊 Preloading data sources...")
        self._preload_data_sources()
        
        # Mark as ready
        self.is_loading = False
        self.loading_status = "Ready"
        
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
            
            # PHASE 2: Generate Domain-Based Expertise (restore original functionality)
            expertise = source_candidate.schema.domain_info.get('expertise_needed', 'financial_analyst')
            print(f"   👨‍💼 Using expertise: {expertise}")
            
            # PHASE 3: Expert Analysis with Selected Source
            schema = source_candidate.schema
            
            # DEBUG: Show exactly what source was selected
            print(f"   🔍 DEBUG SOURCE SELECTION:")
            print(f"      Source name: {schema.source_name}")
            print(f"      File path: {schema.file_path}")
            print(f"      Sheet name: {schema.sheet_name}")
            print(f"      File extension: {schema.file_path.split('.')[-1]}")
            
            # Dynamic file-type based routing
            file_extension = schema.file_path.split('.')[-1].lower()
            
            if file_extension == 'xlsx':
                print(f"   📊 Routing to EXCEL agent for .xlsx file")
                result = self.excel_agent.analyze(source_candidate, query, expertise)
            elif file_extension == 'csv':
                print(f"   📄 Routing to CSV agent for .csv file") 
                result = self.csv_agent.analyze(source_candidate, query, expertise)
            elif file_extension == 'pdf':
                print(f"   📋 Routing to PDF agent for .pdf file")
                result = self.pdf_agent.query(query)
            else:
                return f"❌ Unsupported file type: {file_extension}. Supported types: xlsx, csv, pdf"
            
            return result
            
        except Exception as e:
            return f"🎯 RAG-Enhanced Agent error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status"""
        
        schema_count = len(self.rag_discovery.indexer.schema_store)
        
        return f"""🎯 **RAG-Enhanced Agentic Workflow**

**System Type**: Vector RAG Discovery → File-Type Routing → Expert Analysis
**Discovery Model**: text-embedding-ada-002 (Vector Search)
**Selection Model**: GPT-4o-mini (Schema Reasoning) 
**Analysis Model**: GPT-4o-mini (Fast Analysis)

**Performance**:
- Discovery Phase: ~100-500ms (vs previous 10-20 seconds)
- Indexed Sources: {schema_count} data sources
- Cost Reduction: ~70-80% fewer LLM calls in discovery

**Capabilities**:
✅ Fast semantic source discovery
✅ Dynamic file-type based routing (Excel/CSV/PDF)
✅ Domain expertise generation
✅ Complex calculation support
✅ Hierarchical data handling (Level1/Level2/Level3)
✅ Multi-domain analysis (Financial, Mining, HR, Policy)
✅ PDF document analysis and policy querying

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

**Policy/Document Examples**:
• "What are the safety protocols for mining operations?"
• "Show me environmental compliance requirements"
• "What are best practices for equipment usage?"
• "Find procedures for hazardous material handling"

**System Commands**:
• 'status' - Show system status and indexed sources
• 'help' - Show this help message
• 'rebuild' - Force rebuild of RAG index (if data changed)

**How it works**:
1. 🔍 **RAG Discovery**: Finds relevant sources in ~100ms using vector search
2. 🎯 **File-Type Routing**: Routes to appropriate agent (Excel/CSV/PDF) based on source type
3. 🧠 **Expert Analysis**: Domain-specific agent performs calculations or document analysis
4. ✅ **Scalable Results**: Handles any data source type without code changes"""
    
    def _get_simple_expertise(self, domain_type: str) -> str:
        """Get simple domain-based expertise without LLM call"""
        
        expertise_map = {
            "financial": "financial analyst",
            "mining": "mining operations analyst", 
            "hr": "HR analyst",
            "operational": "operations analyst",
            "unknown": "data analyst"
        }
        
        return expertise_map.get(domain_type, "data analyst")
    
    def _preload_data_sources(self):
        """Preload common data sources at startup for faster query responses"""
        
        preload_count = 0
        
        try:
            # Preload VW_PBI Excel sheet (most common for financial queries)
            excel_path = os.getenv("EXCEL_FILE_PATH")
            if excel_path and os.path.exists(excel_path):
                self.loading_status = "Loading Excel data (VW_PBI)..."
                print(f"   📈 Loading VW_PBI Excel data...")
                self.excel_agent._get_cached_dataframe(excel_path, "VW_PBI")
                preload_count += 1
            
            # Preload common CSV files
            csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
            if os.path.exists(csv_dir):
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
                for i, csv_file in enumerate(csv_files[:3], 1):  # Preload top 3 CSV files
                    self.loading_status = f"Loading CSV files ({i}/3)..."
                    csv_path = os.path.join(csv_dir, csv_file)
                    print(f"   📄 Loading {csv_file}...")
                    self.csv_agent._get_cached_dataframe(csv_path)
                    preload_count += 1
            
            self.loading_status = "Finalizing setup..."
            print(f"   ✅ Preloaded {preload_count} data sources into cache")
            
        except Exception as e:
            print(f"   ⚠️ Preloading warning: {str(e)} (will load on-demand)")
            self.loading_status = f"Warning: {str(e)}"

    def rebuild_index(self) -> str:
        """Force rebuild the RAG index"""
        try:
            self.rag_discovery.rebuild_index()
            # Also clear DataFrame caches when rebuilding
            self.clear_all_caches()
            return "✅ RAG index rebuilt successfully! All data sources re-indexed and caches cleared."
        except Exception as e:
            return f"❌ Failed to rebuild index: {str(e)}"
    
    def clear_all_caches(self) -> str:
        """Clear all DataFrame caches for fresh data loading"""
        try:
            self.excel_agent.clear_cache()
            self.csv_agent.clear_cache()
            return "✅ All DataFrame caches cleared! Next queries will load fresh data."
        except Exception as e:
            return f"❌ Failed to clear caches: {str(e)}"