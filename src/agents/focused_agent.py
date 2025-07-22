import os
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

@dataclass
class DataSourceAnalysis:
    source_type: str  # 'excel_sheet', 'csv', 'sql', 'email'
    name: str
    relevance_score: float
    data_summary: str
    recommended: bool
    reason: str
    file_path: Optional[str] = None
    sheet_name: Optional[str] = None
    # Enhanced domain analysis fields
    domain_type: str = "Unknown"  # 'Financial', 'Mining Production', 'Mining Operations', 'Mixed'
    data_structure: str = ""  # Description of data hierarchy/structure
    key_fields: List[str] = field(default_factory=list)  # Most important columns for analysis
    expertise_needed: str = ""  # What kind of expert the agent should become
    field_variations: Dict[str, List[str]] = field(default_factory=dict)  # Common abbreviations/variations for fields

class DataDiscoveryAgent:
    """
    PHASE 1: Multi-source data discovery agent that explores all available data sources
    and intelligently selects which ones to analyze based on query relevance
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use faster model for discovery
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.cached_dataframes = {}
    
    def _get_mining_financial_expert_system_prompt(self) -> str:
        """Get the persistent mining and financial domain expert system prompt"""
        return """You are an expert mining operations and financial data analyst. This system contains data from mining companies with production, operational, and financial information. Analyze datasets to determine how well they can answer user queries.

**MINING & FINANCIAL DATA DOMAIN KNOWLEDGE**:

**MINING PRODUCTION DATA** (Highest value for production queries):
- MetalProduced, GoldProduced, CopperProduced = Final metal output (tonnes/ounces)
- OreProcessed, OreTreated = Raw material input (tonnes)
- Grade = Metal concentration in ore (g/t, %)
- RecoveryRate = Extraction efficiency (%)
- MillThroughput = Processing capacity utilization

**MINING OPERATIONAL DATA** (For efficiency/logistics queries):
- TonnesMoved, HaulageVolume = Material transportation
- EquipmentUtilization = Machine efficiency (%)
- DowntimeHours = Equipment maintenance time
- FuelConsumption = Operating costs indicator
- BlastingOperations = Mine development activities

**FINANCIAL DATA** (For revenue/cost/profitability queries):
- Amount, Value, Revenue = Monetary values ($)
- Cost, Expense, OPEX, CAPEX = Expenditure categories
- Commodity pricing = Metal sale prices
- Level1/Level2/Level3 = Financial statement hierarchies (BS/PnL/CF)
- EBITDA, NetIncome = Profitability metrics

**QUERY TYPE ANALYSIS**:
1. **PRODUCTION QUERIES** ("production", "output", "generated", "extracted"):
   - Priority: MetalProduced > OreProcessed > TonnesMoved
   - Look for: Actual output quantities, not logistics

2. **FINANCIAL QUERIES** - USE COMPREHENSIVE DATA STRUCTURE INTELLIGENCE:
   
   **COMPLETE DATA STRUCTURE ANALYSIS**:
   - Search the ENTIRE data structure for the requested metric:
     1. Column names (e.g., "Revenue", "NetIncome", "Amount")
     2. Hierarchical line items (Level1/Level2/Level3 containing "Net Income", "Revenue", etc.)
     3. Actual data values and line item descriptions
     4. Sample data content that shows the metric directly
   
   **INTELLIGENT DECISION LOGIC**:
   - **DIRECT PRESENCE** (Score 9-10): Metric found anywhere in the data structure
     - Column name matches query terms
     - OR Level1/Level2/Level3 contains the exact metric as line items  
     - OR sample data shows the metric is directly available
   
   - **CALCULATION REQUIRED** (Score based on components available):
     - Metric NOT found directly anywhere in data structure
     - Check if sheet has BUILDING BLOCKS for calculation
     - Detailed P&L sheets (Level1/Level2/Level3) with revenue/expense components = High score
     - Summary sheets with only totals = Lower score (limited calculation ability)
   
   **PRESERVATION OF EXISTING LOGIC**:
   - Continue prioritizing sheets with strong domain matches (production, operational, financial)
   - Maintain hierarchical data structure scoring (Level1/Level2/Level3 indicates detailed data)
   - Keep sample data analysis for understanding data quality and relevance

3. **OPERATIONAL EFFICIENCY** ("efficiency", "utilization", "performance"):
   - Priority: RecoveryRate, EquipmentUtilization > raw output
   - Look for: Percentage/ratio metrics

4. **LOGISTICS QUERIES** ("moved", "transported", "hauled"):
   - Priority: TonnesMoved, HaulageVolume > MetalProduced
   - Look for: Movement/transportation data

**CRITICAL DISTINCTIONS IN MINING**:
- MetalProduced (final product) ≠ TonnesMoved (logistics)
- OreProcessed (input) ≠ MetalProduced (output)  
- Amount (financial $) ≠ Quantity (physical units)
- Grade (quality) ≠ RecoveryRate (efficiency)
- Production (output focus) ≠ Operations (process focus)

**COMMODITY TYPES**: Gold, Copper, Iron Ore, Coal, Silver, Zinc, Lead, Nickel

**SCORING CRITERIA** (0-10 scale):
- **9-10**: Perfect domain match - exact mining/financial metric requested
- **7-8**: Strong match - correct data type with minor conceptual gap
- **5-6**: Moderate match - related mining/financial data but different focus
- **3-4**: Weak match - same industry but wrong metric type
- **0-2**: Poor match - unrelated to query intent

**REQUIRED OUTPUT FORMAT** (Follow EXACTLY):
Score: X
Data Purpose: [Mining production/operations/financial classification]
Best Column: [Most relevant column name]
Match Quality: [Perfect/Strong/Moderate/Weak/Poor match explanation]
Reasoning: [Domain-specific analysis of why this data matches the query]
Domain Type: [Financial|Mining Production|Mining Operations|Mixed]
Data Structure: [Description of hierarchies/organization]
Key Fields: [Comma-separated list of most important columns]
Expertise Needed: [What kind of expert should analyze this data]
Field Variations: [Common abbreviations for key fields, format: field=abbrev1,abbrev2|field2=abbrev3]

**COMPREHENSIVE DATA STRUCTURE ANALYSIS EXAMPLES**:

**Example 1 - Direct Match in Column:**
Query: "What was total revenue in 2024?"
Sheet has: ["Date", "Entity", "Revenue", "Amount"]
Analysis: "Revenue" column exists directly → Score 9-10

**Example 2 - Direct Match in Line Items:**
Query: "What was net income in 2024?"  
Sheet has: ["Date", "Entity", "Level1", "Level2", "Amount"]
Sample data shows: Level1="Net Income", Level2="After Tax"
Analysis: "Net Income" exists as Level1 line item → Score 9-10

**Example 3 - No Direct Match, Calculation Possible:**
Query: "What was net income in 2024?"
Sheet has: ["Date", "Entity", "Level1", "Level2", "Amount"]
Sample data shows: Level1="Revenue", Level1="Operating Expenses", etc.
Analysis: No direct "Net Income" but has P&L building blocks → Score 7-8

**Example 4 - No Direct Match, Limited Calculation:**
Query: "What was net income in 2024?"
Sheet has: ["Date", "Entity", "Amount"] (summary totals only)
Analysis: No "Net Income" anywhere, no detailed components → Score 3-4

**COMPREHENSIVE INTELLIGENCE PRINCIPLE**:
- Search ALL levels: column names, hierarchical data, sample content, line item descriptions
- Preserve existing domain expertise while adding calculation intelligence
- Score based on data structure depth and component availability"""
    
    def discover_all_data_sources(self, tools: Dict, query: str) -> List[DataSourceAnalysis]:
        """Explore ALL available data sources and intelligently select which ones to analyze"""
        print(f"🔍 PHASE 1: Multi-source discovery for query: {query}")
        
        all_analyses = []
        self.cached_dataframes = {}
        
        # Explore Excel sources
        if 'excel' in tools:
            excel_analyses = self._discover_excel_sources(tools['excel'], query)
            all_analyses.extend(excel_analyses)
        
        # Explore CSV sources
        if 'csv' in tools:
            csv_analyses = self._discover_csv_sources(tools['csv'], query)
            all_analyses.extend(csv_analyses)
        
        # Explore SQL sources (placeholder for future implementation)
        if 'sql' in tools:
            print("   🗄️ SQL database detected (not yet implemented)")
        
        # Explore Email sources (placeholder for future implementation)
        if 'email' in tools:
            print("   📧 Email source detected (not yet implemented)")
        
        # Sort all sources by relevance
        all_analyses.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Intelligent multi-source selection
        recommended_sources = self._intelligent_source_selection(all_analyses, query)
        
        # Mark recommended sources
        for analysis in all_analyses:
            analysis.recommended = analysis in recommended_sources
        
        print(f"🎯 MULTI-SOURCE DISCOVERY COMPLETE: {len(recommended_sources)} sources recommended")
        
        # Show intelligent selection reasoning
        if recommended_sources:
            print("   📋 Intelligently selected sources:")
            for source in recommended_sources:
                print(f"     • {source.source_type}: {source.name} (score: {source.relevance_score:.1f}) - {source.reason}")
        else:
            print("   ❌ No sources met the relevance threshold")
        
        return all_analyses
    
    def _discover_excel_sources(self, excel_path: str, query: str) -> List[DataSourceAnalysis]:
        """Discover and analyze Excel sheets"""
        print("   📊 Exploring Excel data sources...")
        excel_analyses = []
        
        xls = pd.ExcelFile(excel_path)
        
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                print(f"     📄 Loaded sheet {sheet_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Cache the dataframe
                cache_key = f"excel_{sheet_name}"
                self.cached_dataframes[cache_key] = df
                
                if df.empty or df.shape[1] < 2:
                    print(f"     ❌ Skipping {sheet_name}: Empty or insufficient columns")
                    continue
                
                # Analyze relevance
                analysis = self._analyze_source_relevance(
                    source_type="excel_sheet",
                    name=sheet_name,
                    df=df,
                    query=query,
                    file_path=excel_path,
                    sheet_name=sheet_name
                )
                excel_analyses.append(analysis)
                
                print(f"     📊 {sheet_name}: Score {analysis.relevance_score:.1f}")
                
            except Exception as e:
                print(f"     ❌ Error reading {sheet_name}: {e}")
                continue
        
        return excel_analyses
    
    def _discover_csv_sources(self, csv_info: Dict, query: str) -> List[DataSourceAnalysis]:
        """Discover and analyze CSV data sources"""
        print("   📄 Exploring CSV data sources...")
        csv_analyses = []
        
        csv_dir = csv_info['directory']
        csv_files = csv_info['files']
        
        for csv_file in csv_files:
            try:
                csv_path = os.path.join(csv_dir, csv_file)
                df = pd.read_csv(csv_path)
                print(f"     📄 Loaded CSV {csv_file}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Cache the dataframe
                cache_key = f"csv_{csv_file}"
                self.cached_dataframes[cache_key] = df
                
                if df.empty or df.shape[1] < 2:
                    print(f"     ❌ Skipping {csv_file}: Empty or insufficient columns")
                    continue
                
                # Analyze relevance
                analysis = self._analyze_source_relevance(
                    source_type="csv",
                    name=csv_file,
                    df=df,
                    query=query,
                    file_path=csv_path
                )
                csv_analyses.append(analysis)
                
                print(f"     📊 {csv_file}: Score {analysis.relevance_score:.1f}")
                
            except Exception as e:
                print(f"     ❌ Error reading {csv_file}: {e}")
                continue
        
        return csv_analyses
    
    def _intelligent_source_selection(self, all_analyses: List[DataSourceAnalysis], query: str) -> List[DataSourceAnalysis]:
        """Intelligently select data sources using LLM reasoning combined with scoring"""
        
        if not all_analyses:
            return []
        
        # Analyze query complexity to determine selection strategy
        query_complexity = self._analyze_query_complexity(query)
        print(f"   🧠 Query complexity: {query_complexity['level']} - {query_complexity['reason']}")
        
        # Base selection logic with adaptive thresholds
        min_score = 2.0  # Base threshold
        max_sources = 3  # Default max sources
        
        # Adjust based on complexity
        if query_complexity['level'] == 'simple':
            min_score = 3.0  # Higher threshold for simple queries
            max_sources = 2
        elif query_complexity['level'] == 'complex':
            min_score = 1.5  # Lower threshold for complex queries
            max_sources = 4
        
        # Filter by minimum score
        candidates = [a for a in all_analyses if a.relevance_score >= min_score]
        
        if not candidates:
            # Lower the threshold if no candidates
            candidates = all_analyses[:2] if len(all_analyses) >= 2 else all_analyses
            print(f"   📊 No sources met threshold {min_score}, selecting top {len(candidates)} by score")
        
        # Use LLM for intelligent final selection
        selected_sources = self._llm_source_selection(candidates[:max_sources], query, query_complexity)
        
        return selected_sources
    
    def _llm_source_selection(self, candidates: List[DataSourceAnalysis], query: str, complexity: Dict) -> List[DataSourceAnalysis]:
        """Use LLM to make final intelligent source selection"""
        
        if len(candidates) <= 2:
            return candidates  # If 2 or fewer, use all
        
        # Prepare source information for LLM
        source_info = []
        for i, source in enumerate(candidates):
            source_info.append(f"""
Source {i+1}: {source.source_type.upper()} - {source.name}
- Relevance Score: {source.relevance_score:.1f}
- Data Summary: {source.data_summary}
- Selection Reason: {source.reason}
""")
        
        selection_prompt = f"""
You are an intelligent data source selection agent. Given a user query and available data sources, 
select the OPTIMAL combination of sources that will best answer the query.

User Query: "{query}"
Query Complexity: {complexity['level']} - {complexity['reason']}

Available Sources:
{''.join(source_info)}

SELECTION CRITERIA:
1. Relevance: Choose sources most likely to contain the answer
2. Complementarity: If multiple sources could provide different perspectives, select both
3. Efficiency: Don't select redundant sources
4. Coverage: For complex queries, ensure you have sufficient data coverage

RULES:
- For simple queries (single metric/entity): Select 1-2 most relevant sources
- For complex queries (comparisons/analysis): Select 2-3 sources for comprehensive coverage
- Prioritize sources with highest relevance scores regardless of type
- Consider complementarity between different source types for comprehensive analysis
- Never select more than 3 sources unless absolutely necessary

RESPONSE FORMAT:
Selected sources: [comma-separated list of source numbers, e.g., "1, 3"]
Reasoning: [brief explanation of why these sources were selected]
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=selection_prompt)])
            
            # Parse response
            lines = response.content.strip().split('\n')
            selected_numbers = []
            reasoning = ""
            
            for line in lines:
                if line.startswith("Selected sources:"):
                    numbers_str = line.split(":", 1)[1].strip()
                    selected_numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Select based on LLM response
            selected_sources = []
            for num in selected_numbers:
                if 1 <= num <= len(candidates):
                    selected_sources.append(candidates[num-1])
            
            if selected_sources:
                print(f"   🤖 LLM selected {len(selected_sources)} sources: {reasoning}")
                return selected_sources
            
        except Exception as e:
            print(f"   ⚠️ LLM selection failed: {e}, falling back to score-based selection")
        
        # Fallback: select top 2 by score
        return candidates[:2]
    
    def _analyze_source_relevance(self, source_type: str, name: str, df: pd.DataFrame, query: str, 
                                file_path: str, sheet_name: str = None) -> DataSourceAnalysis:
        """Analyze how relevant a data source is for the given query"""
        
        # Get data summary
        data_summary = self._get_data_preview(df)
        
        # Score relevance using enhanced keyword matching + LLM reasoning
        relevance_score, reason, domain_analysis = self._score_source_relevance(source_type, name, df, query, data_summary)
        
        return DataSourceAnalysis(
            source_type=source_type,
            name=name,
            relevance_score=relevance_score,
            data_summary=data_summary,
            recommended=False,  # Will be set later
            reason=reason,
            file_path=file_path,
            sheet_name=sheet_name,
            domain_type=domain_analysis.get('domain_type', 'Unknown'),
            data_structure=domain_analysis.get('data_structure', ''),
            key_fields=domain_analysis.get('key_fields', []),
            expertise_needed=domain_analysis.get('expertise_needed', ''),
            field_variations=domain_analysis.get('field_variations', {})
        )
    
    def _score_source_relevance(self, source_type: str, name: str, df: pd.DataFrame, query: str, data_summary: str) -> Tuple[float, str, Dict]:
        """Score how relevant this source is for the query using intelligent LLM analysis"""
        
        # Use LLM to intelligently analyze data structure and relevance
        return self._llm_score_data_relevance(source_type, name, df, query)
    
    def _llm_score_data_relevance(self, source_type: str, name: str, df: pd.DataFrame, query: str) -> Tuple[float, str, Dict]:
        """Use LLM to intelligently score data source relevance based on complete data structure analysis"""
        
        # Prepare comprehensive data structure information
        try:
            # Get sample data for LLM analysis
            sample_rows = df.head(3).to_string(index=False)
            
            # Get column statistics
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Get unique values for categorical columns (limited to prevent token overflow)
            categorical_samples = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                unique_vals = df[col].unique()
                if len(unique_vals) <= 20:  # Only show if reasonable number of unique values
                    categorical_samples[col] = list(unique_vals)
                else:
                    categorical_samples[col] = list(unique_vals[:10]) + [f"... and {len(unique_vals)-10} more"]
            
            # Use system message for persistent domain knowledge + dynamic query analysis
            system_prompt = self._get_mining_financial_expert_system_prompt()
            
            # Dynamic analysis prompt (efficient - only query-specific data)
            analysis_prompt = f"""
**USER QUERY**: "{query}"

**DATASET**: {name}
**COLUMNS**: {list(df.columns)}
**SAMPLE DATA**:
{sample_rows}

**ADDITIONAL CONTEXT**:

**MINING PRODUCTION DATA** (Highest value for production queries):
- MetalProduced, GoldProduced, CopperProduced = Final metal output (tonnes/ounces)
- OreProcessed, OreTreated = Raw material input (tonnes)
- Grade = Metal concentration in ore (g/t, %)
- RecoveryRate = Extraction efficiency (%)
- MillThroughput = Processing capacity utilization

**MINING OPERATIONAL DATA** (For efficiency/logistics queries):
- TonnesMoved, HaulageVolume = Material transportation
- EquipmentUtilization = Machine efficiency (%)
- DowntimeHours = Equipment maintenance time
- FuelConsumption = Operating costs indicator
- BlastingOperations = Mine development activities

**FINANCIAL DATA** (For revenue/cost/profitability queries):
- Amount, Value, Revenue = Monetary values ($)
- Cost, Expense, OPEX, CAPEX = Expenditure categories
- Commodity pricing = Metal sale prices
- Level1/Level2/Level3 = Financial statement hierarchies (BS/PnL/CF)
- EBITDA, NetIncome = Profitability metrics

**QUERY TYPE ANALYSIS**:
1. **PRODUCTION QUERIES** ("production", "output", "generated", "extracted"):
   - Priority: MetalProduced > OreProcessed > TonnesMoved
   - Look for: Actual output quantities, not logistics

2. **FINANCIAL QUERIES** ("revenue", "sales", "profit", "cost", "financial"):
   - Priority: Amount/Revenue > operational metrics
   - Look for: Dollar values, financial statements

3. **OPERATIONAL EFFICIENCY** ("efficiency", "utilization", "performance"):
   - Priority: RecoveryRate, EquipmentUtilization > raw output
   - Look for: Percentage/ratio metrics

4. **LOGISTICS QUERIES** ("moved", "transported", "hauled"):
   - Priority: TonnesMoved, HaulageVolume > MetalProduced
   - Look for: Movement/transportation data

**CRITICAL DISTINCTIONS IN MINING**:
- MetalProduced (final product) ≠ TonnesMoved (logistics)
- OreProcessed (input) ≠ MetalProduced (output)  
- Amount (financial $) ≠ Quantity (physical units)
- Grade (quality) ≠ RecoveryRate (efficiency)
- Production (output focus) ≠ Operations (process focus)

**COMMODITY TYPES**: Gold, Copper, Iron Ore, Coal, Silver, Zinc, Lead, Nickel

**SCORING CRITERIA** (0-10 scale):
- **9-10**: Perfect domain match - exact mining/financial metric requested
- **7-8**: Strong match - correct data type with minor conceptual gap
- **5-6**: Moderate match - related mining/financial data but different focus
- **3-4**: Weak match - same industry but wrong metric type
- **0-2**: Poor match - unrelated to query intent

**REQUIRED OUTPUT FORMAT** (Follow EXACTLY):
Score: X
Data Purpose: [Mining production/operations/financial classification]
Best Column: [Most relevant column name]
Match Quality: [Perfect/Strong/Moderate/Weak/Poor match explanation]
Reasoning: [Domain-specific analysis of why this data matches the query]
Domain Type: [Financial|Mining Production|Mining Operations|Mixed]
Data Structure: [Description of hierarchies/organization]
Key Fields: [Comma-separated list of most important columns]
Expertise Needed: [What kind of expert should analyze this data]
Field Variations: [Common abbreviations for key fields, format: field=abbrev1,abbrev2|field2=abbrev3]

**DOMAIN-SPECIFIC EXAMPLES**:
Query: "gold production 2024" → MetalProduced/GoldProduced columns = Score 9-10
Query: "tonnes moved" → TonnesMoved columns = Score 9-10  
Query: "revenue breakdown" → Amount with financial Level1/2/3 = Score 9-10
Query: "equipment efficiency" → EquipmentUtilization/RecoveryRate = Score 8-9

NOW ANALYZE THIS MINING/FINANCIAL DATASET:
"""

            # Get LLM analysis using system + user messages (industry best practice)
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=analysis_prompt)
            ])
            
            # Parse the enhanced response
            content = response.content.strip()
            print(f"     🔍 DEBUG - LLM Raw Response for {name}:")
            print(f"         {content}")
            lines = content.split('\n')
            
            score = 0.0
            data_purpose = "Unknown"
            best_column = "None"
            match_quality = "Unknown"
            reasoning = "LLM analysis failed to parse"
            domain_type = "Unknown"
            data_structure = ""
            key_fields = []
            expertise_needed = ""
            field_variations = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Score:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        # Extract just the number
                        import re
                        numbers = re.findall(r'\d+\.?\d*', score_text)
                        if numbers:
                            score = float(numbers[0])
                    except Exception as e:
                        print(f"         ⚠️ Score parsing error: {e}")
                        score = 0.0
                elif line.startswith('Data Purpose:'):
                    data_purpose = line.split(':', 1)[1].strip()
                elif line.startswith('Best Column:'):
                    best_column = line.split(':', 1)[1].strip()
                elif line.startswith('Match Quality:'):
                    match_quality = line.split(':', 1)[1].strip()
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif line.startswith('Domain Type:'):
                    domain_type = line.split(':', 1)[1].strip()
                elif line.startswith('Data Structure:'):
                    data_structure = line.split(':', 1)[1].strip()
                elif line.startswith('Key Fields:'):
                    key_fields_text = line.split(':', 1)[1].strip()
                    key_fields = [field.strip() for field in key_fields_text.split(',') if field.strip()]
                elif line.startswith('Expertise Needed:'):
                    expertise_needed = line.split(':', 1)[1].strip()
                elif line.startswith('Field Variations:'):
                    variations_text = line.split(':', 1)[1].strip()
                    try:
                        # Parse format: field=abbrev1,abbrev2|field2=abbrev3
                        for field_group in variations_text.split('|'):
                            if '=' in field_group:
                                field_name, abbrevs = field_group.split('=', 1)
                                field_variations[field_name.strip()] = [a.strip() for a in abbrevs.split(',')]
                    except Exception as e:
                        print(f"         ⚠️ Field variations parsing error: {e}")
                        field_variations = {}
            
            # If parsing failed, try to extract any useful info from the content
            if score == 0.0 and data_purpose == "Unknown":
                print(f"         ⚠️ Parsing failed, trying fallback extraction...")
                # Try to find score in any format
                import re
                score_matches = re.findall(r'(?:score|rating).*?(\d+(?:\.\d+)?)', content.lower())
                if score_matches:
                    score = float(score_matches[0])
                    print(f"         ✓ Extracted score: {score}")
                
                # Try to find purpose/meaning
                if 'production' in content.lower():
                    data_purpose = "Production/output data"
                elif 'financial' in content.lower() or 'money' in content.lower():
                    data_purpose = "Financial data"
                elif 'operational' in content.lower():
                    data_purpose = "Operational data"
            
            # Ensure score is within valid range
            score = max(0.0, min(10.0, score))
            
            # Create comprehensive reasoning
            full_reasoning = f"Purpose: {data_purpose[:50]}... | Quality: {match_quality[:30]}... | Column: {best_column}"
            
            # Create domain analysis dictionary
            domain_analysis = {
                'domain_type': domain_type,
                'data_structure': data_structure,
                'key_fields': key_fields,
                'expertise_needed': expertise_needed,
                'field_variations': field_variations,
                'data_purpose': data_purpose,
                'best_column': best_column,
                'match_quality': match_quality
            }
            
            print(f"     🤖 LLM Pattern Analysis - {name}:")
            print(f"         Score: {score:.1f}")
            print(f"         Domain Type: {domain_type}")
            print(f"         Purpose: {data_purpose}")
            print(f"         Best Column: {best_column}")
            print(f"         Expertise: {expertise_needed}")
            print(f"         Key Fields: {key_fields}")
            
            return score, full_reasoning, domain_analysis
            
        except Exception as e:
            print(f"     ⚠️ LLM scoring failed for {name}: {e}")
            # Fallback to basic scoring
            return self._fallback_basic_scoring(source_type, name, df, query)
    
    def _fallback_basic_scoring(self, source_type: str, name: str, df: pd.DataFrame, query: str) -> Tuple[float, str, Dict]:
        """Simple fallback scoring when LLM analysis fails"""
        
        score = 0.0
        query_lower = query.lower()
        name_lower = name.lower()
        columns_lower = [col.lower() for col in df.columns]
        
        # Basic relevance indicators
        if any(word in name_lower for word in query_lower.split() if len(word) > 3):
            score += 3.0
        
        if any(word in ' '.join(columns_lower) for word in query_lower.split() if len(word) > 3):
            score += 2.0
        
        # Data quality
        if df.shape[0] > 100:
            score += 0.5
        
        # Create minimal domain analysis for fallback
        fallback_domain_analysis = {
            'domain_type': "Unknown",
            'data_structure': f"Basic tabular data with {df.shape[1]} columns",
            'key_fields': list(df.columns)[:5],
            'expertise_needed': "General data analysis",
            'field_variations': {},
            'data_purpose': "General data source",
            'best_column': df.columns[0] if len(df.columns) > 0 else "None",
            'match_quality': "Basic fallback match"
        }
        
        return score, f"Fallback scoring based on basic name/column matching", fallback_domain_analysis
    
    
    def _get_data_preview(self, df: pd.DataFrame) -> str:
        """Generate a concise data preview for source relevance analysis"""
        preview = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns. "
        preview += f"Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}. "
        
        # Show sample values for key columns
        sample_data = []
        for col in df.columns[:3]:  # First 3 columns
            unique_vals = df[col].unique()
            if len(unique_vals) <= 5:
                sample_data.append(f"{col}: {list(unique_vals)}")
            else:
                sample_data.append(f"{col}: {list(unique_vals[:3])}...")
        
        if sample_data:
            preview += f"Sample values: {'; '.join(sample_data)}"
        
        return preview
    
    def _adaptive_sheet_selection(self, sheet_analyses: List[DataSourceAnalysis], query: str) -> List[DataSourceAnalysis]:
        """Adaptively select sheets based on relevance scores and query complexity"""
        
        if not sheet_analyses:
            return []
        
        # Analyze query complexity to determine selection strategy
        query_complexity = self._analyze_query_complexity(query)
        print(f"   🧠 Query complexity: {query_complexity['level']} - {query_complexity['reason']}")
        
        # Base relevance threshold
        base_threshold = 3.0
        
        # Adjust threshold based on query complexity
        if query_complexity['level'] == 'simple':
            # Simple queries: be more selective, higher threshold
            threshold = base_threshold + 1.0
            max_sheets = 2
        elif query_complexity['level'] == 'moderate':
            # Moderate queries: standard threshold
            threshold = base_threshold
            max_sheets = 4
        else:  # complex
            # Complex queries: lower threshold, allow more sheets
            threshold = base_threshold - 1.0
            max_sheets = 6
        
        # Get sheets above threshold
        candidate_sheets = [s for s in sheet_analyses if s.relevance_score >= threshold]
        
        # If no sheets meet threshold, take the top performers
        if not candidate_sheets:
            candidate_sheets = sheet_analyses[:2]  # Fallback to top 2
            print(f"   📊 No sheets met threshold {threshold:.1f}, using top {len(candidate_sheets)} sheets")
        
        # Limit by max_sheets but use adaptive logic
        if len(candidate_sheets) > max_sheets:
            # Look for natural score breaks
            score_gaps = []
            for i in range(len(candidate_sheets) - 1):
                gap = candidate_sheets[i].relevance_score - candidate_sheets[i + 1].relevance_score
                score_gaps.append((i + 1, gap))  # (cut_point, gap_size)
            
            # Find the largest gap in the top max_sheets
            score_gaps = score_gaps[:max_sheets - 1]
            if score_gaps:
                best_cut = max(score_gaps, key=lambda x: x[1])
                if best_cut[1] > 1.0:  # Significant gap
                    candidate_sheets = candidate_sheets[:best_cut[0]]
                    print(f"   ✂️ Found natural break at position {best_cut[0]} (gap: {best_cut[1]:.1f})")
                else:
                    candidate_sheets = candidate_sheets[:max_sheets]
            else:
                candidate_sheets = candidate_sheets[:max_sheets]
        
        print(f"   🎯 Selection strategy: threshold={threshold:.1f}, max_sheets={max_sheets}")
        print(f"   ✅ Selected {len(candidate_sheets)} sheets for analysis")
        
        return candidate_sheets
    
    def _analyze_query_complexity(self, query: str) -> dict:
        """Analyze query complexity to determine selection strategy"""
        
        query_lower = query.lower()
        complexity_indicators = {
            'simple': 0,
            'moderate': 0,
            'complex': 0
        }
        
        # Simple query indicators
        simple_patterns = [
            'what is', 'what was', 'how much', 'total', 'sum',
            'revenue for', 'profit for', 'sales for'
        ]
        for pattern in simple_patterns:
            if pattern in query_lower:
                complexity_indicators['simple'] += 1
        
        # Moderate query indicators
        moderate_patterns = [
            'compare', 'between', 'difference', 'vs', 'versus',
            'trend', 'change', 'growth', 'increase', 'decrease'
        ]
        for pattern in moderate_patterns:
            if pattern in query_lower:
                complexity_indicators['moderate'] += 1
        
        # Complex query indicators
        complex_patterns = [
            'analyze', 'breakdown', 'detailed', 'comprehensive',
            'all regions', 'all entities', 'across', 'correlation',
            'pattern', 'insight', 'summary', 'overview'
        ]
        for pattern in complex_patterns:
            if pattern in query_lower:
                complexity_indicators['complex'] += 1
        
        # Additional complexity factors
        word_count = len(query.split())
        if word_count > 15:
            complexity_indicators['complex'] += 1
        elif word_count < 8:
            complexity_indicators['simple'] += 1
        
        # Has multiple conditions (AND/OR)
        if ' and ' in query_lower or ' or ' in query_lower:
            complexity_indicators['moderate'] += 1
        
        # Determine complexity level
        max_score = max(complexity_indicators.values())
        if max_score == 0:
            level = 'moderate'  # Default
            reason = 'standard query pattern'
        else:
            level = max(complexity_indicators, key=complexity_indicators.get)
            reason = f"{max_score} complexity indicators detected"
        
        return {'level': level, 'reason': reason, 'scores': complexity_indicators}
    
    def _analyze_sheet_relevance(self, sheet_name: str, df: pd.DataFrame, query: str) -> DataSourceAnalysis:
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
        
        # CRITICAL FIX: Penalize sheets with blank/empty data
        data_quality_penalty = self._calculate_data_quality_penalty(df, query)
        score -= data_quality_penalty
        if data_quality_penalty > 0:
            reasons.append(f"data quality penalty: -{data_quality_penalty:.1f}")
        
        # Final score cannot be negative
        score = max(score, 0.0)
        
        reason = ", ".join(reasons) if reasons else "no specific indicators"
        
        data_summary = f"{df.shape[0]} rows, {df.shape[1]} cols. Columns: {list(df.columns)[:3]}..."
        
        return DataSourceAnalysis(
            source_type="excel_sheet",
            name=sheet_name,
            relevance_score=score,
            data_summary=data_summary,
            recommended=False,  # Will be set later
            reason=reason,
            file_path=None,  # Will be set by caller
            sheet_name=sheet_name
        )
    
    def _calculate_data_quality_penalty(self, df: pd.DataFrame, query: str) -> float:
        """Calculate penalty for poor data quality (blank/empty data)"""
        penalty = 0.0
        
        # Check for excessive blank/null values
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            null_percentage = df.isnull().sum().sum() / total_cells * 100
            if null_percentage > 80:
                penalty += 3.0  # Severe penalty for mostly empty sheets
            elif null_percentage > 50:
                penalty += 2.0  # Moderate penalty
            elif null_percentage > 30:
                penalty += 1.0  # Light penalty
        
        # Check for columns that appear relevant but are mostly empty
        query_lower = query.lower()
        
        # Identify potentially relevant columns based on query
        relevant_column_keywords = []
        if 'ontario' in query_lower:
            relevant_column_keywords.extend(['entity', 'region', 'location', 'office', 'company'])
        if any(year in query_lower for year in ['2023', '2024', '2022']):
            relevant_column_keywords.extend(['year', 'date', 'period'])
        if any(metric in query_lower for metric in ['revenue', 'sales', 'profit', 'income']):
            relevant_column_keywords.extend(['amount', 'value', 'revenue', 'sales', 'level', 'type'])
        
        # Check if columns that should contain query-relevant data are empty
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in relevant_column_keywords:
                if keyword in col_lower:
                    # This column seems relevant to the query
                    non_null_count = df[col].count()
                    if non_null_count == 0:
                        penalty += 2.0  # Heavy penalty for completely empty relevant columns
                        print(f"   ⚠️ Penalty: '{col}' column is completely empty")
                    elif non_null_count < df.shape[0] * 0.1:  # Less than 10% filled
                        penalty += 1.0  # Moderate penalty for mostly empty relevant columns
                        print(f"   ⚠️ Penalty: '{col}' column is mostly empty ({non_null_count}/{df.shape[0]} filled)")
        
        # Check for sheets that appear to be templates or headers only
        if df.shape[0] < 5:
            penalty += 1.0  # Penalty for very small sheets
        
        # Smarter relevance checking - look for query terms but don't harshly penalize absence
        query_terms_bonus = self._calculate_query_relevance_bonus(df, query_lower)
        if query_terms_bonus < 0:
            penalty += abs(query_terms_bonus)  # Convert negative bonus to positive penalty
            print(f"   ⚠️ Penalty: Limited query term relevance: {query_terms_bonus}")
        else:
            # Don't apply penalty, this gets added as bonus in the main scoring
            pass
        
        return penalty
    
    def _calculate_query_relevance_bonus(self, df: pd.DataFrame, query_lower: str) -> float:
        """Calculate smarter query relevance without harsh penalties for missing literal matches"""
        bonus = 0.0
        
        # Instead of harsh penalties, look for positive indicators
        
        # Check for Ontario with flexible matching
        if 'ontario' in query_lower:
            ontario_indicators = 0
            entity_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['entity', 'region', 'location', 'office'])]
            
            for col in entity_cols:
                if col in df.columns:
                    # Look for various Ontario representations
                    ontario_variants = ['ontario', 'on', 'ont', 'ontario-', 'ontario_']
                    for variant in ontario_variants:
                        if df[col].astype(str).str.contains(variant, case=False, na=False).any():
                            ontario_indicators += 1
                            break
            
            if ontario_indicators > 0:
                bonus += 1.0  # Positive bonus for finding Ontario data
            else:
                # Only mild penalty if sheet seems to have entity data but no Ontario
                if entity_cols and len(entity_cols) > 0:
                    penalty = -0.5  # Mild penalty - sheet has entities but not Ontario
                    print(f"   ℹ️ Info: Sheet has entity columns but no obvious Ontario data")
                    return penalty
                # No penalty if sheet doesn't seem to be entity-based at all
        
        # Check for year data
        if any(year in query_lower for year in ['2023', '2024', '2022']):
            year_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['year', 'date', 'period'])]
            if year_cols:
                # Check if the specific year exists
                for col in year_cols:
                    if df[col].dtype in ['int64', 'float64']:
                        if any(year in str(df[col].unique()) for year in ['2023', '2024', '2022']):
                            bonus += 0.5
                            break
        
        # Check for financial terms with context
        financial_terms = ['revenue', 'sales', 'profit', 'income']
        if any(term in query_lower for term in financial_terms):
            # Look for amount/value columns
            value_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['amount', 'value', 'total', 'sum'])]
            if value_cols:
                bonus += 0.5
            
            # Look for financial categorization
            category_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['level', 'type', 'category', 'class'])]
            if category_cols:
                for col in category_cols:
                    if df[col].astype(str).str.contains('revenue|sales|profit|income', case=False, na=False).any():
                        bonus += 0.5
                        break
        
        return bonus


class FocusedAnalysisAgent:
    """
    PHASE 2: Multi-source analysis agent that analyzes pre-selected data sources
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _create_dynamic_expert_prompt(self, source_analysis: DataSourceAnalysis) -> str:
        """Create dynamic expert prompt based on domain analysis from router"""
        
        domain_type = source_analysis.domain_type
        data_structure = source_analysis.data_structure
        expertise_needed = source_analysis.expertise_needed
        key_fields = source_analysis.key_fields
        
        # Base expert identity
        if "Financial" in domain_type:
            expert_prompt = """
🏦 **YOU ARE A FINANCIAL ANALYSIS EXPERT**

FINANCIAL EXPERTISE:
- **Financial Statements**: Understand P&L (Profit & Loss), Balance Sheet, Cash Flow hierarchies
- **Key Metrics**: Revenue, COGS, Gross Profit, EBITDA, Net Income, Assets, Liabilities, Equity
- **Financial Ratios**: ROA, ROE, Debt-to-Equity, Current Ratio, Gross/Net Margins
- **Accounting Principles**: Recognize standard chart of accounts, general ledger structures
- **Period Analysis**: Handle fiscal years, quarters, monthly reporting periods
- **Currency & Amounts**: Work with monetary values, multi-currency scenarios

FINANCIAL DATA STRUCTURES:
- **Hierarchical Categories**: Level1 (BS/PnL/CF) → Level2 (line items) → Level3 (sub-categories)  
- **Standard Abbreviations**: BS=Balance Sheet, PnL/P&L=Profit & Loss, CF=Cash Flow
- **Common Fields**: Amount, Value, Entity, Period, Year, Scenario, Currency, Office, Measure

INTELLIGENT FIELD MATCHING FOR FINANCIAL DATA:
- **Cost of Goods Sold**: Try "COGS", "CoGS", "Cost of Sales", "Direct Costs", "Cost of Goods Sold"
- **Revenue**: Try "Revenue", "Sales", "Income", "Gross Revenue", "Net Sales"
- **Assets**: Try "Assets", "Total Assets", "Current Assets", "Fixed Assets"
- **Liabilities**: Try "Liabilities", "Total Liabilities", "Current Liabilities", "Long-term Debt"
- **Cash**: Try "Cash", "Cash and Equivalents", "Cash Position", "Liquid Assets"
- **EBITDA**: Try "EBITDA", "Operating Income", "Earnings", "Operating Profit"

FINANCIAL CALCULATIONS:
- Apply standard financial formulas (Gross Profit = Revenue - COGS)
- Calculate financial ratios using appropriate denominators
- Handle negative values appropriately (losses, liabilities)
- Understand financial statement reconciliation principles
"""

        elif "Mining Production" in domain_type:
            expert_prompt = """
⛏️ **YOU ARE A MINING PRODUCTION OPERATIONS EXPERT**

MINING PRODUCTION EXPERTISE:
- **Production Metrics**: Understand metal output, ore processing, extraction efficiency
- **Commodity Knowledge**: Gold, Copper, Iron Ore, Coal, Silver, Zinc, Lead, Nickel pricing and units
- **Production Process**: Ore extraction → Processing → Metal production → Recovery rates
- **Units**: Tonnes, ounces (oz), grams per tonne (g/t), percentages for recovery/grade
- **Key Performance Indicators**: Metal produced, ore processed, grade, recovery rate

MINING DATA STRUCTURES:
- **Production Hierarchy**: Site → Entity → Commodity → Processing → Output
- **Standard Fields**: MetalProduced, OreProcessed, Grade, RecoveryRate, Commodity, Site, Entity, Date
- **Scenarios**: Actual vs Budget vs Forecast production scenarios

INTELLIGENT FIELD MATCHING FOR MINING PRODUCTION:
- **Production Output**: Try "MetalProduced", "GoldProduced", "CopperProduced", "Output", "Produced", "Production"
- **Ore Input**: Try "OreProcessed", "OreTreated", "MillFeed", "TotalOre", "Input"
- **Quality Metrics**: Try "Grade", "HeadGrade", "g/t", "Quality", "Concentration"  
- **Efficiency**: Try "RecoveryRate", "Recovery", "Extraction", "Efficiency", "%Recovery"
- **Commodities**: Try exact names "Gold", "Copper", "Iron Ore", "Coal", plus variations

MINING CALCULATIONS:
- Calculate production rates (tonnes/day, ounces/month)
- Compute recovery efficiency (metal out / theoretical maximum)
- Analyze grade trends and processing performance
- Handle commodity-specific units and conversions
"""

        elif "Mining Operations" in domain_type:
            expert_prompt = """
🚛 **YOU ARE A MINING OPERATIONS & LOGISTICS EXPERT**

MINING OPERATIONS EXPERTISE:  
- **Operational Metrics**: Equipment utilization, material movement, logistics efficiency
- **Equipment Management**: Downtime tracking, maintenance schedules, utilization rates
- **Material Handling**: Tonnage moved, haulage operations, transportation logistics
- **Performance Tracking**: Operational efficiency, cost per tonne, productivity metrics
- **Resource Management**: Equipment deployment, operational planning, cost optimization

OPERATIONS DATA STRUCTURES:
- **Operational Hierarchy**: Site → Equipment → Activity → Performance
- **Standard Fields**: TonnesMoved, EquipmentUtilization, DowntimeHours, HaulageVolume, FuelConsumption
- **Tracking Elements**: Date, Site, Equipment, Activity, Performance Metrics

INTELLIGENT FIELD MATCHING FOR MINING OPERATIONS:
- **Material Movement**: Try "TonnesMoved", "HaulageVolume", "Moved", "Transported", "Volume"
- **Equipment Performance**: Try "EquipmentUtilization", "Utilization", "Usage", "Performance"
- **Maintenance**: Try "DowntimeHours", "Downtime", "Maintenance", "OutOfService"
- **Logistics**: Try "Transportation", "Haulage", "Movement", "Logistics"
- **Efficiency**: Try "Efficiency", "Performance", "Productivity", "Rate"

OPERATIONAL CALCULATIONS:
- Calculate utilization rates (active time / total time)
- Compute material movement rates (tonnes/hour, trips/day)
- Analyze downtime patterns and maintenance efficiency
- Measure cost per tonne and operational productivity
"""

        else:
            # Generic expert prompt for unknown domains
            expert_prompt = f"""
🔬 **YOU ARE A DATA ANALYSIS EXPERT**

GENERAL DATA ANALYSIS EXPERTISE:
Based on the router's analysis, this appears to be: {domain_type}
Data structure description: {data_structure}
Expertise needed: {expertise_needed}

KEY FIELDS IDENTIFIED: {key_fields}

Apply your general analytical skills to:
- Understand the data structure and relationships
- Identify patterns and trends in the data
- Perform accurate calculations based on the query requirements
- Use appropriate statistical and analytical methods
"""

        # Add common intelligent matching guidance
        expert_prompt += f"""

🎯 **DOMAIN-SPECIFIC CONTEXT FOR THIS ANALYSIS**:
- **Domain Type**: {domain_type}  
- **Data Structure**: {data_structure}
- **Key Fields Available**: {key_fields}
- **Expertise Required**: {expertise_needed}

**CRITICAL SUCCESS FACTORS**:
1. **Apply Domain Knowledge**: Use your specialized expertise to interpret the query correctly
2. **Intelligent Field Discovery**: Don't give up if exact field names don't match - try variations
3. **Domain-Appropriate Calculations**: Use industry-standard formulas and methods
4. **Unit Awareness**: Apply correct units and scaling based on domain conventions
5. **Context Understanding**: Interpret results within the proper business/operational context
"""

        return expert_prompt
    
    def analyze_selected_sources(self, source_analyses: List[DataSourceAnalysis], query: str, discovery_agent: 'DataDiscoveryAgent' = None) -> str:
        """Analyze recommended data sources with intelligent cross-source validation"""
        recommended_sources = [a for a in source_analyses if a.recommended]
        
        if not recommended_sources:
            return "No relevant data sources found for analysis"
        
        print(f"🎯 PHASE 2: Analyzing {len(recommended_sources)} recommended sources with cross-source validation")
        
        analysis_report = "🔍 **MULTI-SOURCE CROSS-VALIDATION ANALYSIS REPORT**\n"
        analysis_report += "=" * 60 + "\n\n"
        
        # Collect all results for cross-validation
        source_results = []
        
        for source_analysis in recommended_sources:
            print(f"   📊 Analyzing {source_analysis.source_type}: {source_analysis.name}...")
            analysis_report += f"**{source_analysis.source_type.upper()}: {source_analysis.name}**\n"
            analysis_report += f"Selected because: {source_analysis.reason}\n"
            analysis_report += f"Data summary: {source_analysis.data_summary}\n\n"
            
            try:
                # Get cached dataframe
                cache_key = f"{source_analysis.source_type}_{source_analysis.name}"
                if source_analysis.source_type == "excel_sheet":
                    cache_key = f"excel_{source_analysis.name}"
                elif source_analysis.source_type == "csv":
                    cache_key = f"csv_{source_analysis.name}"
                
                if discovery_agent and hasattr(discovery_agent, 'cached_dataframes') and cache_key in discovery_agent.cached_dataframes:
                    df = discovery_agent.cached_dataframes[cache_key]
                    print(f"   📊 USING CACHED DATA: {df.shape[0]} rows, {df.shape[1]} columns")
                else:
                    # Fallback: Load fresh
                    if source_analysis.source_type == "excel_sheet":
                        df = pd.read_excel(source_analysis.file_path, sheet_name=source_analysis.sheet_name)
                    elif source_analysis.source_type == "csv":
                        df = pd.read_csv(source_analysis.file_path)
                    else:
                        print(f"   ❌ Unsupported source type: {source_analysis.source_type}")
                        continue
                    print(f"   📊 LOADED FRESH DATA: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Verify full data is loaded
                analysis_report += f"🔍 **FULL DATA VERIFICATION**: Using {df.shape[0]} total rows, {df.shape[1]} columns\n\n"
                
                # Show data structure for transparency
                data_preview = self._generate_data_preview(df)
                analysis_report += f"Data structure preview:\n{data_preview}\n\n"
                
                # Focused analysis on this specific source with calculation details
                result, calculation_details = self._analyze_source_focused_with_details(df, source_analysis, query)
                
                analysis_report += f"Calculation process:\n{calculation_details}\n\n"
                
                if result:
                    # Score the result
                    result_score = self._score_analysis_result(result, query)
                    analysis_report += f"Result: {result}\n"
                    analysis_report += f"Confidence score: {result_score:.1f}\n\n"
                    
                    # Store result for cross-validation
                    source_results.append({
                        'source': source_analysis,
                        'result': result,
                        'score': result_score,
                        'details': calculation_details,
                        'dataframe': df
                    })
                    
                    print(f"   ✅ Result found (score: {result_score:.1f})")
                else:
                    analysis_report += "Result: No relevant data found\n\n"
                    print(f"   ❌ No relevant data found")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                analysis_report += f"{error_msg}\n\n"
                print(f"   ❌ Error analyzing {source_analysis.name}: {e}")
                continue
            
            analysis_report += "-" * 40 + "\n\n"
        
        # Multi-Source Cross-Validation Phase
        if len(source_results) > 1:
            print(f"🔄 CROSS-SOURCE VALIDATION: Comparing {len(source_results)} results...")
            cross_validation_report = self._cross_source_validation(source_results, query)
            analysis_report += cross_validation_report
            
            # Get the best result after cross-validation
            best_result_info = self._select_best_source_result(source_results, query)
        elif len(source_results) == 1:
            print("📋 Single result found, no cross-validation needed")
            best_result_info = source_results[0]
        else:
            print("❌ No valid results found")
            return f"❌ No relevant data found in the recommended sources\n\n📊 **ANALYSIS REPORT**:\n{analysis_report}"
        
        # Final result with complete transparency
        if best_result_info:
            best_result = best_result_info['result']
            best_calculation_details = best_result_info['details']
            best_source = best_result_info['source']
            best_df = best_result_info['dataframe']
            
            final_report = f"✅ **FINAL ANSWER**: {best_result}\n\n"
            final_report += f"🧮 **HOW THIS WAS CALCULATED**:\n{best_calculation_details}\n\n"
            
            # Add verification report
            verification_report = self._create_verification_report(best_df, query, best_result, f"{best_source.source_type}:{best_source.name}")
            final_report += f"🔍 **VERIFICATION REPORT**:\n{verification_report}\n\n"
            
            final_report += f"🎯 **Analysis Method**: Multi-Source Intelligence (Discovery → Selection → Cross-Validation)\n"
            final_report += f"📊 **Selected Source**: {best_source.source_type.upper()} - {best_source.name}\n\n"
            final_report += f"📋 **DETAILED MULTI-SOURCE ANALYSIS REPORT**:\n{analysis_report}"
            return final_report
        else:
            return f"❌ No relevant data found in the recommended sources\n\n📊 **ANALYSIS REPORT**:\n{analysis_report}"
    
    def _react_cross_check_results(self, sheet_results: List[dict], query: str) -> str:
        """ReAct mechanism to cross-check results across multiple sheets"""
        
        cross_check_report = "🔄 **REACT CROSS-CHECKING PHASE**\n"
        cross_check_report += "=" * 50 + "\n\n"
        
        cross_check_report += f"Comparing {len(sheet_results)} results:\n\n"
        
        # Extract numerical values from results for comparison
        numerical_results = []
        for i, result_info in enumerate(sheet_results):
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', result_info['result'])
            sheet_name = result_info['sheet'].name
            cross_check_report += f"**{sheet_name}**: {result_info['result']}\n"
            cross_check_report += f"  - Extracted numbers: {numbers}\n"
            cross_check_report += f"  - Confidence score: {result_info['score']:.1f}\n\n"
            
            if numbers:
                # Convert to float for comparison (remove commas)
                clean_numbers = [float(num.replace(',', '')) for num in numbers if num.replace(',', '').replace('.', '').isdigit()]
                numerical_results.append({
                    'sheet': sheet_name,
                    'numbers': clean_numbers,
                    'result_info': result_info
                })
        
        # Cross-checking logic
        cross_check_report += "**Cross-Check Analysis**:\n"
        
        if len(numerical_results) >= 2:
            # Check for consistency
            consistency_analysis = self._analyze_result_consistency(numerical_results, query)
            cross_check_report += consistency_analysis + "\n"
            
            # Check for potential data relationships
            relationship_analysis = self._analyze_data_relationships(sheet_results, query)
            cross_check_report += relationship_analysis + "\n"
            
        else:
            cross_check_report += "- Insufficient numerical results for cross-checking\n"
        
        return cross_check_report
    
    def _analyze_result_consistency(self, numerical_results: List[dict], query: str) -> str:
        """Analyze consistency between numerical results"""
        
        analysis = "**Consistency Analysis**:\n"
        
        if len(numerical_results) < 2:
            return analysis + "- Only one numerical result available"
        
        # Get primary numbers (usually the largest or most relevant)
        primary_numbers = []
        for result in numerical_results:
            if result['numbers']:
                primary_number = max(result['numbers'])  # Assume largest is most relevant
                primary_numbers.append({
                    'sheet': result['sheet'],
                    'value': primary_number,
                    'result_info': result['result_info']
                })
        
        if len(primary_numbers) >= 2:
            # Check if results are similar (within 10% tolerance)
            values = [p['value'] for p in primary_numbers]
            max_val = max(values)
            min_val = min(values)
            
            if max_val > 0:
                variance_percent = ((max_val - min_val) / max_val) * 100
                
                if variance_percent < 10:
                    analysis += f"- Results are CONSISTENT (variance: {variance_percent:.1f}%)\n"
                    analysis += f"- Values range from {min_val:,.0f} to {max_val:,.0f}\n"
                    analysis += f"- Likely measuring the same metric\n"
                elif variance_percent < 50:
                    analysis += f"- Results are MODERATELY CONSISTENT (variance: {variance_percent:.1f}%)\n"
                    analysis += f"- Values range from {min_val:,.0f} to {max_val:,.0f}\n"
                    analysis += f"- May be different time periods or categories\n"
                else:
                    analysis += f"- Results are INCONSISTENT (variance: {variance_percent:.1f}%)\n"
                    analysis += f"- Values range from {min_val:,.0f} to {max_val:,.0f}\n"
                    analysis += f"- Likely measuring different metrics or have errors\n"
            
            # Show individual sheet analysis
            analysis += f"\n**Individual Sheet Results**:\n"
            for p in primary_numbers:
                analysis += f"- {p['sheet']}: {p['value']:,.0f} (score: {p['result_info']['score']:.1f})\n"
        
        return analysis
    
    def _analyze_data_relationships(self, sheet_results: List[dict], query: str) -> str:
        """Analyze potential relationships between data in different sheets"""
        
        analysis = "**Data Relationship Analysis**:\n"
        
        # Analyze sheet names for relationships
        sheet_names = [result['sheet'].name for result in sheet_results]
        analysis += f"- Analyzing sheets: {', '.join(sheet_names)}\n"
        
        # Check for hierarchical relationships
        hierarchical_terms = ['summary', 'detail', 'breakdown', 'total', 'sub']
        summary_sheets = [name for name in sheet_names if any(term in name.lower() for term in hierarchical_terms)]
        
        if summary_sheets:
            analysis += f"- Detected potential summary sheets: {summary_sheets}\n"
            analysis += f"- Consider using summary sheets for high-level queries\n"
        
        # Check for temporal relationships
        temporal_terms = ['2023', '2024', '2022', 'current', 'historical', 'ytd', 'quarterly']
        temporal_sheets = [name for name in sheet_names if any(term in name.lower() for term in temporal_terms)]
        
        if temporal_sheets:
            analysis += f"- Detected time-specific sheets: {temporal_sheets}\n"
            
        # Check for regional/entity relationships
        regional_terms = ['ontario', 'quebec', 'bc', 'alberta', 'region', 'entity', 'office']
        regional_sheets = [name for name in sheet_names if any(term in name.lower() for term in regional_terms)]
        
        if regional_sheets:
            analysis += f"- Detected entity/regional sheets: {regional_sheets}\n"
        
        return analysis
    
    def _select_best_result_after_cross_check(self, sheet_results: List[dict], query: str) -> dict:
        """Select the best result after cross-checking analysis"""
        
        print("🏆 Selecting best result after cross-checking...")
        
        # Enhanced scoring based on cross-check analysis
        for result in sheet_results:
            enhanced_score = result['score']
            
            # Bonus for summary sheets if query is high-level
            if any(term in query.lower() for term in ['total', 'overall', 'summary']):
                if any(term in result['sheet'].name.lower() for term in ['summary', 'total', 'main']):
                    enhanced_score += 1.0
                    print(f"   📊 Bonus for summary sheet: {result['sheet'].name}")
            
            # Bonus for sheets with complete data
            if 'complete' in result['details'].lower() or 'found' in result['details'].lower():
                enhanced_score += 0.5
            
            # Penalty for incomplete results
            if any(term in result['result'].lower() for term in ['not found', 'no data', 'insufficient']):
                enhanced_score -= 2.0
            
            result['enhanced_score'] = enhanced_score
            print(f"   📊 {result['sheet'].name}: original={result['score']:.1f}, enhanced={enhanced_score:.1f}")
        
        # Select best result
        best_result = max(sheet_results, key=lambda x: x['enhanced_score'])
        print(f"🏆 Selected: {best_result['sheet'].name} (enhanced score: {best_result['enhanced_score']:.1f})")
        
        return best_result
    
    def _generate_data_preview(self, df: pd.DataFrame) -> str:
        """Generate a preview of the data structure for transparency"""
        preview = f"🔍 FULL DATASET VERIFICATION:\n"
        preview += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        preview += f"Columns: {list(df.columns)}\n"
        preview += f"Data types: {dict(df.dtypes)}\n"
        preview += f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB\n\n"
        
        # Show first and last few rows to confirm full dataset
        preview += f"First 3 rows (to confirm data structure):\n{df.head(3).to_string()}\n\n"
        preview += f"Last 3 rows (to confirm full dataset loaded):\n{df.tail(3).to_string()}\n\n"
        
        # Show unique value counts for key categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        preview += f"Categorical column unique value counts:\n"
        for col in categorical_cols[:3]:  # Show first 3 categorical columns
            unique_count = df[col].nunique()
            preview += f"  {col}: {unique_count} unique values\n"
            if unique_count <= 10:
                preview += f"    Values: {df[col].unique().tolist()}\n"
            else:
                preview += f"    Sample: {df[col].unique()[:5].tolist()}...\n"
        
        # Show sample of numeric data if available
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            preview += f"\nNumeric columns summary:\n{df[numeric_cols].describe().to_string()}"
        
        return preview
    
    def _analyze_sheet_focused_with_details(self, df: pd.DataFrame, sheet_analysis: DataSourceAnalysis, query: str) -> tuple[Optional[str], str]:
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
        CRITICAL: You have access to a loaded pandas DataFrame called 'df' with {df.shape[0]} rows and {df.shape[1]} columns.
        This DataFrame contains the data from Excel sheet "{sheet_analysis.name}".
        
        DataFrame details:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {dict(df.dtypes)}
        - This is REAL DATA loaded from Excel sheet "{sheet_analysis.name}"
        
        Sheet selection reason: {sheet_analysis.reason}
        Data summary: {sheet_analysis.data_summary}
        
        Your task: {query}
        
        IMPORTANT: Do NOT say you don't have access to data. You DO have access to the DataFrame 'df'.
        
        BEFORE ANSWERING: First run these commands to verify data access:
        1. print(f"DataFrame shape: {{df.shape}}")
        2. print(f"Columns: {{df.columns.tolist()}}")
        3. print(f"First few rows: {{df.head(3)}}")
        4. Then proceed with your analysis
        
        If you cannot access 'df', there is a technical error - do not make excuses about file access.
        
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
        
        CRITICAL REQUIREMENTS:
        1. Always show the exact pandas code you execute
        2. Show row counts after each filter operation
        3. Display the actual filtered data (not just the result)
        4. Verify you're using the FULL dataset (not just a sample)
        5. Show the shape of the original dataframe before filtering
        
        EXAMPLE VERIFICATION FORMAT:
        "Original dataframe shape: (5000, 10) - confirms full dataset loaded
         Filter 1: df[df['Entity'] == 'Ontario'] → 1200 rows remaining
         Filter 2: filtered_df[filtered_df['Year'] == 2023] → 180 rows remaining  
         Filter 3: final_df[final_df['Level2'] == 'Revenue'] → 15 rows remaining
         
         Filtered data preview:
         [Show first few rows of the filtered data]
         
         Calculation: sum of 15 revenue values = $X,XXX,XXX"
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
            calculation_log.append(f"Dataframe verification: shape={df.shape}, columns={list(df.columns)}")
            
            # Test that the agent can access the dataframe
            test_query = f"What is the shape of the dataframe df? Show df.head(2) and df.columns"
            test_result = agent.run(test_query)
            calculation_log.append(f"Dataframe access test: {test_result}")
            
            # Now run the actual query
            result = agent.run(query)
            calculation_log.append(f"Agent completed analysis")
            calculation_log.append(f"Raw result: {result}")
            
            # CRITICAL: Manual verification of the filtering
            manual_verification = self._manual_filter_verification(df, query)
            calculation_log.append("Manual filter verification:")
            calculation_log.append(manual_verification)
            
            # Parse and enhance the calculation details
            calculation_details = "\n".join(calculation_log)
            calculation_details += f"\n\nAgent Response:\n{result}"
            
            # Check if the agent gave a proper answer or complained about file access
            if "don't have access" in result.lower() or "can't access" in result.lower() or "csv file" in result.lower():
                print("     🔧 Agent failed to access data, trying manual analysis...")
                manual_result = self._manual_analysis_fallback(df, query, sheet_analysis.name)
                calculation_log.append("Agent failed to access data, using manual fallback:")
                calculation_log.append(manual_result)
                calculation_details = "\n".join(calculation_log)
                return manual_result, calculation_details
            
            return result, calculation_details
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}"
            calculation_log.append(error_details)
            print(f"     ⚠️ Analysis error: {e}")
            
            # Try manual fallback
            try:
                print("     🔧 Trying manual analysis fallback...")
                manual_result = self._manual_analysis_fallback(df, query, sheet_analysis.name)
                calculation_log.append("Using manual fallback due to agent error:")
                calculation_log.append(manual_result)
                calculation_details = "\n".join(calculation_log)
                return manual_result, calculation_details
            except Exception as fallback_error:
                calculation_log.append(f"Manual fallback also failed: {str(fallback_error)}")
                calculation_details = "\n".join(calculation_log)
                return None, calculation_details
    
    def _analyze_source_focused_with_details(self, df: pd.DataFrame, source_analysis: DataSourceAnalysis, query: str) -> tuple[Optional[str], str]:
        """Perform focused analysis on any data source and return both result and calculation details"""
        
        calculation_log = []
        calculation_log.append(f"Starting analysis on {source_analysis.source_type}: {source_analysis.name}")
        calculation_log.append(f"Query: {query}")
        calculation_log.append(f"Data shape: {df.shape}")
        calculation_log.append(f"Available columns: {list(df.columns)}")
        
        # Create dynamic expert prompt based on domain analysis
        expertise_prompt = self._create_dynamic_expert_prompt(source_analysis)
        
        # Enhanced focused prompt with domain expertise
        focused_prompt = f"""
        {expertise_prompt}
        
        CRITICAL: You have access to a loaded pandas DataFrame called 'df' with {df.shape[0]} rows and {df.shape[1]} columns.
        This DataFrame contains data from {source_analysis.source_type}: "{source_analysis.name}".
        
        DataFrame details:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {dict(df.dtypes)}
        - Source: {source_analysis.source_type} - {source_analysis.name}
        
        DOMAIN ANALYSIS:
        - Domain Type: {source_analysis.domain_type}
        - Data Structure: {source_analysis.data_structure}
        - Key Fields: {source_analysis.key_fields}
        - Field Variations Available: {list(source_analysis.field_variations.keys())}
        
        Source selection reason: {source_analysis.reason}
        Data summary: {source_analysis.data_summary}
        
        Your task: {query}
        
        IMPORTANT: Do NOT say you don't have access to data. You DO have access to the DataFrame 'df'.
        
        INTELLIGENT FIELD MATCHING:
        If you cannot find a field directly, try these intelligent approaches:
        1. **Abbreviations**: Try common abbreviations from field_variations: {source_analysis.field_variations}
        2. **Case variations**: Try different capitalizations (e.g., COGS, CoGS, Cogs, cogs)
        3. **Synonyms**: Try related terms based on your domain expertise
        4. **Partial matching**: Look for columns containing the key terms
        5. **Data type analysis**: Check if the query asks for financial amounts, quantities, percentages, etc.
        
        CALCULATION TRANSPARENCY INSTRUCTIONS:
        1. SHOW YOUR WORK: Explain every step of your calculation
        2. SHOW PANDAS CODE: Show the exact pandas code you're running  
        3. SHOW DATA FILTERS: Show exactly what data you're filtering/selecting
        4. SHOW ROW COUNTS: Show how many rows match your filters
        5. VERIFY YOUR LOGIC: Double-check your approach and calculations
        6. SHOW FIELD MATCHING: If using abbreviations/variations, explain which ones you tried
        
        REQUIRED OUTPUT FORMAT:
        1. Expert domain analysis: Apply your specialized knowledge to understand the query
        2. Intelligent field discovery: Show how you found the right columns (including variations tried)
        3. Data exploration: What columns and data structure you found
        4. Filtering logic: Exactly what filters you applied and why
        5. Row verification: How many rows match your filters
        6. Calculation steps: Step-by-step calculation with actual values using domain expertise
        7. Final answer: The result with proper units/context and domain insights
        """
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True,
            prefix=focused_prompt
        )
        
        try:
            calculation_log.append("Starting pandas agent analysis...")
            calculation_log.append(f"Dataframe verification: shape={df.shape}, columns={list(df.columns)}")
            
            # Run the analysis
            result = agent.run(query)
            calculation_log.append(f"Agent completed analysis")
            calculation_log.append(f"Raw result: {result}")
            
            # Parse and enhance the calculation details
            calculation_details = "\n".join(calculation_log)
            calculation_details += f"\n\nAgent Response:\n{result}"
            
            # Check if the agent gave a proper answer
            if "don't have access" in result.lower() or "can't access" in result.lower():
                print(f"     🔧 Agent failed to access data, trying manual analysis...")
                manual_result = self._manual_source_analysis(df, source_analysis, query)
                calculation_log.append("Agent failed to access data, using manual fallback:")
                calculation_log.append(manual_result)
                calculation_details = "\n".join(calculation_log)
                return manual_result, calculation_details
            
            return result, calculation_details
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}"
            calculation_log.append(error_details)
            print(f"     ⚠️ Analysis error: {e}")
            
            # Try manual fallback
            try:
                print(f"     🔧 Trying manual analysis fallback...")
                manual_result = self._manual_source_analysis(df, source_analysis, query)
                calculation_log.append("Using manual fallback due to agent error:")
                calculation_log.append(manual_result)
                calculation_details = "\n".join(calculation_log)
                return manual_result, calculation_details
            except Exception as fallback_error:
                calculation_log.append(f"Manual fallback also failed: {str(fallback_error)}")
                calculation_details = "\n".join(calculation_log)
                return None, calculation_details
    
    def _manual_source_analysis(self, df: pd.DataFrame, source_analysis: DataSourceAnalysis, query: str) -> str:
        """Manual fallback analysis for any data source type"""
        
        analysis = []
        analysis.append(f"Manual Analysis of {source_analysis.source_type}: {source_analysis.name}")
        analysis.append(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        analysis.append(f"Columns: {list(df.columns)}")
        
        try:
            query_lower = query.lower()
            
            # Look for relevant columns based on query keywords
            relevant_cols = []
            for col in df.columns:
                col_lower = col.lower()
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 3 and word in col_lower:
                        relevant_cols.append(col)
                        break
            
            if relevant_cols:
                analysis.append(f"Found potentially relevant columns: {relevant_cols}")
                
                # Show sample data from relevant columns
                for col in relevant_cols[:3]:  # Limit to 3 columns
                    analysis.append(f"Sample data from {col}: {df[col].head(3).tolist()}")
                    if df[col].dtype in ['int64', 'float64']:
                        analysis.append(f"{col} statistics: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
            
            # Try to find numeric columns for potential calculations
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                analysis.append(f"Numeric columns available: {numeric_cols}")
                
                # Look for year/date filters
                if any(year in query_lower for year in ['2022', '2023', '2024']):
                    year_cols = [col for col in df.columns if any(word in col.lower() for word in ['year', 'date', 'period'])]
                    if year_cols:
                        analysis.append(f"Date/year columns found: {year_cols}")
                        for year_col in year_cols[:1]:  # Check first year column
                            unique_years = df[year_col].unique()
                            analysis.append(f"Available years in {year_col}: {sorted(unique_years) if len(unique_years) < 20 else 'Many years'}")
                
                # Look for entity/location filters
                if any(entity in query_lower for entity in ['ontario', 'quebec', 'alberta', 'gold', 'copper', 'production']):
                    text_cols = df.select_dtypes(include=['object']).columns.tolist()
                    for col in text_cols[:3]:  # Check first 3 text columns
                        unique_vals = df[col].unique()
                        if len(unique_vals) < 50:  # Reasonable number of categories
                            analysis.append(f"Categories in {col}: {unique_vals[:10].tolist()}")
                
                # Simple aggregation attempt
                if 'production' in query_lower and 'gold' in query_lower:
                    gold_cols = [col for col in numeric_cols if 'gold' in col.lower()]
                    if gold_cols:
                        for gold_col in gold_cols[:1]:
                            total = df[gold_col].sum()
                            analysis.append(f"Total {gold_col}: {total:,.2f}")
                            
                            # Try to filter by year if available
                            year_cols = [col for col in df.columns if any(word in col.lower() for word in ['year', 'date'])]
                            if year_cols and '2024' in query_lower:
                                year_col = year_cols[0]
                                filtered_df = df[df[year_col].astype(str).str.contains('2024', na=False)]
                                if not filtered_df.empty:
                                    filtered_total = filtered_df[gold_col].sum()
                                    analysis.append(f"2024 {gold_col}: {filtered_total:,.2f} ({len(filtered_df)} records)")
            
            if len(analysis) <= 3:  # If we didn't find much
                analysis.append("No obvious matches found for the query in this data source")
                analysis.append(f"Consider checking if the query matches the data type: {source_analysis.source_type}")
            
        except Exception as e:
            analysis.append(f"Manual analysis error: {str(e)}")
        
        return "\n".join(analysis)
    
    def _cross_source_validation(self, source_results: List[dict], query: str) -> str:
        """Perform cross-source validation to compare results from different data sources"""
        
        validation_report = "🔄 **CROSS-SOURCE VALIDATION REPORT**\n"
        validation_report += "=" * 50 + "\n\n"
        
        # Analyze consistency between sources
        validation_report += "**Source Comparison**:\n"
        for i, result in enumerate(source_results, 1):
            source = result['source']
            validation_report += f"{i}. {source.source_type.upper()}: {source.name}\n"
            validation_report += f"   Score: {result['score']:.1f}\n"
            validation_report += f"   Result: {result['result'][:200]}{'...' if len(result['result']) > 200 else ''}\n\n"
        
        # Check for data consistency patterns
        validation_report += "**Consistency Analysis**:\n"
        
        # Look for numeric values in results for comparison
        numeric_values = []
        for result in source_results:
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', result['result'])
            if numbers:
                # Try to parse the largest number (likely the main result)
                try:
                    largest_num = max([float(n.replace(',', '')) for n in numbers])
                    numeric_values.append((result['source'].name, largest_num, result['source'].source_type))
                except:
                    pass
        
        if len(numeric_values) > 1:
            validation_report += "Numeric values found for comparison:\n"
            for name, value, source_type in numeric_values:
                validation_report += f"  {source_type}: {name} = {value:,.2f}\n"
            
            # Check if values are similar
            values = [v[1] for v in numeric_values]
            max_val, min_val = max(values), min(values)
            if max_val > 0:
                variance = (max_val - min_val) / max_val
                if variance < 0.1:  # Within 10%
                    validation_report += "✅ Values are consistent (within 10% variance)\n"
                elif variance < 0.5:  # Within 50%
                    validation_report += "⚠️ Values show moderate variance (10-50%)\n"
                else:
                    validation_report += "❌ Values show high variance (>50%) - may indicate different metrics\n"
        
        # Source type complementarity analysis
        source_types = [result['source'].source_type for result in source_results]
        if 'csv' in source_types and 'excel_sheet' in source_types:
            validation_report += "\n**Complementarity**: CSV (operational) + Excel (financial) sources provide comprehensive coverage\n"
        
        validation_report += "\n" + "-" * 50 + "\n\n"
        return validation_report
    
    def _select_best_source_result(self, source_results: List[dict], query: str) -> dict:
        """Select the best result respecting the router's original prioritization"""
        
        print("🏆 Selecting best result based on router's recommendations...")
        
        # Sort by the router's original relevance scores (highest first)
        # This respects the intelligent routing decision without second-guessing it
        source_results_sorted = sorted(source_results, key=lambda x: x['source'].relevance_score, reverse=True)
        
        # Log the router's prioritization
        for result in source_results_sorted:
            source = result['source']
            print(f"   📊 {source.source_type}:{source.name}: router_score={source.relevance_score:.1f}, result_quality_score={result['score']:.1f}")
        
        # Only override router's decision if there's a significant quality issue with the top choice
        best_result = source_results_sorted[0]
        
        # Check if the router's top choice has a fundamentally broken result
        if any(term in best_result['result'].lower() for term in ['error', 'failed', 'could not', 'unable to']):
            print(f"   ⚠️ Router's top choice has execution errors, checking alternatives...")
            
            # Look for the highest-scoring alternative that actually worked
            for result in source_results_sorted[1:]:
                if not any(term in result['result'].lower() for term in ['error', 'failed', 'could not', 'unable to']):
                    print(f"   🔄 Switching to: {result['source'].source_type}:{result['source'].name} (functional alternative)")
                    best_result = result
                    break
        
        print(f"🏆 Selected: {best_result['source'].source_type}:{best_result['source'].name} (router score: {best_result['source'].relevance_score:.1f})")
        
        return best_result

    def _analyze_sheet_focused(self, df: pd.DataFrame, sheet_analysis: DataSourceAnalysis, query: str) -> Optional[str]:
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
    
    def _manual_filter_verification(self, df: pd.DataFrame, query: str) -> str:
        """Manually verify filtering to catch discrepancies"""
        verification = []
        verification.append("=== MANUAL FILTER VERIFICATION ===")
        verification.append(f"Original dataframe shape: {df.shape}")
        verification.append(f"Total rows available: {len(df)}")
        verification.append("")
        
        # Try to identify likely filter terms from the query
        query_lower = query.lower()
        potential_filters = {}
        
        # Look for entity/location terms
        entity_terms = ['ontario', 'quebec', 'alberta', 'bc', 'nova scotia', 'manitoba', 'saskatchewan']
        for term in entity_terms:
            if term in query_lower:
                potential_filters['entity'] = term.title()
        
        # Look for year terms
        import re
        years = re.findall(r'\b(20\d{2})\b', query)
        if years:
            potential_filters['year'] = int(years[0])
        
        # Look for metric terms
        metric_terms = ['revenue', 'sales', 'income', 'profit', 'cost', 'expense']
        for term in metric_terms:
            if term in query_lower:
                potential_filters['metric'] = term.title()
        
        verification.append(f"Detected potential filters from query: {potential_filters}")
        verification.append("")
        
        # Try to apply these filters manually and show results
        if potential_filters:
            verification.append("Manual filtering test:")
            
            # Find columns that might contain these values
            for filter_type, filter_value in potential_filters.items():
                verification.append(f"\nLooking for {filter_type} = '{filter_value}':")
                
                matching_columns = []
                for col in df.columns:
                    if filter_type == 'entity' and any(term in col.lower() for term in ['entity', 'region', 'location', 'office', 'company']):
                        matching_columns.append(col)
                    elif filter_type == 'year' and any(term in col.lower() for term in ['year', 'date', 'period']):
                        matching_columns.append(col)
                    elif filter_type == 'metric' and any(term in col.lower() for term in ['level', 'type', 'category', 'class', 'metric']):
                        matching_columns.append(col)
                
                verification.append(f"  Potential columns: {matching_columns}")
                
                # Check actual values in these columns
                for col in matching_columns[:2]:  # Limit to first 2 to avoid spam
                    if col in df.columns:
                        if filter_type == 'year':
                            # For year, check if any values match
                            matches = df[df[col] == filter_value]
                        else:
                            # For text, check for partial matches
                            matches = df[df[col].astype(str).str.contains(str(filter_value), case=False, na=False)]
                        
                        verification.append(f"  Column '{col}': {len(matches)} rows match '{filter_value}'")
                        
                        if len(matches) > 0:
                            verification.append(f"    Sample matching values: {matches[col].unique()[:5].tolist()}")
        
        verification.append(f"\nIf the agent found only a few rows, check:")
        verification.append(f"1. Are the filter column names correct?")
        verification.append(f"2. Are the filter values case-sensitive?")
        verification.append(f"3. Are there extra spaces or formatting issues?")
        verification.append(f"4. Is the agent using the full dataframe?")
        
        return "\n".join(verification)
    
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
    
    def _manual_analysis_fallback(self, df: pd.DataFrame, query: str, sheet_name: str) -> str:
        """Manual analysis fallback when pandas agent fails"""
        analysis = []
        analysis.append("🔧 **MANUAL ANALYSIS FALLBACK**")
        analysis.append(f"Sheet: {sheet_name}")
        analysis.append(f"Query: {query}")
        analysis.append(f"DataFrame shape: {df.shape}")
        analysis.append(f"Columns: {list(df.columns)}")
        analysis.append("")
        
        query_lower = query.lower()
        
        try:
            # Try to identify filter criteria from the query
            filters = {}
            
            # Look for entity/location
            entity_terms = ['ontario', 'quebec', 'alberta', 'bc', 'nova scotia']
            for term in entity_terms:
                if term in query_lower:
                    filters['entity'] = term.title()
                    break
            
            # Look for year
            import re
            years = re.findall(r'\b(20\d{2})\b', query)
            if years:
                filters['year'] = int(years[0])
            
            # Look for metric
            metric_terms = ['revenue', 'sales', 'income', 'profit']
            for term in metric_terms:
                if term in query_lower:
                    filters['metric'] = term.title()
                    break
            
            analysis.append(f"Detected filters: {filters}")
            analysis.append("")
            
            # Try to apply filters
            filtered_df = df.copy()
            filter_steps = []
            
            for filter_type, filter_value in filters.items():
                # Find potential columns for this filter
                potential_cols = []
                
                if filter_type == 'entity':
                    potential_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['entity', 'region', 'location', 'office', 'company'])]
                elif filter_type == 'year':
                    potential_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['year', 'date', 'period'])]
                elif filter_type == 'metric':
                    potential_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['level', 'type', 'category', 'class'])]
                
                # Apply filter if we found a suitable column
                for col in potential_cols:
                    if col in filtered_df.columns:
                        if filter_type == 'year':
                            mask = filtered_df[col] == filter_value
                        else:
                            mask = filtered_df[col].astype(str).str.contains(str(filter_value), case=False, na=False)
                        
                        before_count = len(filtered_df)
                        filtered_df = filtered_df[mask]
                        after_count = len(filtered_df)
                        
                        filter_steps.append(f"Filter {col} = '{filter_value}': {before_count} → {after_count} rows")
                        analysis.append(f"Applied filter: {col} = '{filter_value}' → {after_count} rows remaining")
                        break
            
            analysis.append("")
            analysis.append("Filter steps:")
            for step in filter_steps:
                analysis.append(f"  - {step}")
            
            analysis.append("")
            analysis.append(f"Final filtered data: {filtered_df.shape[0]} rows")
            
            if len(filtered_df) > 0:
                # Show filtered data
                analysis.append(f"Filtered data preview:")
                analysis.append(filtered_df.head(5).to_string())
                
                # Try to find amount/value columns
                value_cols = [col for col in filtered_df.columns if any(term in str(col).lower() for term in ['amount', 'value', 'revenue', 'sales', 'total'])]
                
                if value_cols:
                    for col in value_cols:
                        if filtered_df[col].dtype in ['int64', 'float64']:
                            total = filtered_df[col].sum()
                            analysis.append(f"")
                            analysis.append(f"Sum of {col}: {total:,.2f}")
                            
                            if 'revenue' in query_lower and len(filtered_df) > 0:
                                return f"Manual Analysis Result: {filters.get('entity', 'Entity')} {filters.get('year', 'Year')} revenue = ${total:,.2f}"
                
                return f"Manual Analysis: Found {len(filtered_df)} matching rows in {sheet_name}"
            else:
                analysis.append("No data matches the filters")
                return f"Manual Analysis: No matching data found in {sheet_name}"
                
        except Exception as e:
            analysis.append(f"Manual analysis error: {str(e)}")
            return f"Manual Analysis Error: {str(e)}"
        
        return "\n".join(analysis)

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
        print(f"🔍 Checking Excel file path: {excel_path}")
        if excel_path and os.path.exists(excel_path):
            tools['excel'] = excel_path
            print(f"✅ Excel file found: {excel_path}")
        else:
            print(f"❌ Excel file not found or EXCEL_FILE_PATH not set")
            print(f"   Current EXCEL_FILE_PATH: {excel_path}")
            print(f"   File exists: {os.path.exists(excel_path) if excel_path else 'N/A'}")
        
        # CSV tools
        csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                tools['csv'] = {'directory': csv_dir, 'files': csv_files}
        
        return tools
    
    def process_query(self, query: str) -> str:
        """Process query with intelligent multi-source approach"""
        print(f"🎯 Multi-Source Intelligent Agent processing: {query}")
        
        try:
            # PHASE 1: Multi-Source Discovery
            all_source_analyses = self.discovery_agent.discover_all_data_sources(self.tools, query)
            
            if not any(a.recommended for a in all_source_analyses):
                return "🔍 Discovery Phase: No relevant data sources found for this query"
            
            # PHASE 2: Multi-Source Analysis
            result = self.analysis_agent.analyze_selected_sources(all_source_analyses, query, self.discovery_agent)
            
            return result
                
        except Exception as e:
            return f"🎯 Multi-Source Agent error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status"""
        return """🎯 **Multi-Source Intelligent Agent with Cross-Validation & Transparency**

**System Type**: Multi-Source Discovery → Intelligent Selection → Cross-Validation
**Agent Model**: GPT-4o-mini (Discovery) + GPT-4o (Analysis & Validation)
**Strategy**: Explore ALL data sources, intelligently select optimal combination, cross-validate results

**Three-Phase Multi-Source Process**:
🔍 **Phase 1 - Multi-Source Discovery**:
   • Explores ALL available data sources (Excel sheets, CSV files, SQL, Email)
   • Analyzes query complexity (simple/moderate/complex) 
   • Scores each source using keyword matching + LLM reasoning
   • Uses GPT-4o-mini for fast discovery across all sources

🤖 **Phase 2 - Intelligent Source Selection**:
   • LLM-powered selection of optimal source combination
   • Considers complementarity (CSV + Excel for comprehensive coverage)
   • Adapts selection strategy based on query type
   • Prioritizes CSV for operational/production queries
   • Prioritizes Excel for financial queries

🎯 **Phase 3 - Multi-Source Analysis**:
   • Analyzes selected sources with full transparency
   • Shows exact pandas code and calculations
   • Provides step-by-step verification
   • Uses GPT-4o for detailed analysis

🔄 **Phase 4 - Cross-Source Validation**:
   • Compares results across different source types
   • Validates consistency and identifies discrepancies
   • Enhanced scoring based on source-query alignment
   • Detects summary vs detail sheet patterns
   • Applies enhanced scoring based on cross-check
   • Selects best result with reasoning

**Adaptive Selection Features**:
🧠 Query complexity analysis (simple/moderate/complex)
📊 Dynamic relevance thresholds
✂️ Natural score breakpoint detection
🎯 Optimal sheet count selection (2-6 sheets)
📋 Intelligent fallback mechanisms

**ReAct Cross-Checking Features**:
🔄 Multi-sheet result comparison
📊 Numerical consistency analysis
🏗️ Data relationship detection
🏆 Enhanced result scoring
✅ Best result selection with reasoning

**Calculation Transparency Features**:
🔍 Data structure preview with column types
🧮 Step-by-step calculation process
📊 Exact pandas code execution details
✅ Row count verification after each filter
🔍 Manual verification instructions
📋 Detailed calculation reports

**Performance Benefits**:
✅ Intelligent sheet selection (not hardcoded limits)
✅ Query-adaptive analysis depth
✅ Cross-sheet result validation
✅ Complete calculation transparency
✅ Easy debugging of incorrect results
✅ Concrete answers with verification
✅ ReAct loop for quality assurance

**Available Data Sources**: """ + f"{len(self.tools)} sources discovered"
    
    def get_available_commands(self) -> str:
        """Get available commands"""
        return """🎯 **Adaptive ReAct Agent with Cross-Checking & Calculation Transparency**

**How It Works**:
1. 🧠 **Query Analysis**: Determines complexity (simple/moderate/complex)
2. 🔍 **Adaptive Discovery**: Dynamically selects relevant sheets (2-6 sheets)
3. 📊 **Transparent Analysis**: Analyzes each sheet with full transparency
4. 🔄 **ReAct Cross-Check**: Compares and validates results across sheets
5. 🏆 **Best Result Selection**: Intelligently selects optimal answer

**Query Examples with Adaptive Processing**:

**Simple Query**: "What was revenue in 2023 for Ontario?"
  → Complexity: Simple (higher threshold, 2 sheets max)
  → Discovery: Finds 2 most relevant financial sheets
  → Analysis: Detailed analysis of each sheet
  → Cross-Check: Compares results for consistency
  → Result: "Ontario 2023 revenue: $4,200,000 (verified across 2 sheets)"

**Complex Query**: "Analyze all regional revenue trends and breakdowns"
  → Complexity: Complex (lower threshold, up to 6 sheets)
  → Discovery: Finds 4-6 relevant sheets (summary, detail, regional)
  → Analysis: Comprehensive analysis across multiple sheets
  → Cross-Check: Validates relationships between summary and detail sheets
  → Result: Complete analysis with cross-sheet validation

**Adaptive Selection Features**:
🧠 **Query Complexity Analysis**: Simple/Moderate/Complex classification
📊 **Dynamic Thresholds**: Adjusts relevance cutoffs based on query type
✂️ **Natural Breakpoints**: Finds optimal sheet count using score gaps
🎯 **Intelligent Limits**: 2 sheets (simple) to 6 sheets (complex)
📋 **Smart Fallback**: Always ensures minimum viable sheet selection

**ReAct Cross-Checking Features**:
🔄 **Multi-Sheet Comparison**: Compares numerical results across sheets
📊 **Consistency Analysis**: Detects if results are consistent (variance %)
🏗️ **Relationship Detection**: Identifies summary vs detail vs regional sheets
🏆 **Enhanced Scoring**: Bonus/penalty based on cross-check analysis
✅ **Best Result Logic**: Selects optimal result with reasoning

**Calculation Transparency Features**:
🔍 **Data Structure Preview**: Column types, sample data, unique values
🧮 **Step-by-Step Process**: Every calculation step explained
📊 **Pandas Code Display**: Exact code executed for filtering and calculations
✅ **Row Verification**: Shows how many rows match each filter
🔍 **Manual Verification**: Instructions to manually verify the result
📋 **Detailed Reports**: Complete calculation transparency

**Performance Features**:
🚀 **Adaptive Discovery**: Query-aware sheet selection
🎯 **Smart Selection**: No hardcoded limits, intelligent thresholds
📊 **Focused Analysis**: Only analyzes truly relevant sheets
🔄 **Quality Assurance**: ReAct loop validates results
✅ **Transparent Results**: Complete calculation and cross-check details
🔍 **Easy Debugging**: See exactly how values were calculated and validated

**Adaptive Process**:
1. **Analyzes** query complexity and sets selection strategy
2. **Scans** all available data sources with dynamic scoring
3. **Selects** optimal number of sheets based on relevance gaps
4. **Analyzes** each sheet with complete transparency
5. **Cross-checks** results for consistency and relationships
6. **Selects** best result using enhanced scoring
7. **Provides** verification instructions and detailed reports

**No More**:
❌ Fixed "top 2 sheets" limitation
❌ One-size-fits-all analysis approach
❌ Single sheet analysis without validation
❌ Mystery calculations without transparency
❌ Incorrect results without cross-checking

**Instead**:
✅ Adaptive sheet selection based on query needs
✅ Query-complexity-aware analysis depth
✅ Multi-sheet cross-validation
✅ Complete calculation transparency
✅ ReAct loop quality assurance
✅ Intelligent result selection with reasoning

**Debug Incorrect Results**:
The agent now provides complete transparency:
• Query complexity analysis and selection strategy
• Adaptive sheet selection reasoning
• Exact data structure and pandas code for each sheet
• Cross-sheet result comparison and consistency analysis
• Enhanced scoring logic and best result selection
• Manual verification instructions

Perfect for any query complexity - the agent adapts its approach automatically!
"""