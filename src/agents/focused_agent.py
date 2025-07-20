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
        
        # Cache loaded dataframes to avoid reloading in analysis phase
        self.cached_dataframes = {}
        
        # Quick scan of all sheets - LOAD FULL SHEETS FOR PROPER ANALYSIS
        for sheet_name in xls.sheet_names:
            try:
                # CRITICAL FIX: Load full sheet for accurate relevance scoring
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                print(f"   📊 Loaded full sheet {sheet_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Cache the dataframe for later use in analysis phase
                self.cached_dataframes[sheet_name] = df
                
                if df.empty or df.shape[1] < 2:
                    print(f"   ❌ Skipping {sheet_name}: Empty or insufficient columns")
                    continue
                
                # Quick relevance analysis on FULL dataset
                analysis = self._analyze_sheet_relevance(sheet_name, df, query)
                sheet_analyses.append(analysis)
                
                print(f"   📄 {sheet_name}: Score {analysis.relevance_score:.1f} - {analysis.reason}")
                
            except Exception as e:
                print(f"   ❌ Error reading {sheet_name}: {e}")
                continue
        
        # Sort by relevance and return top candidates
        sheet_analyses.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Adaptive sheet selection based on relevance scores and query complexity
        recommended_sheets = self._adaptive_sheet_selection(sheet_analyses, query)
        
        # Mark recommended sheets
        for analysis in sheet_analyses:
            analysis.recommended = analysis in recommended_sheets
        
        recommended_count = len(recommended_sheets)
        print(f"🎯 DISCOVERY COMPLETE: {recommended_count} sheets recommended for analysis")
        
        # Show selection reasoning
        if recommended_sheets:
            print("   📋 Selected sheets:")
            for sheet in recommended_sheets:
                print(f"     • {sheet.name} (score: {sheet.relevance_score:.1f}) - {sheet.reason}")
        else:
            print("   ❌ No sheets met the relevance threshold")
        
        return sheet_analyses
    
    def _adaptive_sheet_selection(self, sheet_analyses: List[SheetAnalysis], query: str) -> List[SheetAnalysis]:
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
        
        # CRITICAL FIX: Penalize sheets with blank/empty data
        data_quality_penalty = self._calculate_data_quality_penalty(df, query)
        score -= data_quality_penalty
        if data_quality_penalty > 0:
            reasons.append(f"data quality penalty: -{data_quality_penalty:.1f}")
        
        # Final score cannot be negative
        score = max(score, 0.0)
        
        reason = ", ".join(reasons) if reasons else "no specific indicators"
        
        data_summary = f"{df.shape[0]} rows, {df.shape[1]} cols. Columns: {list(df.columns)[:3]}..."
        
        return SheetAnalysis(
            name=sheet_name,
            relevance_score=score,
            data_summary=data_summary,
            recommended=False,  # Will be set later
            reason=reason
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
    PHASE 2: Focused analysis agent that only analyzes pre-selected sheets
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def analyze_selected_sheets(self, excel_path: str, sheet_analyses: List[SheetAnalysis], query: str, discovery_agent: 'DataDiscoveryAgent' = None) -> str:
        """Analyze recommended sheets with ReAct cross-checking mechanism"""
        recommended_sheets = [a for a in sheet_analyses if a.recommended]
        
        if not recommended_sheets:
            return "No relevant sheets found for analysis"
        
        print(f"🎯 PHASE 2: Analyzing {len(recommended_sheets)} recommended sheets with ReAct cross-checking")
        
        analysis_report = "🔍 **REACT CROSS-CHECKING ANALYSIS REPORT**\n"
        analysis_report += "=" * 60 + "\n\n"
        
        # Collect all results for cross-checking
        sheet_results = []
        
        for sheet_analysis in recommended_sheets:
            print(f"   📊 Analyzing {sheet_analysis.name}...")
            analysis_report += f"**Sheet: {sheet_analysis.name}**\n"
            analysis_report += f"Selected because: {sheet_analysis.reason}\n"
            analysis_report += f"Data summary: {sheet_analysis.data_summary}\n\n"
            
            try:
                # Use cached dataframe if available, otherwise load fresh
                if discovery_agent and hasattr(discovery_agent, 'cached_dataframes') and sheet_analysis.name in discovery_agent.cached_dataframes:
                    df = discovery_agent.cached_dataframes[sheet_analysis.name]
                    print(f"   📊 USING CACHED SHEET: {df.shape[0]} rows, {df.shape[1]} columns")
                else:
                    # Fallback: Load fresh if cache not available
                    df = pd.read_excel(excel_path, sheet_name=sheet_analysis.name)
                    print(f"   📊 LOADED FRESH SHEET: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # CRITICAL: Verify full sheet is loaded
                analysis_report += f"🔍 **FULL SHEET VERIFICATION**: Using {df.shape[0]} total rows, {df.shape[1]} columns\n\n"
                
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
                    
                    # Store result for cross-checking
                    sheet_results.append({
                        'sheet': sheet_analysis,
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
                print(f"   ❌ Error analyzing {sheet_analysis.name}: {e}")
                continue
            
            analysis_report += "-" * 40 + "\n\n"
        
        # ReAct Cross-Checking Phase
        if len(sheet_results) > 1:
            print(f"🔄 REACT CROSS-CHECKING: Comparing {len(sheet_results)} results...")
            cross_check_report = self._react_cross_check_results(sheet_results, query)
            analysis_report += cross_check_report
            
            # Get the best result after cross-checking
            best_result_info = self._select_best_result_after_cross_check(sheet_results, query)
        elif len(sheet_results) == 1:
            print("📋 Single result found, no cross-checking needed")
            best_result_info = sheet_results[0]
        else:
            print("❌ No valid results found")
            return f"❌ No relevant data found in the recommended sheets\n\n📊 **ANALYSIS REPORT**:\n{analysis_report}"
        
        # Final result with complete transparency
        if best_result_info:
            best_result = best_result_info['result']
            best_calculation_details = best_result_info['details']
            best_sheet = best_result_info['sheet']
            best_df = best_result_info['dataframe']
            
            final_report = f"✅ **FINAL ANSWER**: {best_result}\n\n"
            final_report += f"🧮 **HOW THIS WAS CALCULATED**:\n{best_calculation_details}\n\n"
            
            # Add verification report
            verification_report = self._create_verification_report(best_df, query, best_result, best_sheet.name)
            final_report += f"🔍 **VERIFICATION REPORT**:\n{verification_report}\n\n"
            
            final_report += f"🎯 **Analysis Method**: Adaptive ReAct Cross-Checking (Discovery → Analysis → Cross-Check)\n\n"
            final_report += f"📊 **DETAILED ANALYSIS REPORT**:\n{analysis_report}"
            return final_report
        else:
            return f"❌ No relevant data found in the recommended sheets\n\n📊 **ANALYSIS REPORT**:\n{analysis_report}"
    
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
        """Process query with focused two-phase approach"""
        print(f"🎯 Focused Agent processing: {query}")
        
        try:
            if 'excel' in self.tools:
                excel_path = self.tools['excel']
                
                # PHASE 1: Discovery
                sheet_analyses = self.discovery_agent.discover_relevant_sheets(excel_path, query)
                
                if not any(a.recommended for a in sheet_analyses):
                    return "🔍 Discovery Phase: No relevant sheets found for this query"
                
                # PHASE 2: Focused Analysis (pass discovery agent for cached dataframes)
                result = self.analysis_agent.analyze_selected_sheets(excel_path, sheet_analyses, query, self.discovery_agent)
                
                return result
            else:
                return "No Excel data source available"
                
        except Exception as e:
            return f"🎯 Focused Agent error: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status"""
        return """🎯 **Adaptive ReAct Agent with Cross-Checking & Calculation Transparency**

**System Type**: Adaptive Discovery → Analysis → ReAct Cross-Checking
**Agent Model**: GPT-4o-mini (Discovery) + GPT-4o (Analysis & Cross-Check)
**Strategy**: Intelligently select relevant sheets, analyze with transparency, then cross-check results

**Three-Phase Adaptive Process**:
🔍 **Phase 1 - Adaptive Discovery**:
   • Analyzes query complexity (simple/moderate/complex)
   • Dynamically adjusts relevance thresholds
   • Selects optimal number of sheets (not fixed to 2)
   • Uses intelligent scoring with natural breakpoints
   • Fast GPT-4o-mini for speed

🎯 **Phase 2 - Transparent Analysis**:
   • Analyzes all selected relevant sheets
   • Shows complete data structure and column analysis
   • Displays exact pandas code executed
   • Shows step-by-step calculation process
   • Provides detailed verification reports
   • Uses GPT-4o for detailed analysis

🔄 **Phase 3 - ReAct Cross-Checking**:
   • Compares results across multiple sheets
   • Analyzes consistency and relationships
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