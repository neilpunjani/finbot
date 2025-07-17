import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass

@dataclass
class FinancialInsight:
    finding: str
    confidence: float
    worksheets_used: List[str]
    calculation_formula: Optional[str] = None
    business_context: Optional[str] = None

class AgenticExcelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        excel_file_path = os.getenv("EXCEL_FILE_PATH")
        if not excel_file_path or not os.path.exists(excel_file_path):
            raise ValueError(f"Excel file not found: {excel_file_path}")
        
        self.excel_file_path = excel_file_path
        self.worksheets = {}
        self.worksheet_info = {}
        self.financial_insights = []
        
        # Load worksheet structure
        self._initialize_worksheets()
        
        # Financial knowledge base
        self.financial_formulas = {
            'profit_margin': '(Revenue - Costs) / Revenue * 100',
            'gross_margin': '(Revenue - COGS) / Revenue * 100',
            'operating_margin': '(Operating Income) / Revenue * 100',
            'net_margin': '(Net Income) / Revenue * 100',
            'roe': 'Net Income / Shareholders Equity * 100',
            'roa': 'Net Income / Total Assets * 100',
            'current_ratio': 'Current Assets / Current Liabilities',
            'debt_to_equity': 'Total Debt / Total Equity',
            'asset_turnover': 'Revenue / Average Total Assets',
            'inventory_turnover': 'COGS / Average Inventory',
            'receivables_turnover': 'Revenue / Average Accounts Receivable'
        }
        
    def _initialize_worksheets(self):
        """Initialize all worksheets with intelligent preprocessing"""
        try:
            xls = pd.ExcelFile(self.excel_file_path)
            
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
                    
                    # Intelligent preprocessing based on sheet type
                    processed_df = self._intelligent_preprocessing(df, sheet_name)
                    
                    self.worksheets[sheet_name] = processed_df
                    self.worksheet_info[sheet_name] = {
                        'columns': list(processed_df.columns),
                        'shape': processed_df.shape,
                        'data_types': processed_df.dtypes.to_dict(),
                        'sample_data': processed_df.head(3).to_dict('records'),
                        'sheet_type': self._classify_sheet_type(sheet_name, processed_df)
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not load worksheet '{sheet_name}': {e}")
                    
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")
    
    def _intelligent_preprocessing(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Intelligent preprocessing based on sheet type and content"""
        df = df.copy()
        
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle different sheet types
        if sheet_name == 'VW_PBI':
            df = self._preprocess_vw_pbi(df)
        elif sheet_name == 'TB':
            df = self._preprocess_trial_balance(df)
        elif sheet_name == 'AR':
            df = self._preprocess_accounts_receivable(df)
        elif sheet_name == 'Debt Schedule':
            df = self._preprocess_debt_schedule(df)
        elif sheet_name.startswith('By '):
            df = self._preprocess_summary_sheet(df)
        
        # Smart data type conversion
        df = self._smart_data_conversion(df)
        
        # Create computed financial columns
        df = self._create_financial_columns(df, sheet_name)
        
        return df
    
    def _preprocess_vw_pbi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced VW_PBI preprocessing with financial intelligence"""
        
        # Standardize column names and data types
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        if 'Period' in df.columns:
            df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Create financial categorization
        if 'Level1' in df.columns:
            df['Statement_Type'] = df['Level1'].map({
                'BS': 'Balance Sheet',
                'PnL': 'Profit and Loss',
                'PL': 'Profit and Loss',
                'CF': 'Cash Flow'
            }).fillna(df['Level1'])
        
        # Enhanced revenue/cost identification
        if 'Level2' in df.columns:
            df['Account_Category'] = df['Level2'].apply(self._categorize_account)
            df['Is_Revenue'] = df['Level2'].str.contains('Revenue|Income|Sales', case=False, na=False)
            df['Is_Cost'] = df['Level2'].str.contains('Cost|Expense|COGS', case=False, na=False)
            df['Is_Asset'] = df['Level2'].str.contains('Asset|Cash|Inventory|Receivable', case=False, na=False)
            df['Is_Liability'] = df['Level2'].str.contains('Liability|Payable|Debt', case=False, na=False)
        
        # Create time-based analysis columns
        if 'Year' in df.columns and 'Period' in df.columns:
            df['Year_Period'] = df['Year'].astype(str) + '-' + df['Period'].astype(str).str.zfill(2)
            df['Quarter'] = ((df['Period'] - 1) // 3 + 1).astype(int)
            df['Year_Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
        
        return df
    
    def _categorize_account(self, account_name: str) -> str:
        """Categorize account based on name patterns"""
        if pd.isna(account_name):
            return 'Unknown'
        
        account_lower = str(account_name).lower()
        
        # Revenue categories
        if any(term in account_lower for term in ['revenue', 'sales', 'income', 'fees']):
            return 'Revenue'
        # Cost categories
        elif any(term in account_lower for term in ['cost', 'expense', 'cogs', 'depreciation']):
            return 'Cost'
        # Asset categories
        elif any(term in account_lower for term in ['asset', 'cash', 'inventory', 'receivable', 'equipment']):
            return 'Asset'
        # Liability categories
        elif any(term in account_lower for term in ['liability', 'payable', 'debt', 'loan']):
            return 'Liability'
        # Equity categories
        elif any(term in account_lower for term in ['equity', 'capital', 'retained']):
            return 'Equity'
        else:
            return 'Other'
    
    def _preprocess_trial_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced trial balance preprocessing"""
        # Similar to VW_PBI but focused on trial balance structure
        return self._preprocess_vw_pbi(df)  # Reuse VW_PBI logic
    
    def _preprocess_accounts_receivable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced AR preprocessing with aging analysis"""
        
        # Date processing
        date_columns = ['InvoiceDate', 'PaymentDate', 'LastRefresh']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Aging analysis
        if 'InvoiceDate' in df.columns:
            df['Days_Outstanding'] = (pd.Timestamp.now() - df['InvoiceDate']).dt.days
            df['Aging_Category'] = pd.cut(df['Days_Outstanding'], 
                                        bins=[0, 30, 60, 90, 365, float('inf')], 
                                        labels=['0-30 days', '31-60 days', '61-90 days', '91-365 days', '>365 days'])
        
        # Risk assessment
        if 'InvAmount' in df.columns and 'Days_Outstanding' in df.columns:
            df['Risk_Score'] = (df['Days_Outstanding'] / 365) * (df['InvAmount'] / df['InvAmount'].max())
        
        return df
    
    def _preprocess_debt_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced debt schedule preprocessing"""
        
        # Date processing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Debt ratios and analysis
        debt_columns = ['Beginning Debt Balance', 'Ending Debt Balance', 'Principal Amount', 'Interest Amount']
        for col in debt_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Interest rate analysis
        if 'Interest Amount' in df.columns and 'Beginning Debt Balance' in df.columns:
            df['Effective_Interest_Rate'] = (df['Interest Amount'] / df['Beginning Debt Balance']) * 100
        
        return df
    
    def _preprocess_summary_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced summary sheet preprocessing"""
        
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Smart numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_series
        
        return df
    
    def _smart_data_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart data type conversion based on content analysis"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric conversion for financial data
                if any(keyword in col.lower() for keyword in ['amount', 'value', 'balance', 'total', 'sum']):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Try date conversion
                elif any(keyword in col.lower() for keyword in ['date', 'time']):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _create_financial_columns(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Create computed financial columns based on available data"""
        
        # Revenue and cost aggregations
        if 'Amount' in df.columns and 'Is_Revenue' in df.columns:
            # These will be used for aggregations, not row-level calculations
            pass
        
        # Financial ratios (only if we have the right data structure)
        if sheet_name == 'VW_PBI' and all(col in df.columns for col in ['Amount', 'Account_Category', 'Entity']):
            # Create pivot for easier financial calculations
            try:
                pivot_df = df.pivot_table(
                    values='Amount', 
                    index=['Entity', 'Year'], 
                    columns='Account_Category', 
                    aggfunc='sum', 
                    fill_value=0
                )
                
                # Store pivot for later use
                self.financial_pivots = pivot_df
                
            except Exception as e:
                pass  # If pivot fails, continue without it
        
        return df
    
    def _classify_sheet_type(self, sheet_name: str, df: pd.DataFrame) -> str:
        """Classify the type of financial sheet"""
        
        sheet_lower = sheet_name.lower()
        columns_lower = [col.lower() for col in df.columns]
        
        if 'vw_pbi' in sheet_lower or 'pbi' in sheet_lower:
            return 'detailed_financial'
        elif 'tb' in sheet_lower or 'trial' in sheet_lower:
            return 'trial_balance'
        elif 'ar' in sheet_lower or 'receivable' in sheet_lower:
            return 'accounts_receivable'
        elif 'debt' in sheet_lower or 'loan' in sheet_lower:
            return 'debt_schedule'
        elif sheet_name.startswith('By '):
            return 'summary_report'
        elif any(term in ' '.join(columns_lower) for term in ['revenue', 'income', 'expense', 'cost']):
            return 'financial_statement'
        else:
            return 'general'
    
    def query(self, question: str) -> str:
        """Main agentic query processing with financial reasoning"""
        
        # Step 1: Analyze financial query intent
        query_analysis = self._analyze_financial_query(question)
        
        # Step 2: Determine relevant worksheets
        relevant_sheets = self._determine_relevant_worksheets(question, query_analysis)
        
        # Step 3: Financial reasoning loop
        insights = []
        for sheet_name in relevant_sheets:
            sheet_insights = self._financial_reasoning(sheet_name, question, query_analysis)
            insights.extend(sheet_insights)
        
        # Step 4: Cross-worksheet financial analysis
        if len(relevant_sheets) > 1:
            cross_insights = self._cross_worksheet_analysis(insights, question, query_analysis)
            insights.extend(cross_insights)
        
        # Step 5: Generate final financial response
        return self._generate_financial_response(question, insights, query_analysis)
    
    def _analyze_financial_query(self, question: str) -> Dict[str, Any]:
        """Analyze financial query intent with business context"""
        
        financial_prompt = PromptTemplate.from_template("""
        Analyze this financial query to understand the business intent and required calculations.
        
        QUERY: {question}
        
        AVAILABLE FINANCIAL FORMULAS:
        {formulas}
        
        Extract:
        1. Financial concept requested (profitability, liquidity, efficiency, etc.)
        2. Specific metrics or ratios needed
        3. Time period or entity filters
        4. Calculation methodology required
        5. Expected business insights
        6. Data sources likely needed
        
        Respond with JSON:
        {{
            "financial_concept": "main concept",
            "metrics_needed": ["list of metrics"],
            "time_filters": ["time periods"],
            "entity_filters": ["entities"],
            "calculation_method": "methodology",
            "business_context": "business interpretation needed",
            "complexity": "simple|medium|complex",
            "formula_needed": "specific formula if applicable"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=financial_prompt.format(
                question=question,
                formulas=json.dumps(self.financial_formulas, indent=2)
            ))
        ])
        
        try:
            return json.loads(response.content.strip())
        except:
            return {
                "financial_concept": "general_analysis",
                "metrics_needed": [],
                "time_filters": [],
                "entity_filters": [],
                "calculation_method": "summary",
                "business_context": "financial_performance",
                "complexity": "medium",
                "formula_needed": None
            }
    
    def _determine_relevant_worksheets(self, question: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Determine relevant worksheets based on financial query analysis"""
        
        sheet_scores = {}
        
        for sheet_name, sheet_info in self.worksheet_info.items():
            score = 0
            
            # Score based on sheet type relevance
            sheet_type = sheet_info['sheet_type']
            if query_analysis.get('financial_concept') == 'profitability':
                if sheet_type in ['detailed_financial', 'trial_balance']:
                    score += 20
            elif query_analysis.get('financial_concept') == 'liquidity':
                if sheet_type in ['detailed_financial', 'accounts_receivable']:
                    score += 20
            elif query_analysis.get('financial_concept') == 'leverage':
                if sheet_type in ['detailed_financial', 'debt_schedule']:
                    score += 20
            
            # Score based on metrics needed
            for metric in query_analysis.get('metrics_needed', []):
                metric_lower = metric.lower()
                for col in sheet_info['columns']:
                    if metric_lower in col.lower():
                        score += 15
            
            # Score based on column relevance
            for col in sheet_info['columns']:
                col_lower = col.lower()
                if any(word in col_lower for word in question.lower().split()):
                    score += 5
            
            sheet_scores[sheet_name] = score
        
        # Return sheets with score > 0, sorted by relevance
        relevant = [name for name, score in sheet_scores.items() if score > 0]
        if not relevant:
            # Fallback to VW_PBI if no specific matches
            relevant = ['VW_PBI'] if 'VW_PBI' in self.worksheets else list(self.worksheets.keys())[:1]
        
        return sorted(relevant, key=lambda x: sheet_scores.get(x, 0), reverse=True)
    
    def _financial_reasoning(self, sheet_name: str, question: str, query_analysis: Dict[str, Any]) -> List[FinancialInsight]:
        """Perform financial reasoning on a specific worksheet"""
        
        df = self.worksheets[sheet_name]
        sheet_info = self.worksheet_info[sheet_name]
        
        reasoning_prompt = PromptTemplate.from_template("""
        You are a financial analyst reasoning through data to answer a business question.
        
        QUESTION: {question}
        FINANCIAL ANALYSIS: {query_analysis}
        
        WORKSHEET: {sheet_name}
        SHEET TYPE: {sheet_type}
        COLUMNS: {columns}
        SAMPLE DATA: {sample_data}
        
        Financial reasoning steps:
        1. What financial concept is being asked?
        2. Which columns contain the necessary financial data?
        3. What calculations or aggregations are needed?
        4. Are there any business rules or financial principles to consider?
        5. What's the best approach to get accurate financial insights?
        
        Generate pandas code to perform the financial analysis.
        
        Respond with JSON:
        {{
            "financial_reasoning": "step-by-step business reasoning",
            "data_columns": ["relevant columns"],
            "pandas_code": "executable pandas code using 'df'",
            "business_interpretation": "what the results mean",
            "confidence": "high|medium|low",
            "formula_used": "financial formula if applicable"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=reasoning_prompt.format(
                question=question,
                query_analysis=str(query_analysis),
                sheet_name=sheet_name,
                sheet_type=sheet_info['sheet_type'],
                columns=sheet_info['columns'],
                sample_data=str(sheet_info['sample_data'])
            ))
        ])
        
        try:
            reasoning_result = json.loads(response.content.strip())
            
            # Execute pandas code
            pandas_code = reasoning_result.get('pandas_code', '')
            if pandas_code:
                local_vars = {'df': df, 'pd': pd}
                try:
                    exec(pandas_code, {"__builtins__": {}}, local_vars)
                    
                    if 'result' in local_vars:
                        result = local_vars['result']
                        
                        return [FinancialInsight(
                            finding=f"{reasoning_result.get('business_interpretation', '')}: {str(result)}",
                            confidence=0.9 if reasoning_result.get('confidence') == 'high' else 0.7,
                            worksheets_used=[sheet_name],
                            calculation_formula=reasoning_result.get('formula_used'),
                            business_context=reasoning_result.get('financial_reasoning')
                        )]
                        
                except Exception as e:
                    return self._fallback_financial_analysis(df, question, sheet_name)
            
        except Exception as e:
            return self._fallback_financial_analysis(df, question, sheet_name)
        
        return []
    
    def _fallback_financial_analysis(self, df: pd.DataFrame, question: str, sheet_name: str) -> List[FinancialInsight]:
        """Fallback financial analysis"""
        
        insights = []
        
        # Basic financial aggregations
        if 'Amount' in df.columns:
            total_amount = df['Amount'].sum()
            insights.append(FinancialInsight(
                finding=f"Total amount in {sheet_name}: ${total_amount:,.2f}",
                confidence=0.6,
                worksheets_used=[sheet_name]
            ))
        
        return insights
    
    def _cross_worksheet_analysis(self, insights: List[FinancialInsight], question: str, query_analysis: Dict[str, Any]) -> List[FinancialInsight]:
        """Perform cross-worksheet financial analysis"""
        
        if len(insights) < 2:
            return []
        
        cross_analysis_prompt = PromptTemplate.from_template("""
        Perform advanced financial analysis by combining insights from multiple worksheets.
        
        QUESTION: {question}
        FINANCIAL CONTEXT: {query_analysis}
        
        INDIVIDUAL INSIGHTS:
        {insights}
        
        Cross-worksheet analysis:
        1. Can you calculate financial ratios using data from multiple sheets?
        2. Are there complementary metrics that provide deeper insights?
        3. What business conclusions can be drawn from the combined data?
        4. Are there any financial relationships or trends across sheets?
        
        Provide enhanced financial insights that go beyond individual worksheet analysis.
        """)
        
        insights_text = "\n".join([
            f"• {insight.finding} (from {', '.join(insight.worksheets_used)})"
            for insight in insights
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=cross_analysis_prompt.format(
                question=question,
                query_analysis=str(query_analysis),
                insights=insights_text
            ))
        ])
        
        cross_insight = FinancialInsight(
            finding=response.content.strip(),
            confidence=0.8,
            worksheets_used=["cross_analysis"],
            business_context="integrated_financial_analysis"
        )
        
        return [cross_insight]
    
    def _generate_financial_response(self, question: str, insights: List[FinancialInsight], query_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive financial response"""
        
        # Calculate overall confidence
        if insights:
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
        else:
            avg_confidence = 0.3
        
        response_prompt = PromptTemplate.from_template("""
        Generate a comprehensive financial analysis response.
        
        QUESTION: {question}
        FINANCIAL CONTEXT: {query_analysis}
        
        INSIGHTS:
        {insights}
        
        OVERALL CONFIDENCE: {confidence}
        
        Create a professional financial analysis that:
        1. Directly answers the financial question with specific numbers
        2. Provides business context and interpretation
        3. Shows calculations and formulas used
        4. Explains the financial implications
        5. Indicates confidence level and any limitations
        6. Suggests additional analysis if relevant
        
        Format as a professional financial report with clear metrics and insights.
        """)
        
        insights_text = "\n".join([
            f"• {insight.finding}\n  Formula: {insight.calculation_formula or 'N/A'}\n  Context: {insight.business_context or 'N/A'}\n  Confidence: {insight.confidence:.1f}"
            for insight in insights
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=response_prompt.format(
                question=question,
                query_analysis=str(query_analysis),
                insights=insights_text,
                confidence=avg_confidence
            ))
        ])
        
        final_response = response.content.strip()
        
        # Add confidence indicator
        if avg_confidence >= 0.8:
            confidence_text = "🟢 High confidence"
        elif avg_confidence >= 0.6:
            confidence_text = "🟡 Medium confidence"
        else:
            confidence_text = "🔴 Low confidence - may need additional data"
        
        return f"{final_response}\n\n**Financial Analysis Confidence: {confidence_text}**"
    
    def get_data_info(self) -> str:
        """Get information about available Excel worksheets"""
        info = f"Available Excel worksheets: {len(self.worksheets)}\n\n"
        
        for sheet_name, sheet_info in self.worksheet_info.items():
            info += f"📊 {sheet_name}:\n"
            info += f"  • Type: {sheet_info['sheet_type']}\n"
            info += f"  • Shape: {sheet_info['shape'][0]:,} rows × {sheet_info['shape'][1]} columns\n"
            info += f"  • Key columns: {', '.join(sheet_info['columns'][:5])}{'...' if len(sheet_info['columns']) > 5 else ''}\n\n"
        
        return info