import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from typing import Dict, List, Any
import json

class ExcelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        excel_file_path = os.getenv("EXCEL_FILE_PATH")
        if not excel_file_path or not os.path.exists(excel_file_path):
            raise ValueError(f"Excel file not found: {excel_file_path}")
        
        # Store path and lazy load worksheets
        self.excel_file_path = excel_file_path
        self.worksheets = {}  # Lazy-loaded worksheets
        self.worksheet_info = self._get_worksheet_info()
        
        # Don't create agents immediately - lazy load them when needed
        self.worksheet_agents = {}
    
    def _get_worksheet_info(self) -> Dict[str, Any]:
        """Get basic worksheet info without loading all data"""
        try:
            # Read Excel file structure only (without data)
            xls = pd.ExcelFile(self.excel_file_path)
            sheet_names = xls.sheet_names
            
            return {
                'sheet_names': sheet_names,
                'total_worksheets': len(sheet_names),
                'worksheets': {name: {'loaded': False} for name in sheet_names},
                'common_columns': {}  # Will be populated as sheets are loaded
            }
        except Exception as e:
            raise ValueError(f"Could not read Excel file structure: {e}")
    
    def _load_worksheet(self, sheet_name: str):
        """Load a specific worksheet on demand with comprehensive preprocessing"""
        if sheet_name not in self.worksheets:
            try:
                df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
                
                # Apply comprehensive preprocessing
                df = self._preprocess_excel_dataframe(df, sheet_name)
                
                self.worksheets[sheet_name] = df
                # Update worksheet info
                self.worksheet_info['worksheets'][sheet_name] = {
                    'loaded': True,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict()
                }
            except Exception as e:
                print(f"Warning: Could not load worksheet '{sheet_name}': {e}")
                return None
        return self.worksheets.get(sheet_name)
    
    def _preprocess_excel_dataframe(self, df, sheet_name):
        """Comprehensive preprocessing for Excel dataframes"""
        df = df.copy()
        
        # Handle different worksheet types
        if sheet_name == 'VW_PBI':
            df = self._preprocess_vw_pbi(df)
        elif sheet_name == 'TB':
            df = self._preprocess_tb(df)
        elif sheet_name == 'AR':
            df = self._preprocess_ar(df)
        elif sheet_name == 'Debt Schedule':
            df = self._preprocess_debt_schedule(df)
        elif sheet_name.startswith('By '):
            df = self._preprocess_summary_sheet(df, sheet_name)
        
        return df
    
    def _preprocess_vw_pbi(self, df):
        """Preprocessing specific to VW_PBI worksheet"""
        # Date and time processing
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        if 'Period' in df.columns:
            df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
        
        # Standardize text columns
        text_columns = ['Entity', 'Office', 'Scenario', 'Currency', 'Measure', 'Level1', 'Level2', 'Level3', 'Commodity']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Standardize amount column
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Create computed columns for common queries
        if 'Level1' in df.columns:
            df['Financial_Category'] = df['Level1'].map({
                'BS': 'Balance Sheet',
                'PnL': 'Profit and Loss',
                'PL': 'Profit and Loss'
            }).fillna(df['Level1'])
        
        # Create revenue indicators
        if 'Level2' in df.columns:
            df['Is_Revenue'] = df['Level2'].str.contains('Revenue|Income|Sales', case=False, na=False)
        
        return df
    
    def _preprocess_tb(self, df):
        """Preprocessing specific to TB (Trial Balance) worksheet"""
        # Similar processing to VW_PBI but focused on trial balance
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        if 'Period' in df.columns:
            df['Period'] = pd.to_numeric(df['Period'], errors='coerce')
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Standardize text columns
        text_columns = ['Entity', 'Office', 'Scenario', 'Currency', 'Measure']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _preprocess_ar(self, df):
        """Preprocessing specific to AR (Accounts Receivable) worksheet"""
        # Date processing
        date_columns = ['InvoiceDate', 'PaymentDate', 'LastRefresh']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Numeric processing
        numeric_columns = ['InvAmount', 'DaysOpen', 'ProjectNumber', 'Phase', 'Task', 'Period']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Text standardization
        text_columns = ['BillingClientName', 'PrimaryClientName']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _preprocess_debt_schedule(self, df):
        """Preprocessing specific to Debt Schedule worksheet"""
        # Date processing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Numeric processing for financial columns
        numeric_columns = ['Beginning Debt Balance', 'Ending Debt Balance', 'Principal Amount', 
                          'Interest Amount', 'Expected SOFR2', 'Expected SOFR1', 'Loan Interest Rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Text standardization
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str).str.strip()
        
        return df
    
    def _preprocess_summary_sheet(self, df, sheet_name):
        """Preprocessing for summary sheets (By Entity, By Office, etc.)"""
        # These sheets often have irregular structure, try to clean them up
        
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Try to identify header rows and data rows
        if df.shape[0] > 1:
            # Look for row that contains "Row Labels" or similar
            header_row = None
            for idx, row in df.iterrows():
                if any('label' in str(val).lower() for val in row if pd.notna(val)):
                    header_row = idx
                    break
            
            if header_row is not None:
                # Use the row after header_row as column names if it makes sense
                if header_row + 1 < len(df):
                    new_columns = df.iloc[header_row].fillna('').astype(str).tolist()
                    df.columns = new_columns
                    df = df.iloc[header_row + 1:].reset_index(drop=True)
        
        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # If more than 50% of values are numeric, convert the column
                if numeric_series.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_series
        
        return df
    
    def _analyze_worksheets(self) -> Dict[str, Dict[str, Any]]:
        """Analyze structure and relationships across worksheets"""
        analysis = {}
        
        for sheet_name, df in self.worksheets.items():
            if df.empty:
                continue
                
            analysis[sheet_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records'),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'date_columns': df.select_dtypes(include=['datetime']).columns.tolist()
            }
        
        # Find common columns across worksheets
        all_columns = set()
        for info in analysis.values():
            all_columns.update(info['columns'])
        
        common_columns = {}
        for col in all_columns:
            sheets_with_col = [sheet for sheet, info in analysis.items() if col in info['columns']]
            if len(sheets_with_col) > 1:
                common_columns[col] = sheets_with_col
        
        return {
            'worksheets': analysis,
            'common_columns': common_columns,
            'total_worksheets': len(analysis)
        }
    
    def _get_or_create_worksheet_agent(self, sheet_name: str):
        """Lazily create and return agent for the specified worksheet"""
        if sheet_name not in self.worksheet_agents:
            # Load worksheet if not already loaded
            df = self._load_worksheet(sheet_name)
            if df is not None and not df.empty:
                self.worksheet_agents[sheet_name] = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=df,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True
                )
        return self.worksheet_agents.get(sheet_name)
    
    def _determine_relevant_worksheets(self, query: str) -> List[str]:
        """Determine which worksheets are relevant for the query using keyword matching and LLM fallback"""
        query_lower = query.lower()
        
        # First try keyword-based matching
        keyword_matches = []
        
        # Define keyword mappings for worksheet names
        worksheet_keywords = {
            'TB': ['trial balance', 'tb', 'balance sheet'],
            'VW_PBI': ['vw_pbi', 'pbi', 'power bi', 'main data', 'detailed', 'breakdown', 'transactions', 'revenue breakdown', 'detailed revenue', 'commodity breakdown'],
            'AR': ['accounts receivable', 'ar', 'receivables', 'aging'],
            'Debt Schedule': ['debt', 'debt schedule', 'loans', 'borrowing'],
            'COA': ['chart of accounts', 'coa', 'accounts', 'account structure'],
            'By Entity': ['entity summary', 'entities summary', 'total by entity', 'summary by entity'],
            'By Office': ['office summary', 'offices summary', 'total by office'],
            'By Project': ['project summary', 'projects summary', 'total by project'],
            'By Vendor': ['vendor summary', 'vendors summary', 'supplier summary'],
            'By LOB': ['lob', 'line of business'],
            'By COA Group': ['coa group', 'account group'],
            'Arora Hierarchy': ['hierarchy', 'arora']
        }
        
        # Check for keyword matches
        for sheet_name, keywords in worksheet_keywords.items():
            if sheet_name in self.worksheet_info['sheet_names']:
                for keyword in keywords:
                    if keyword in query_lower:
                        keyword_matches.append(sheet_name)
                        break
        
        if keyword_matches:
            # If we have matches, check for ambiguous cases
            if len(keyword_matches) == 1:
                return keyword_matches
            elif 'By Entity' in keyword_matches and 'VW_PBI' in keyword_matches:
                # If both match, prefer VW_PBI for detailed queries unless explicitly asking for summary
                if any(word in query_lower for word in ['summary', 'total', 'aggregate']):
                    return ['By Entity']
                else:
                    return ['VW_PBI']
            else:
                return keyword_matches
        
        # If no keyword matches but query mentions entity/revenue, default to VW_PBI for detailed data
        if any(word in query_lower for word in ['entity', 'revenue', 'amount']) and not any(word in query_lower for word in ['summary', 'total']):
            return ['VW_PBI']
        
        # If no keyword matches, try LLM
        prompt = f"""
        Given the following Excel worksheets and a user query, determine which worksheets are most relevant.
        
        Available worksheets: {', '.join(self.worksheet_info['sheet_names'])}
        
        User query: {query}
        
        Return ONLY the exact sheet names that are relevant, separated by commas. 
        For example: "TB, VW_PBI" or "AR" or "Debt Schedule"
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            relevant_sheets = [sheet.strip().strip("'\"") for sheet in response.content.split(',')]
            
            # Filter to only existing sheets
            valid_sheets = [sheet for sheet in relevant_sheets if sheet in self.worksheet_info['sheet_names']]
            
            # If LLM didn't return valid sheets, fallback to first sheet
            if not valid_sheets:
                return [self.worksheet_info['sheet_names'][0]]
            
            return valid_sheets
            
        except Exception as e:
            # Ultimate fallback: return the first worksheet
            return [self.worksheet_info['sheet_names'][0]]
    
    def _create_cross_sheet_analysis(self, query: str, relevant_sheets: List[str]) -> str:
        """Create a combined analysis across multiple worksheets"""
        if len(relevant_sheets) == 1:
            agent = self._get_or_create_worksheet_agent(relevant_sheets[0])
            if agent:
                result = agent.invoke({"input": query})
                return result['output']
            else:
                return f"Could not create agent for worksheet '{relevant_sheets[0]}'"
        
        # For multiple sheets, create a comprehensive analysis
        combined_data = {}
        sheet_summaries = {}
        
        for sheet_name in relevant_sheets:
            df = self._load_worksheet(sheet_name)
            if df is not None:
                combined_data[sheet_name] = df
            
            # Get basic summary from individual sheet agent
            try:
                agent = self._get_or_create_worksheet_agent(sheet_name)
                if agent:
                    summary_result = agent.invoke({"input": f"Provide a brief summary of this data relevant to: {query}"})
                    summary = summary_result['output']
                    sheet_summaries[sheet_name] = summary
                else:
                    sheet_summaries[sheet_name] = f"Data from {sheet_name} with {df.shape[0]} rows and columns: {list(df.columns)}"
            except:
                sheet_summaries[sheet_name] = f"Data from {sheet_name} with {df.shape[0]} rows and columns: {list(df.columns)}"
        
        # Create a cross-sheet analysis prompt
        analysis_prompt = f"""
        I need to analyze data across multiple Excel worksheets to answer: {query}
        
        Available worksheets and their summaries:
        {json.dumps(sheet_summaries, indent=2)}
        
        Common columns across sheets: {self.worksheet_info['common_columns']}
        
        Please provide a comprehensive analysis that:
        1. Identifies relationships between the worksheets
        2. Combines or compares data where appropriate
        3. Provides specific insights to answer the query
        4. Mentions any data that needs to be joined or aggregated
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Try to perform actual data operations if possible
        try:
            data_operations = self._perform_cross_sheet_operations(query, relevant_sheets)
            if data_operations:
                return f"{response.content}\n\nData Analysis Results:\n{data_operations}"
        except Exception as e:
            pass
        
        return response.content
    
    def _perform_cross_sheet_operations(self, query: str, relevant_sheets: List[str]) -> str:
        """Attempt to perform actual data operations across sheets"""
        results = []
        
        # Check for common columns that could be used for joining
        common_cols = []
        for col, sheets in self.worksheet_info['common_columns'].items():
            if len([s for s in sheets if s in relevant_sheets]) > 1:
                common_cols.append(col)
        
        if common_cols:
            # Try to join data on common columns
            base_sheet = relevant_sheets[0]
            base_df = self.worksheets[base_sheet].copy()
            
            for sheet_name in relevant_sheets[1:]:
                other_df = self.worksheets[sheet_name].copy()
                
                # Find best join column
                join_col = None
                for col in common_cols:
                    if col in base_df.columns and col in other_df.columns:
                        join_col = col
                        break
                
                if join_col:
                    try:
                        # Perform left join
                        base_df = base_df.merge(other_df, on=join_col, how='left', suffixes=('', f'_{sheet_name}'))
                        results.append(f"Successfully joined {base_sheet} with {sheet_name} on '{join_col}'")
                    except Exception as e:
                        results.append(f"Could not join {base_sheet} with {sheet_name}: {str(e)}")
            
            if results:
                # Create a temporary agent for the combined data
                try:
                    combined_agent = create_pandas_dataframe_agent(
                        llm=self.llm,
                        df=base_df,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=False,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True
                    )
                    
                    result = combined_agent.invoke({"input": query})
                    combined_result = result['output']
                    results.append(f"Combined Analysis: {combined_result}")
                except Exception as e:
                    results.append(f"Error in combined analysis: {str(e)}")
        
        return "\n".join(results) if results else ""
    
    def query(self, question: str) -> str:
        try:
            # Determine which worksheets are relevant
            relevant_sheets = self._determine_relevant_worksheets(question)
            
            if not relevant_sheets:
                return "Could not determine relevant worksheets for your query."
            
            # Perform cross-sheet analysis
            result = self._create_cross_sheet_analysis(question, relevant_sheets)
            return result
            
        except Exception as e:
            return f"Error querying Excel file: {str(e)}"
    
    def get_data_info(self) -> str:
        info = f"Excel file contains {self.worksheet_info['total_worksheets']} worksheets:\n\n"
        
        for sheet_name, sheet_info in self.worksheet_info['worksheets'].items():
            info += f"Sheet '{sheet_name}':\n"
            info += f"  - Shape: {sheet_info['shape']}\n"
            info += f"  - Columns: {sheet_info['columns']}\n"
            info += f"  - Numeric columns: {sheet_info['numeric_columns']}\n"
            info += f"  - Categorical columns: {sheet_info['categorical_columns']}\n\n"
        
        if self.worksheet_info['common_columns']:
            info += f"Common columns across sheets: {self.worksheet_info['common_columns']}\n"
        
        return info
    
    def is_applicable(self, query: str) -> bool:
        excel_keywords = [
            'excel', 'spreadsheet', 'worksheet', 'workbook',
            'cell', 'formula', 'pivot', 'chart', 'xlsx', 'sheet'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in excel_keywords)