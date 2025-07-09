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
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        excel_file_path = os.getenv("EXCEL_FILE_PATH")
        if not excel_file_path or not os.path.exists(excel_file_path):
            raise ValueError(f"Excel file not found: {excel_file_path}")
        
        # Load all worksheets
        self.worksheets = pd.read_excel(excel_file_path, sheet_name=None)
        self.worksheet_info = self._analyze_worksheets()
        
        # Create agents for each worksheet
        self.worksheet_agents = {}
        for sheet_name, df in self.worksheets.items():
            if not df.empty:
                self.worksheet_agents[sheet_name] = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=df,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True
                )
    
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
    
    def _determine_relevant_worksheets(self, query: str) -> List[str]:
        """Use LLM to determine which worksheets are relevant for the query"""
        worksheet_descriptions = []
        for sheet_name, info in self.worksheet_info['worksheets'].items():
            description = f"Sheet '{sheet_name}': {info['shape'][0]} rows, columns: {info['columns']}"
            worksheet_descriptions.append(description)
        
        prompt = f"""
        Given the following Excel worksheets and a user query, determine which worksheets are most relevant.
        
        Worksheets:
        {chr(10).join(worksheet_descriptions)}
        
        Common columns across sheets: {self.worksheet_info['common_columns']}
        
        User query: {query}
        
        Return only the sheet names that are relevant, separated by commas. If multiple sheets need to be combined, include all relevant sheets.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        relevant_sheets = [sheet.strip() for sheet in response.content.split(',')]
        
        # Filter to only existing sheets
        return [sheet for sheet in relevant_sheets if sheet in self.worksheets.keys()]
    
    def _create_cross_sheet_analysis(self, query: str, relevant_sheets: List[str]) -> str:
        """Create a combined analysis across multiple worksheets"""
        if len(relevant_sheets) == 1:
            return self.worksheet_agents[relevant_sheets[0]].run(query)
        
        # For multiple sheets, create a comprehensive analysis
        combined_data = {}
        sheet_summaries = {}
        
        for sheet_name in relevant_sheets:
            df = self.worksheets[sheet_name]
            combined_data[sheet_name] = df
            
            # Get basic summary from individual sheet agent
            try:
                summary = self.worksheet_agents[sheet_name].run(f"Provide a brief summary of this data relevant to: {query}")
                sheet_summaries[sheet_name] = summary
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
                    
                    combined_result = combined_agent.run(query)
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