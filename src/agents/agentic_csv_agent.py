import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import json
import re
from dataclasses import dataclass

@dataclass
class DataInsight:
    finding: str
    confidence: float
    source_columns: List[str]
    calculation_used: Optional[str] = None

class AgenticCSVAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load CSV files
        self.csv_files = {}
        self.data_insights = []
        self._load_csv_files()
        
    def _load_csv_files(self):
        """Load all configured CSV files dynamically"""
        loaded_files = []
        
        # Method 1: Load from CSV directory
        csv_dir = os.getenv("CSV_DIRECTORY", ".")
        if os.path.exists(csv_dir):
            for filename in os.listdir(csv_dir):
                if filename.endswith('.csv'):
                    self._load_single_csv(os.path.join(csv_dir, filename), loaded_files)
        
        # Method 2: Load from individual environment variables
        env_vars = dict(os.environ)
        for env_name, path in env_vars.items():
            if env_name.startswith("CSV_FILE_PATH") and path and os.path.exists(path):
                self._load_single_csv(path, loaded_files)
        
        # Method 3: Load from CSV_FILES environment variable
        csv_files_env = os.getenv("CSV_FILES")
        if csv_files_env:
            for path in csv_files_env.split(','):
                path = path.strip()
                if path and os.path.exists(path):
                    self._load_single_csv(path, loaded_files)
        
        if not self.csv_files:
            raise ValueError("No valid CSV files found. Check your environment configuration.")
        
        print(f"Loaded CSV files: {loaded_files}")
    
    def _load_single_csv(self, file_path: str, loaded_files: List[str]):
        """Load a single CSV file"""
        try:
            filename = os.path.basename(file_path)
            name = filename.replace('.csv', '').replace('_', ' ').replace('-', ' ').lower()
            
            if name in self.csv_files:
                return
            
            df = pd.read_csv(file_path)
            
            # Enhanced preprocessing
            df = self._intelligent_preprocessing(df, name)
            
            self.csv_files[name] = {
                'dataframe': df,
                'path': file_path,
                'shape': df.shape,
                'columns': list(df.columns),
                'filename': filename,
                'data_types': df.dtypes.to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
            
            loaded_files.append(f"{name}: {file_path}")
            
        except Exception as e:
            print(f"Warning: Could not load CSV file {file_path}: {e}")
    
    def _intelligent_preprocessing(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Intelligent preprocessing based on data understanding"""
        df = df.copy()
        
        # Smart date detection and processing
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
                if col.lower() == 'year' and df[col].dtype in ['int64', 'float64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():
                        df[f'{col}_Year'] = df[col].dt.year
                        df[f'{col}_Month'] = df[col].dt.month
                        df[f'{col}_Quarter'] = df[col].dt.quarter
        
        # Smart numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() / len(df) > 0.8:  # 80% numeric
                    df[col] = numeric_series
        
        # Create computed columns based on domain knowledge
        self._create_domain_specific_columns(df, dataset_name)
        
        return df
    
    def _create_domain_specific_columns(self, df: pd.DataFrame, dataset_name: str):
        """Create domain-specific computed columns"""
        
        # Mining/Production specific columns
        if any(keyword in dataset_name for keyword in ['production', 'mining', 'operational']):
            # Production efficiency metrics
            if 'OreProcessed' in df.columns and 'MetalProduced' in df.columns:
                df['ProductionEfficiency'] = df['MetalProduced'] / df['OreProcessed']
            
            # Equipment utilization scoring
            if 'EquipmentUtilization' in df.columns:
                df['UtilizationScore'] = pd.cut(df['EquipmentUtilization'], 
                                               bins=[0, 60, 80, 95, 100], 
                                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # ESG specific columns
        if 'esg' in dataset_name or any(col in df.columns for col in ['GHGEmissions_tCO2e', 'WaterUse_m3']):
            # Environmental impact scoring
            if 'GHGEmissions_tCO2e' in df.columns:
                df['GHG_Impact_Level'] = pd.cut(df['GHGEmissions_tCO2e'], 
                                               bins=[0, 100, 500, 1000, float('inf')], 
                                               labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Financial ratios if financial data present
        if any(keyword in str(df.columns).lower() for keyword in ['revenue', 'cost', 'profit']):
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower()]
            cost_cols = [col for col in df.columns if any(term in col.lower() for term in ['cost', 'expense'])]
            
            if revenue_cols and cost_cols:
                df['Profit_Margin'] = ((df[revenue_cols[0]] - df[cost_cols[0]]) / df[revenue_cols[0]]) * 100
    
    def query(self, question: str) -> str:
        """Main agentic query processing with reasoning loops"""
        
        # Step 1: Understand the query intent
        query_analysis = self._analyze_query_intent(question)
        
        # Step 2: Determine relevant datasets
        relevant_datasets = self._determine_relevant_datasets(question, query_analysis)
        
        # Step 3: Reasoning loop for data exploration
        insights = []
        for dataset_name in relevant_datasets:
            dataset_insights = self._reason_through_dataset(dataset_name, question, query_analysis)
            insights.extend(dataset_insights)
        
        # Step 4: Cross-dataset reasoning if multiple datasets
        if len(relevant_datasets) > 1:
            cross_insights = self._cross_dataset_reasoning(insights, question)
            insights.extend(cross_insights)
        
        # Step 5: Generate final response with confidence
        return self._generate_final_response(question, insights, query_analysis)
    
    def _analyze_query_intent(self, question: str) -> Dict[str, Any]:
        """Analyze what the user is really asking for"""
        
        intent_prompt = PromptTemplate.from_template("""
        Analyze this query to understand the user's intent and requirements.
        
        QUERY: {question}
        
        Extract:
        1. Main intent (calculation, comparison, trend analysis, summary, etc.)
        2. Key metrics or KPIs mentioned
        3. Entities or filters (locations, time periods, categories)
        4. Calculation type needed (sum, average, ratio, percentage, etc.)
        5. Expected answer format
        6. Business context (financial, operational, environmental, etc.)
        
        Respond with JSON:
        {{
            "intent": "primary intent",
            "metrics": ["list of metrics"],
            "entities": ["list of entities/filters"],
            "calculation_type": "calculation needed",
            "answer_format": "expected format",
            "business_context": "context area",
            "complexity": "simple|medium|complex"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=intent_prompt.format(question=question))
        ])
        
        try:
            return json.loads(response.content.strip())
        except:
            return {
                "intent": "analysis",
                "metrics": [],
                "entities": [],
                "calculation_type": "summary",
                "answer_format": "descriptive",
                "business_context": "general",
                "complexity": "medium"
            }
    
    def _determine_relevant_datasets(self, question: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Intelligently determine which datasets are relevant"""
        
        dataset_scores = {}
        
        for dataset_name, dataset_info in self.csv_files.items():
            score = 0
            
            # Score based on query analysis
            for metric in query_analysis.get('metrics', []):
                metric_lower = metric.lower()
                for col in dataset_info['columns']:
                    if metric_lower in col.lower():
                        score += 10
                    elif any(word in col.lower() for word in metric_lower.split()):
                        score += 5
            
            # Score based on business context
            context = query_analysis.get('business_context', '').lower()
            if context in dataset_name.lower():
                score += 15
            
            # Score based on column content analysis
            for col in dataset_info['columns']:
                col_lower = col.lower()
                if any(word in col_lower for word in question.lower().split()):
                    score += 3
            
            dataset_scores[dataset_name] = score
        
        # Return datasets with score > 0, sorted by score
        relevant = [name for name, score in dataset_scores.items() if score > 0]
        return sorted(relevant, key=lambda x: dataset_scores[x], reverse=True)
    
    def _reason_through_dataset(self, dataset_name: str, question: str, query_analysis: Dict[str, Any]) -> List[DataInsight]:
        """Reason through a single dataset to extract insights"""
        
        dataset_info = self.csv_files[dataset_name]
        df = dataset_info['dataframe']
        
        # Generate reasoning prompt
        reasoning_prompt = PromptTemplate.from_template("""
        You are analyzing a dataset to answer a user's question. Use step-by-step reasoning.
        
        USER QUESTION: {question}
        QUERY ANALYSIS: {query_analysis}
        
        DATASET: {dataset_name}
        COLUMNS: {columns}
        SAMPLE DATA: {sample_data}
        SHAPE: {shape}
        
        Think step by step:
        1. What specific columns are relevant to this question?
        2. What calculations or analysis do I need to perform?
        3. Are there any data quality issues I need to consider?
        4. What filters or groupings are needed?
        5. What's the best approach to get the answer?
        
        Provide your reasoning and then generate specific pandas code to execute.
        
        Respond with JSON:
        {{
            "reasoning": "your step-by-step reasoning",
            "relevant_columns": ["list of columns"],
            "pandas_code": "executable pandas code using variable 'df'",
            "expected_outcome": "what the code should produce",
            "confidence": "high|medium|low"
        }}
        """)
        
        response = self.llm.invoke([
            HumanMessage(content=reasoning_prompt.format(
                question=question,
                query_analysis=str(query_analysis),
                dataset_name=dataset_name,
                columns=dataset_info['columns'],
                sample_data=str(dataset_info['sample_data']),
                shape=dataset_info['shape']
            ))
        ])
        
        try:
            reasoning_result = json.loads(response.content.strip())
            
            # Execute the pandas code
            pandas_code = reasoning_result.get('pandas_code', '')
            if pandas_code:
                # Create a safe execution environment
                local_vars = {'df': df, 'pd': pd}
                try:
                    exec(pandas_code, {"__builtins__": {}}, local_vars)
                    
                    # Get the result (assume it's stored in 'result' variable)
                    if 'result' in local_vars:
                        result = local_vars['result']
                        
                        return [DataInsight(
                            finding=str(result),
                            confidence=0.8 if reasoning_result.get('confidence') == 'high' else 0.6,
                            source_columns=reasoning_result.get('relevant_columns', []),
                            calculation_used=pandas_code
                        )]
                    
                except Exception as e:
                    # If execution fails, try alternative approach
                    return self._fallback_analysis(df, question, dataset_name)
            
        except Exception as e:
            return self._fallback_analysis(df, question, dataset_name)
        
        return []
    
    def _fallback_analysis(self, df: pd.DataFrame, question: str, dataset_name: str) -> List[DataInsight]:
        """Fallback analysis when code execution fails"""
        
        insights = []
        
        # Simple statistical analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if any(keyword in col.lower() for keyword in question.lower().split()):
                    summary = df[col].describe()
                    insights.append(DataInsight(
                        finding=f"{col} statistics: Mean={summary['mean']:.2f}, Max={summary['max']:.2f}",
                        confidence=0.5,
                        source_columns=[col]
                    ))
        
        return insights
    
    def _cross_dataset_reasoning(self, insights: List[DataInsight], question: str) -> List[DataInsight]:
        """Reason across multiple datasets to find connections"""
        
        if len(insights) < 2:
            return []
        
        cross_reasoning_prompt = PromptTemplate.from_template("""
        You have insights from multiple datasets. Find connections and provide cross-dataset analysis.
        
        USER QUESTION: {question}
        
        INSIGHTS:
        {insights}
        
        Analyze:
        1. Are there complementary insights that can be combined?
        2. Are there contradictions that need to be resolved?
        3. Can you calculate new metrics by combining datasets?
        4. What additional insights emerge from the combination?
        
        Provide cross-dataset insights that add value beyond individual dataset analysis.
        """)
        
        insights_text = "\n".join([
            f"- {insight.finding} (confidence: {insight.confidence})"
            for insight in insights
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=cross_reasoning_prompt.format(
                question=question,
                insights=insights_text
            ))
        ])
        
        # Parse response for new insights
        cross_insight = DataInsight(
            finding=response.content.strip(),
            confidence=0.7,
            source_columns=["multiple_datasets"]
        )
        
        return [cross_insight]
    
    def _generate_final_response(self, question: str, insights: List[DataInsight], query_analysis: Dict[str, Any]) -> str:
        """Generate final response with confidence indicators"""
        
        # Calculate overall confidence
        if insights:
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
        else:
            avg_confidence = 0.3
        
        # Generate response
        response_prompt = PromptTemplate.from_template("""
        Generate a comprehensive response to the user's question based on the insights gathered.
        
        USER QUESTION: {question}
        
        INSIGHTS GATHERED:
        {insights}
        
        OVERALL CONFIDENCE: {confidence}
        
        Create a response that:
        1. Directly answers the question with specific data
        2. Provides context and explanations
        3. Shows calculations if applicable
        4. Indicates confidence level and any limitations
        5. Suggests additional analysis if relevant
        
        Format the response professionally with clear data points and reasoning.
        """)
        
        insights_text = "\n".join([
            f"• {insight.finding} (Confidence: {insight.confidence:.1f}, Sources: {', '.join(insight.source_columns)})"
            for insight in insights
        ])
        
        response = self.llm.invoke([
            HumanMessage(content=response_prompt.format(
                question=question,
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
        
        return f"{final_response}\n\n**Confidence Level: {confidence_text}**"
    
    def get_data_info(self) -> str:
        """Get information about available datasets"""
        info = f"Available CSV datasets: {len(self.csv_files)}\n\n"
        
        for name, csv_info in self.csv_files.items():
            info += f"📊 {name.title()} Dataset:\n"
            info += f"  • Shape: {csv_info['shape'][0]:,} rows × {csv_info['shape'][1]} columns\n"
            info += f"  • Key columns: {', '.join(csv_info['columns'][:5])}{'...' if len(csv_info['columns']) > 5 else ''}\n"
            info += f"  • Data types: {len([t for t in csv_info['data_types'].values() if 'int' in str(t) or 'float' in str(t)])} numeric, {len([t for t in csv_info['data_types'].values() if 'object' in str(t)])} text\n\n"
        
        return info