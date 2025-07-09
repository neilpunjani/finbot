import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import HumanMessage
from typing import Dict, List, Any

class CSVAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load multiple CSV files
        self.csv_files = {}
        self.agents = {}  # Lazy-loaded agents
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
        
        # Method 3: Load from CSV_FILES environment variable (comma-separated paths)
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
        """Load a single CSV file and create its agent"""
        try:
            # Generate a clean name from the filename
            filename = os.path.basename(file_path)
            name = filename.replace('.csv', '').replace('_', ' ').replace('-', ' ').lower()
            
            # Avoid duplicates
            if name in self.csv_files:
                return
            
            df = pd.read_csv(file_path)
            self.csv_files[name] = {
                'dataframe': df,
                'path': file_path,
                'shape': df.shape,
                'columns': list(df.columns),
                'filename': filename
            }
            
            # Don't create agents yet - lazy load them when needed
            
            loaded_files.append(f"{name}: {file_path}")
            
        except Exception as e:
            print(f"Warning: Could not load CSV file {file_path}: {e}")
    
    def _get_or_create_agent(self, csv_name: str):
        """Lazily create and return agent for the specified CSV"""
        if csv_name not in self.agents:
            if csv_name in self.csv_files:
                df = self.csv_files[csv_name]['dataframe']
                self.agents[csv_name] = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=df,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True
                )
        return self.agents.get(csv_name)
    
    def _determine_relevant_csv(self, query: str) -> str:
        """Determine which CSV file is most relevant for the query using intelligent scoring"""
        query_lower = query.lower()
        
        # Score each CSV file
        scores = {}
        for csv_name, csv_info in self.csv_files.items():
            score = 0
            
            # Score based on filename keywords
            filename_words = csv_name.split()
            for word in filename_words:
                if word in query_lower:
                    score += 3
            
            # Score based on column name matches
            for column in csv_info['columns']:
                column_lower = column.lower()
                if column_lower in query_lower:
                    score += 2
                # Check for partial matches
                for word in column_lower.split('_'):
                    if word in query_lower:
                        score += 1
            
            # Enhanced scoring for production-related queries
            if any(keyword in query_lower for keyword in ['produced', 'production', 'metal', 'recovery', 'grade', 'ore']):
                if 'production' in csv_name.lower():
                    score += 10  # Strong preference for production data
                elif any(col in csv_info['columns'] for col in ['MetalProduced', 'RecoveryRate', 'Grade', 'OreProcessed']):
                    score += 5
            
            # Enhanced scoring for operational queries
            if any(keyword in query_lower for keyword in ['equipment', 'utilization', 'downtime', 'tonnes']):
                if 'operational' in csv_name.lower():
                    score += 10
                elif any(col in csv_info['columns'] for col in ['EquipmentUtilization', 'DowntimeHours', 'TonnesMoved']):
                    score += 5
            
            # Enhanced scoring for workforce queries
            if any(keyword in query_lower for keyword in ['training', 'headcount', 'workforce', 'contractor']):
                if 'workforce' in csv_name.lower():
                    score += 10
                elif any(col in csv_info['columns'] for col in ['TrainingHours', 'Headcount', 'ContractorRatio']):
                    score += 5
            
            # Enhanced scoring for environmental queries
            if any(keyword in query_lower for keyword in ['ghg', 'emissions', 'water', 'energy', 'environmental']):
                if 'esg' in csv_name.lower():
                    score += 10
                elif any(col in csv_info['columns'] for col in ['GHGEmissions_tCO2e', 'WaterUse_m3', 'Energy_kWh']):
                    score += 5
            
            # Score based on data content similarity
            if 'customer' in query_lower:
                if any('customer' in col.lower() for col in csv_info['columns']):
                    score += 2
                if any('name' in col.lower() for col in csv_info['columns']):
                    score += 1
            
            if any(keyword in query_lower for keyword in ['sales', 'revenue', 'product', 'purchase', 'order']):
                if any(keyword in col.lower() for col in csv_info['columns'] for keyword in ['sales', 'revenue', 'product', 'amount', 'price']):
                    score += 2
            
            scores[csv_name] = score
        
        # Return the CSV with highest score
        if scores:
            best_csv = max(scores.items(), key=lambda x: x[1])[0]
            if scores[best_csv] > 0:
                return best_csv
        
        # Fallback: use LLM to decide
        return self._llm_csv_selection(query)
    
    def _llm_csv_selection(self, query: str) -> str:
        """Use LLM to select the best CSV file when scoring doesn't give clear winner"""
        try:
            csv_descriptions = []
            for csv_name, csv_info in self.csv_files.items():
                description = f"{csv_name}: {csv_info['shape'][0]} rows, columns: {csv_info['columns']}"
                csv_descriptions.append(description)
            
            prompt = f"""
            Given the following CSV files and a user query, determine which CSV file would be most relevant.
            
            Available CSV files:
            {chr(10).join(csv_descriptions)}
            
            User query: "{query}"
            
            Return only the CSV name (without extension) that would be most relevant for answering this query.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            selected_csv = response.content.strip().lower()
            
            # Validate the selection
            if selected_csv in self.csv_files:
                return selected_csv
            
            # If LLM response doesn't match, try partial matching
            for csv_name in self.csv_files.keys():
                if csv_name in selected_csv or selected_csv in csv_name:
                    return csv_name
                    
        except Exception as e:
            print(f"Error in LLM CSV selection: {e}")
        
        # Ultimate fallback: return first available CSV
        return list(self.csv_files.keys())[0]
    
    def _analyze_cross_csv(self, query: str) -> str:
        """Analyze data across multiple CSV files if needed"""
        
        # Check if query mentions multiple data types
        query_lower = query.lower()
        mentions_sales = any(word in query_lower for word in ['sales', 'revenue', 'product', 'purchase'])
        mentions_customers = any(word in query_lower for word in ['customer', 'client', 'user'])
        
        if mentions_sales and mentions_customers and 'sales' in self.csv_files and 'customers' in self.csv_files:
            try:
                # Try to join the data if there's a common column
                sales_df = self.csv_files['sales']['dataframe']
                customers_df = self.csv_files['customers']['dataframe']
                
                # Find common columns for joining
                common_cols = set(sales_df.columns) & set(customers_df.columns)
                
                if common_cols:
                    join_col = list(common_cols)[0]  # Use first common column
                    
                    # Perform join
                    combined_df = sales_df.merge(customers_df, on=join_col, how='left', suffixes=('_sales', '_customer'))
                    
                    # Direct analysis of combined data
                    result = self._analyze_combined_data(combined_df, query, join_col)
                    return f"[Combined Sales + Customer Analysis]\n{result}"
                
            except Exception as e:
                return f"Could not combine CSV data: {str(e)}"
        
        return None
    
    def query(self, question: str) -> str:
        try:
            # Determine which CSV to use
            relevant_csv = self._determine_relevant_csv(question)
            
            # Check if this is a specific query pattern we can handle directly
            direct_result = self._try_direct_calculation(question, relevant_csv)
            if direct_result:
                return f"[{relevant_csv.title()} CSV Analysis]\n{direct_result}"
            
            # Use the pandas agent for intelligent query handling
            agent = self._get_or_create_agent(relevant_csv)
            if agent:
                # Add data context to help the agent understand the data structure
                enhanced_question = self._enhance_question_with_context(question, relevant_csv)
                result = agent.invoke({"input": enhanced_question})
                return f"[{relevant_csv.title()} CSV Analysis]\n{result['output']}"
            else:
                return "No suitable CSV file found for this query."
                
        except Exception as e:
            return f"Error querying CSV files: {str(e)}"
    
    def _try_direct_calculation(self, question: str, csv_name: str) -> str:
        """Try to handle query patterns with direct calculations using comprehensive preprocessing"""
        if csv_name not in self.csv_files:
            return None
            
        try:
            question_lower = question.lower()
            df = self.csv_files[csv_name]['dataframe'].copy()
            
            # Comprehensive preprocessing for all CSV files
            df = self._preprocess_dataframe(df)
            
            # Parse the query to extract components
            query_components = self._parse_query_components(question_lower)
            
            # Apply filters based on query components
            filtered_df = self._apply_query_filters(df, query_components)
            
            # Calculate the requested metric
            result = self._calculate_metric(filtered_df, query_components, csv_name)
            
            return result
            
        except Exception as e:
            # If direct calculation fails, return None to fall back to pandas agent
            return None
    
    def _preprocess_dataframe(self, df):
        """Comprehensive preprocessing for all CSV dataframes"""
        df = df.copy()
        
        # Date processing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
        
        # Standardize text columns
        text_columns = ['Entity', 'Site', 'Commodity', 'Scenario']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Create computed columns for common queries
        if 'GHGEmissions_tCO2e' in df.columns:
            df['GHG_Emissions'] = df['GHGEmissions_tCO2e']
        
        if 'WaterUse_m3' in df.columns:
            df['Water_Use'] = df['WaterUse_m3']
        
        if 'Energy_kWh' in df.columns:
            df['Energy'] = df['Energy_kWh']
        
        return df
    
    def _parse_query_components(self, question_lower):
        """Parse query to extract entities, metrics, time periods, etc."""
        components = {
            'entities': [],
            'sites': [],
            'commodities': [],
            'scenarios': [],
            'years': [],
            'metrics': [],
            'aggregations': []
        }
        
        # Extract entities
        entities = ['nova scotia', 'alberta', 'ontario', 'quebec', 'british columbia', 'canada', 'holding company']
        for entity in entities:
            if entity in question_lower:
                components['entities'].append(entity.title())
        
        # Extract sites with flexible matching
        site_mappings = {
            'north pit': ['north pit', 'north site', 'north'],
            'south pit': ['south pit', 'south site', 'south'],
            'mill a': ['mill a', 'mill 1', 'first mill'],
            'mill b': ['mill b', 'mill 2', 'second mill']
        }
        
        for canonical_site, variations in site_mappings.items():
            for variation in variations:
                if variation in question_lower:
                    components['sites'].append(canonical_site.title())
                    break
        
        # Extract commodities
        commodities = ['gold', 'copper', 'zinc', 'nickel']
        for commodity in commodities:
            if commodity in question_lower:
                components['commodities'].append(commodity.title())
        
        # Extract scenarios
        scenarios = ['actual', 'budget']
        for scenario in scenarios:
            if scenario in question_lower:
                components['scenarios'].append(scenario.title())
        
        # Extract years
        import re
        years = re.findall(r'\b(202[0-9])\b', question_lower)
        components['years'] = [int(year) for year in years]
        
        # SMART METRIC INTERPRETATION - This is the key enhancement!
        # Parse the query intelligently to infer what metric is actually being requested
        components['metrics'] = self._smart_metric_interpretation(question_lower, components)
        
        # Extract aggregation type
        if any(word in question_lower for word in ['total', 'sum']):
            components['aggregations'].append('sum')
        elif any(word in question_lower for word in ['average', 'mean']):
            components['aggregations'].append('mean')
        elif any(word in question_lower for word in ['maximum', 'max']):
            components['aggregations'].append('max')
        elif any(word in question_lower for word in ['minimum', 'min']):
            components['aggregations'].append('min')
        else:
            components['aggregations'].append('sum')  # default
        
        # Extract decimal place requirements
        decimal_patterns = [
            r'(\d+)\s*decimal\s*place',
            r'(\d+)\s*decimal',
            r'(\d+)\s*dp',
            r'to\s*(\d+)\s*decimal'
        ]
        
        components['decimal_places'] = None
        for pattern in decimal_patterns:
            match = re.search(pattern, question_lower)
            if match:
                components['decimal_places'] = int(match.group(1))
                break
        
        return components
    
    def _smart_metric_interpretation(self, question_lower, components):
        """Intelligently interpret what metric the user is actually asking for"""
        import re
        
        # Define smart patterns that map natural language to actual metrics
        smart_patterns = {
            # Pattern: (regex_pattern, target_metric, required_commodity_context)
            
            # PRODUCTION METRICS  
            (r'\b(?:total|sum|amount of|how much|quantity of)?\s*(?:gold|copper|zinc|nickel)\s+(?:produced|production|output|generated)', 'MetalProduced', True),
            (r'\b(?:how much|amount of|quantity of)\s+(?:gold|copper|zinc|nickel)\s+(?:was|were|is|are)?\s*(?:produced|generated)', 'MetalProduced', True),
            (r'\b(?:gold|copper|zinc|nickel)\s+(?:produced|production|output)', 'MetalProduced', True),
            (r'\b(?:total|sum|amount of)?\s*(?:metal|metals)\s+(?:produced|production|output)', 'MetalProduced', False),
            (r'\b(?:total|sum|amount of)?\s*(?:ore|material)\s+(?:processed|processing|handled)', 'OreProcessed', False),
            (r'\b(?:total|sum|amount of)?\s*(?:tonnes|tons)\s+(?:moved|transported|handled)', 'TonnesMoved', False),
            
            # RECOVERY AND GRADE METRICS  
            (r'\b(?:recovery|recovery rate|extraction rate)\s+(?:of|for)?\s*(?:gold|copper|zinc|nickel)', 'RecoveryRate', True),
            (r'\b(?:grade|ore grade|metal grade)\s+(?:of|for)?\s*(?:gold|copper|zinc|nickel)', 'Grade', True),
            (r'\b(?:average|mean)?\s*(?:recovery|recovery rate)', 'RecoveryRate', False),
            (r'\b(?:average|mean)?\s*(?:grade|ore grade)', 'Grade', False),
            (r'\b(?:gold|copper|zinc|nickel)\s+(?:recovery|recovery rate)', 'RecoveryRate', True),
            (r'\b(?:gold|copper|zinc|nickel)\s+(?:grade|ore grade)', 'Grade', True),
            
            # OPERATIONAL METRICS
            (r'\b(?:equipment|machinery)\s+(?:utilization|usage|efficiency)', 'EquipmentUtilization', False),
            (r'\b(?:downtime|down time)\s+(?:hours|time)', 'DowntimeHours', False),
            (r'\b(?:utilization|usage)\s+(?:rate|percentage)', 'EquipmentUtilization', False),
            
            # WORKFORCE METRICS
            (r'\b(?:training|train)\s+(?:hours|time)', 'TrainingHours', False),
            (r'\b(?:headcount|head count|workforce|employees|staff)', 'Headcount', False),
            (r'\b(?:contractor|contractors)\s+(?:ratio|percentage)', 'ContractorRatio', False),
            
            # SAFETY METRICS
            (r'\btrifr\b', 'TRIFR', False),
            (r'\bltifr\b', 'LTIFR', False),
            (r'\b(?:total recordable injury frequency|total recordable injury)', 'TRIFR', False),
            (r'\b(?:lost time injury frequency|lost time injury)', 'LTIFR', False),
            
            # ENVIRONMENTAL METRICS
            (r'\b(?:ghg|greenhouse gas|carbon)\s+(?:emissions|emission)', 'GHG_Emissions', False),
            (r'\b(?:emissions|emission)\b', 'GHG_Emissions', False),
            (r'\b(?:water|h2o)\s+(?:use|usage|consumption)', 'Water_Use', False),
            (r'\b(?:energy|power)\s+(?:use|usage|consumption)', 'Energy', False),
        }
        
        detected_metrics = []
        
        # Check each smart pattern
        for pattern, metric, requires_commodity in smart_patterns:
            if re.search(pattern, question_lower):
                # If the pattern requires commodity context, check if commodity is mentioned
                if requires_commodity and components['commodities']:
                    detected_metrics.append(metric)
                elif not requires_commodity:
                    detected_metrics.append(metric)
        
        # If no smart patterns matched, fall back to traditional keyword matching
        if not detected_metrics:
            detected_metrics = self._traditional_metric_matching(question_lower)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(detected_metrics))
    
    def _traditional_metric_matching(self, question_lower):
        """Traditional keyword-based metric matching as fallback"""
        metric_mappings = {
            'training hours': 'TrainingHours',
            'headcount': 'Headcount',
            'contractor ratio': 'ContractorRatio',
            'ghg emissions': 'GHG_Emissions',
            'emissions': 'GHG_Emissions',
            'water use': 'Water_Use',
            'energy': 'Energy',
            'equipment utilization': 'EquipmentUtilization',
            'downtime hours': 'DowntimeHours',
            'tonnes moved': 'TonnesMoved',
            'ore processed': 'OreProcessed',
            'grade': 'Grade',
            'recovery rate': 'RecoveryRate',
            'metal produced': 'MetalProduced',
            'trifr': 'TRIFR',
            'ltifr': 'LTIFR'
        }
        
        detected_metrics = []
        for metric_phrase, column_name in metric_mappings.items():
            if metric_phrase in question_lower:
                detected_metrics.append(column_name)
        
        return detected_metrics
    
    def _apply_query_filters(self, df, components):
        """Apply filters based on parsed query components"""
        filtered_df = df.copy()
        
        # Filter by entities
        if components['entities']:
            filtered_df = filtered_df[filtered_df['Entity'].isin(components['entities'])]
        
        # Filter by sites
        if components['sites'] and 'Site' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Site'].isin(components['sites'])]
        
        # Filter by commodities
        if components['commodities'] and 'Commodity' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Commodity'].isin(components['commodities'])]
        
        # Filter by scenarios
        if components['scenarios'] and 'Scenario' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Scenario'].isin(components['scenarios'])]
        
        # Filter by years
        if components['years'] and 'Year' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Year'].isin(components['years'])]
        
        return filtered_df
    
    def _calculate_metric(self, filtered_df, components, csv_name):
        """Calculate the requested metric from filtered data"""
        if filtered_df.empty:
            return f"No data found matching the specified criteria."
        
        if not components['metrics']:
            return f"Found {len(filtered_df)} records matching the criteria."
        
        results = []
        for metric in components['metrics']:
            if metric not in filtered_df.columns:
                continue
            
            # Get aggregation type
            agg_type = components['aggregations'][0] if components['aggregations'] else 'sum'
            
            # Calculate the metric
            if agg_type == 'sum':
                value = filtered_df[metric].sum()
            elif agg_type == 'mean':
                value = filtered_df[metric].mean()
            elif agg_type == 'max':
                value = filtered_df[metric].max()
            elif agg_type == 'min':
                value = filtered_df[metric].min()
            else:
                value = filtered_df[metric].sum()
            
            # Determine decimal places for formatting
            decimal_places = components.get('decimal_places')
            
            # Format the result with appropriate decimal places
            if decimal_places is not None:
                # Use the specified decimal places
                formatted_value = f"{value:.{decimal_places}f}"
            else:
                # Use default formatting based on metric type
                if metric == 'TrainingHours':
                    formatted_value = f"{value:.1f}"
                elif metric == 'Headcount':
                    formatted_value = f"{value:.1f}"
                elif metric in ['GHG_Emissions', 'GHGEmissions_tCO2e']:
                    formatted_value = f"{value:.1f}"
                elif metric in ['Water_Use', 'WaterUse_m3']:
                    formatted_value = f"{value:.1f}"
                elif metric in ['Energy', 'Energy_kWh']:
                    formatted_value = f"{value:.1f}"
                elif metric == 'EquipmentUtilization':
                    formatted_value = f"{value:.1f}"
                elif metric == 'DowntimeHours':
                    formatted_value = f"{value:.1f}"
                elif metric == 'TonnesMoved':
                    formatted_value = f"{value:.1f}"
                elif metric == 'OreProcessed':
                    formatted_value = f"{value:.1f}"
                elif metric == 'Grade':
                    formatted_value = f"{value:.4f}"
                elif metric == 'RecoveryRate':
                    formatted_value = f"{value:.1f}"
                elif metric == 'MetalProduced':
                    formatted_value = f"{value:.1f}"
                elif metric == 'TRIFR':
                    formatted_value = f"{value:.2f}"
                elif metric == 'LTIFR':
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2f}"
            
            # Create the result string with proper units
            if metric == 'TrainingHours':
                results.append(f"{agg_type.title()} training hours: {formatted_value}")
            elif metric == 'Headcount':
                results.append(f"{agg_type.title()} headcount: {formatted_value}")
            elif metric in ['GHG_Emissions', 'GHGEmissions_tCO2e']:
                results.append(f"{agg_type.title()} GHG emissions: {formatted_value} tCO2e")
            elif metric in ['Water_Use', 'WaterUse_m3']:
                results.append(f"{agg_type.title()} water use: {formatted_value} m3")
            elif metric in ['Energy', 'Energy_kWh']:
                results.append(f"{agg_type.title()} energy use: {formatted_value} kWh")
            elif metric == 'EquipmentUtilization':
                results.append(f"{agg_type.title()} equipment utilization: {formatted_value}%")
            elif metric == 'DowntimeHours':
                results.append(f"{agg_type.title()} downtime hours: {formatted_value}")
            elif metric == 'TonnesMoved':
                results.append(f"{agg_type.title()} tonnes moved: {formatted_value}")
            elif metric == 'OreProcessed':
                results.append(f"{agg_type.title()} ore processed: {formatted_value}")
            elif metric == 'Grade':
                results.append(f"{agg_type.title()} grade: {formatted_value}")
            elif metric == 'RecoveryRate':
                results.append(f"{agg_type.title()} recovery rate: {formatted_value}%")
            elif metric == 'MetalProduced':
                results.append(f"{agg_type.title()} metal produced: {formatted_value}")
            elif metric == 'TRIFR':
                results.append(f"{agg_type.title()} TRIFR: {formatted_value}")
            elif metric == 'LTIFR':
                results.append(f"{agg_type.title()} LTIFR: {formatted_value}")
            else:
                results.append(f"{agg_type.title()} {metric}: {formatted_value}")
        
        # Add context about filters applied
        context = []
        if components['entities']:
            context.append(f"Entity: {', '.join(components['entities'])}")
        if components['sites']:
            context.append(f"Site: {', '.join(components['sites'])}")
        if components['commodities']:
            context.append(f"Commodity: {', '.join(components['commodities'])}")
        if components['years']:
            context.append(f"Year: {', '.join(map(str, components['years']))}")
        if components['scenarios']:
            context.append(f"Scenario: {', '.join(components['scenarios'])}")
        
        result_text = "\\n".join(results)
        if context:
            result_text += f"\\n\\nFilters applied: {'; '.join(context)}"
        
        return result_text
    
    def _enhance_question_with_context(self, question: str, csv_name: str) -> str:
        """Add helpful context about the data structure to improve pandas agent performance"""
        csv_info = self.csv_files[csv_name]
        columns = csv_info['columns']
        
        context_hints = []
        
        # Add hints for date handling
        if 'Date' in columns and any(word in question.lower() for word in ['2023', '2024', '2022', 'year']):
            context_hints.append("Note: To work with years, first convert the Date column to datetime using pd.to_datetime(df['Date']) and then extract year with df['Date'].dt.year")
        
        # Add hints for entity filtering
        if 'Entity' in columns and any(entity in question.lower() for entity in ['nova scotia', 'alberta', 'ontario', 'quebec', 'canada']):
            context_hints.append("Note: Entity values are exact strings like 'Nova Scotia', 'Alberta', etc.")
        
        # Add hints for common aggregations
        if any(word in question.lower() for word in ['total', 'sum', 'average', 'mean']):
            context_hints.append("Note: Use appropriate pandas aggregation functions like sum(), mean(), etc.")
        
        if context_hints:
            enhanced = f"{question}\n\nContext hints:\n" + "\n".join(context_hints)
            return enhanced
        
        return question
    
    def _direct_csv_analysis(self, question: str, csv_name: str) -> str:
        """Perform direct analysis on CSV data without using pandas agent"""
        try:
            df = self.csv_files[csv_name]['dataframe']
            csv_info = self.csv_files[csv_name]
            question_lower = question.lower()
            
            # Handle common query types
            if 'columns' in question_lower or 'column' in question_lower:
                return f"Columns in {csv_name}: {', '.join(csv_info['columns'])}"
            
            elif 'shape' in question_lower or 'size' in question_lower or 'rows' in question_lower:
                return f"Dataset size: {csv_info['shape'][0]} rows, {csv_info['shape'][1]} columns"
            
            elif 'summary' in question_lower or 'describe' in question_lower:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    desc = df[numeric_cols].describe()
                    return f"Summary statistics:\n{desc.to_string()}"
                else:
                    return f"Dataset contains categorical data. Columns: {', '.join(csv_info['columns'])}"
            
            elif 'top' in question_lower and ('customer' in question_lower or 'sales' in question_lower):
                # Handle "top customers" or "top sales" queries
                amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'sales', 'revenue', 'total'])]
                if amount_cols:
                    top_data = df.nlargest(5, amount_cols[0])
                    return f"Top 5 records by {amount_cols[0]}:\n{top_data.to_string(index=False)}"
            
            elif 'customer' in question_lower and 'analysis' in question_lower:
                return self._customer_analysis(df, csv_name)
            
            elif 'head' in question_lower or 'sample' in question_lower:
                return f"Sample data:\n{df.head().to_string(index=False)}"
            
            # Handle location-based queries (state, city, country)
            elif any(location in question_lower for location in ['texas', 'california', 'new york', 'florida', 'illinois']) or 'state' in question_lower or 'city' in question_lower:
                return self._location_based_analysis(df, question, csv_name)
            
            # Handle specific customer queries
            elif 'customers' in question_lower and any(word in question_lower for word in ['in', 'from', 'present', 'located', 'where']):
                return self._customer_location_query(df, question, csv_name)
            
            # Default: provide basic info and sample
            sample_data = df.head(3).to_string(index=False)
            return f"Dataset info: {csv_info['shape'][0]} rows, {csv_info['shape'][1]} columns\nColumns: {', '.join(csv_info['columns'])}\n\nSample data:\n{sample_data}"
            
        except Exception as e:
            return f"Error analyzing {csv_name}: {str(e)}"
    
    def _customer_analysis(self, df, csv_name: str) -> str:
        """Perform customer-specific analysis"""
        try:
            result = []
            
            # Customer count
            customer_cols = [col for col in df.columns if 'customer' in col.lower()]
            if customer_cols:
                unique_customers = df[customer_cols[0]].nunique()
                result.append(f"Total unique customers: {unique_customers}")
            
            # Sales analysis if available
            amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'sales', 'revenue'])]
            if amount_cols:
                total_sales = df[amount_cols[0]].sum()
                avg_sales = df[amount_cols[0]].mean()
                result.append(f"Total sales: ${total_sales:,.2f}")
                result.append(f"Average transaction: ${avg_sales:,.2f}")
            
            # Region analysis if available
            region_cols = [col for col in df.columns if 'region' in col.lower()]
            if region_cols:
                region_counts = df[region_cols[0]].value_counts()
                result.append(f"Sales by region:\n{region_counts.to_string()}")
            
            return "\n".join(result) if result else f"Customer data available with columns: {', '.join(df.columns)}"
            
        except Exception as e:
            return f"Error in customer analysis: {str(e)}"
    
    def _location_based_analysis(self, df, question: str, csv_name: str) -> str:
        """Handle location-based queries (state, city, country)"""
        try:
            question_lower = question.lower()
            
            # Find location columns
            location_cols = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'state' in col_lower:
                    location_cols['state'] = col
                elif 'city' in col_lower:
                    location_cols['city'] = col
                elif 'country' in col_lower:
                    location_cols['country'] = col
            
            # Determine what location is being asked about
            target_location = None
            location_type = None
            
            if 'texas' in question_lower or 'tx' in question_lower:
                target_location = 'TX'
                location_type = 'state'
            elif 'california' in question_lower or 'ca' in question_lower:
                target_location = 'CA'
                location_type = 'state'
            elif 'new york' in question_lower or 'ny' in question_lower:
                target_location = 'NY'
                location_type = 'state'
            elif 'florida' in question_lower or 'fl' in question_lower:
                target_location = 'FL'
                location_type = 'state'
            elif 'illinois' in question_lower or 'il' in question_lower:
                target_location = 'IL'
                location_type = 'state'
            
            if target_location and location_type in location_cols:
                # Filter data by location
                location_col = location_cols[location_type]
                filtered_df = df[df[location_col] == target_location]
                
                if len(filtered_df) > 0:
                    result = f"Customers in {target_location}:\n"
                    result += f"Total customers: {len(filtered_df)}\n\n"
                    
                    # Show customer details
                    display_cols = []
                    for col in ['Name', 'Customer_ID', 'Email', 'City', 'State', 'Customer_Type']:
                        if col in filtered_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        result += filtered_df[display_cols].to_string(index=False)
                    else:
                        result += filtered_df.to_string(index=False)
                    
                    return result
                else:
                    return f"No customers found in {target_location}"
            
            # Fallback: show all unique locations
            if 'state' in location_cols:
                state_counts = df[location_cols['state']].value_counts()
                return f"Customers by state:\n{state_counts.to_string()}"
            
            return f"Location data available in columns: {list(location_cols.keys())}"
            
        except Exception as e:
            return f"Error in location analysis: {str(e)}"
    
    def _customer_location_query(self, df, question: str, csv_name: str) -> str:
        """Handle specific customer location queries"""
        try:
            # This is a more general handler for customer location queries
            return self._location_based_analysis(df, question, csv_name)
            
        except Exception as e:
            return f"Error in customer location query: {str(e)}"
    
    def _analyze_combined_data(self, combined_df, query: str, join_col: str) -> str:
        """Analyze combined sales and customer data"""
        try:
            result = []
            result.append(f"Successfully joined data on '{join_col}' column")
            result.append(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            
            # Customer insights
            if any('customer' in col.lower() for col in combined_df.columns):
                customer_count = combined_df[join_col].nunique()
                result.append(f"Total customers with sales data: {customer_count}")
            
            # Sales insights
            amount_cols = [col for col in combined_df.columns if any(term in col.lower() for term in ['amount', 'sales', 'revenue'])]
            if amount_cols:
                total_sales = combined_df[amount_cols[0]].sum()
                result.append(f"Total sales amount: ${total_sales:,.2f}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error analyzing combined data: {str(e)}"
    
    def get_data_info(self) -> str:
        info = f"Available CSV files: {len(self.csv_files)}\n\n"
        
        for name, csv_info in self.csv_files.items():
            info += f"{name.title()} CSV:\n"
            info += f"  - Path: {csv_info['path']}\n"
            info += f"  - Shape: {csv_info['shape']}\n"
            info += f"  - Columns: {csv_info['columns']}\n"
            info += f"  - Sample data:\n{csv_info['dataframe'].head(2)}\n\n"
        
        return info
    
    def is_applicable(self, query: str) -> bool:
        csv_keywords = [
            'csv', 'comma separated', 'data file', 'text file',
            'delimiter', 'tabular data', 'sales data', 'customer data'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in csv_keywords)