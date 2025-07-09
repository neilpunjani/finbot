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
        self.agents = {}
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
            
            # Create individual agent for each CSV
            self.agents[name] = create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )
            
            loaded_files.append(f"{name}: {file_path}")
            
        except Exception as e:
            print(f"Warning: Could not load CSV file {file_path}: {e}")
    
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
            
            # Use the pandas agent for intelligent query handling
            if relevant_csv in self.agents:
                agent = self.agents[relevant_csv]
                result = agent.run(question)
                return f"[{relevant_csv.title()} CSV Analysis]\n{result}"
            else:
                return "No suitable CSV file found for this query."
                
        except Exception as e:
            return f"Error querying CSV files: {str(e)}"
    
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