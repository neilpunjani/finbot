from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os

class RouterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.data_sources = {
            "sql": {
                "description": "SQL Database - Contains structured data in tables, good for complex queries, joins, aggregations",
                "keywords": ["database", "sql", "table", "join", "select", "count", "sum", "average", "records", "rows"]
            },
            "excel": {
                "description": "Excel Files - Contains spreadsheet data with multiple worksheets, good for financial data, reports",
                "keywords": ["excel", "spreadsheet", "worksheet", "workbook", "cell", "formula", "pivot", "chart", "xlsx"]
            },
            "csv": {
                "description": "CSV Files - Contains comma-separated tabular data, good for simple data analysis, customer data",
                "keywords": ["csv", "comma separated", "data file", "text file", "delimiter", "tabular", "customer", "customers", "sales data", "customer data", "sales", "revenue", "transactions"]
            },
            "email": {
                "description": "Outlook Emails - Contains email messages, good for communication analysis, finding correspondence",
                "keywords": ["email", "emails", "inbox", "message", "outlook", "mail", "sender", "recipient", "subject", "attachment"]
            }
        }
    
    def classify_query_type(self, query: str) -> str:
        """Classify if query is analytical or informational"""
        analytical_keywords = [
            "analyze", "analysis", "compare", "trend", "pattern", "correlation",
            "summary", "statistics", "average", "sum", "count", "total",
            "calculate", "compute", "aggregate", "group", "sort", "rank"
        ]
        
        informational_keywords = [
            "what", "who", "when", "where", "which", "how", "show", "list",
            "find", "search", "get", "retrieve", "display", "tell"
        ]
        
        query_lower = query.lower()
        
        analytical_score = sum(1 for keyword in analytical_keywords if keyword in query_lower)
        informational_score = sum(1 for keyword in informational_keywords if keyword in query_lower)
        
        if analytical_score > informational_score:
            return "analytical"
        elif informational_score > analytical_score:
            return "informational"
        else:
            return "mixed"
    
    def determine_data_source(self, query: str) -> Dict[str, Any]:
        """Determine which data source should handle the query"""
        
        # Score each data source based on keyword matching
        scores = {}
        query_lower = query.lower()
        
        for source, info in self.data_sources.items():
            score = 0
            matched_keywords = []
            
            for keyword in info["keywords"]:
                if keyword in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            scores[source] = {
                "score": score,
                "matched_keywords": matched_keywords
            }
        
        # Find the best matching source
        max_score = max(scores.values(), key=lambda x: x["score"])["score"]
        
        if max_score == 0:
            # No clear match, use LLM to decide
            return self._llm_route_decision(query)
        
        # Get the best source
        best_sources = [source for source, info in scores.items() if info["score"] == max_score]
        primary_source = best_sources[0]
        
        return {
            "primary_source": primary_source,
            "alternative_sources": best_sources[1:] if len(best_sources) > 1 else [],
            "confidence": "high" if max_score >= 2 else "medium",
            "reasoning": f"Matched keywords: {scores[primary_source]['matched_keywords']}"
        }
    
    def _llm_route_decision(self, query: str) -> Dict[str, Any]:
        """Use LLM to make routing decision when keyword matching fails"""
        
        source_descriptions = []
        for source, info in self.data_sources.items():
            source_descriptions.append(f"{source.upper()}: {info['description']}")
        
        routing_prompt = f"""
        Given the following data sources and a user query, determine which data source would be best to answer the query.
        
        Available data sources:
        {chr(10).join(source_descriptions)}
        
        User query: "{query}"
        
        Consider:
        1. What type of data would likely contain the answer
        2. What kind of analysis or information is being requested
        3. Which source would have the most relevant information
        
        Respond with only the data source name (sql, excel, csv, or email) and a brief reason.
        Format: SOURCE_NAME: reason
        """
        
        response = self.llm.invoke([HumanMessage(content=routing_prompt)])
        
        # Parse the response
        try:
            parts = response.content.split(":", 1)
            source = parts[0].strip().lower()
            reason = parts[1].strip() if len(parts) > 1 else "LLM recommendation"
            
            if source in self.data_sources:
                return {
                    "primary_source": source,
                    "alternative_sources": [],
                    "confidence": "medium",
                    "reasoning": reason
                }
        except:
            pass
        
        # Fallback to SQL if parsing fails
        return {
            "primary_source": "sql",
            "alternative_sources": [],
            "confidence": "low",
            "reasoning": "Fallback to SQL database"
        }
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Main routing method that determines data source and query type"""
        
        # Classify query type
        query_type = self.classify_query_type(query)
        
        # Determine data source
        routing_decision = self.determine_data_source(query)
        
        # Combine results
        result = {
            "query": query,
            "query_type": query_type,
            "data_source": routing_decision["primary_source"],
            "alternative_sources": routing_decision["alternative_sources"],
            "confidence": routing_decision["confidence"],
            "reasoning": routing_decision["reasoning"]
        }
        
        return result
    
    def get_routing_info(self) -> str:
        """Get information about available data sources and routing logic"""
        info = "Available data sources:\n\n"
        
        for source, details in self.data_sources.items():
            info += f"{source.upper()}:\n"
            info += f"  Description: {details['description']}\n"
            info += f"  Keywords: {', '.join(details['keywords'])}\n\n"
        
        info += "Query types:\n"
        info += "  - Analytical: Queries that require analysis, calculations, or aggregations\n"
        info += "  - Informational: Queries that seek specific information or data retrieval\n"
        info += "  - Mixed: Queries that combine both analytical and informational elements\n"
        
        return info