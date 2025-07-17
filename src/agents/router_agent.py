from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os

class RouterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.data_sources = {
            "sql": {
                "description": "SQL Database - Contains structured data in tables, good for complex queries, joins, aggregations",
                "keywords": ["database", "sql", "table", "join", "select", "records", "rows"],
                "entities": [],
                "metrics": [],
                "time_periods": []
            },
            "excel": {
                "description": "Excel Files - Financial data with worksheets including VW_PBI, Trial Balance, Accounts Receivable, Debt Schedule, COA",
                "keywords": ["excel", "spreadsheet", "worksheet", "workbook", "cell", "formula", "pivot", "chart", "xlsx", 
                           "trial balance", "accounts receivable", "debt schedule", "vw_pbi", "coa", "chart of accounts",
                           "financial", "accounting", "balance sheet", "income statement", "profit and loss", "revenue", "amount"],
                "entities": ["Canada", "Alberta", "Ontario", "Quebec", "British Columbia", "Nova Scotia", "Holding Company"],
                "metrics": ["revenue", "amount", "balance", "debt", "receivables", "assets", "liabilities"],
                "time_periods": ["2022", "2023", "2024", "yearly", "monthly", "quarterly"]
            },
            "csv": {
                "description": "CSV Files - Mining operations data including ESG, Production, Operational, and Workforce metrics",
                "keywords": ["csv", "comma separated", "data file", "text file", "delimiter", "tabular",
                           "esg", "environmental", "social", "governance", "ghg", "emissions", "co2", "water", "energy",
                           "production", "operational", "workforce", "mining", "ore", "metal", "commodity", "grade",
                           "recovery", "equipment", "utilization", "downtime", "tonnes", "headcount", "training hours",
                           "site", "pit", "gold", "copper", "zinc", "trifr", "ltifr", "contractor", "training",
                           "produced", "production", "output", "generated"],
                "entities": ["Canada", "Alberta", "Ontario", "Quebec", "British Columbia", "Nova Scotia", "Holding Company"],
                "metrics": ["training hours", "headcount", "ghg emissions", "water use", "energy", "equipment utilization", 
                           "downtime", "tonnes moved", "ore processed", "grade", "recovery rate", "metal produced", "trifr", "ltifr"],
                "time_periods": ["2022", "2023", "2024", "daily", "monthly", "yearly"],
                "sites": ["North Pit", "South Pit", "Mill A", "Mill B"],
                "commodities": ["Gold", "Copper", "Zinc", "Nickel"],
                "natural_language_patterns": [
                    "gold produced", "copper produced", "zinc produced", "nickel produced",
                    "gold production", "copper production", "zinc production", "nickel production",
                    "how much gold", "how much copper", "how much zinc", "how much nickel",
                    "gold recovery", "copper recovery", "zinc recovery", "nickel recovery",
                    "gold grade", "copper grade", "zinc grade", "nickel grade"
                ]
            },
            "email": {
                "description": "Outlook Emails - Contains email messages, good for communication analysis, finding correspondence",
                "keywords": ["email", "emails", "inbox", "message", "outlook", "mail", "sender", "recipient", "subject", "attachment"],
                "entities": [],
                "metrics": [],
                "time_periods": []
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
        """Determine which data source should handle the query using comprehensive pattern matching"""
        
        # Score each data source based on multiple factors
        scores = {}
        query_lower = query.lower()
        
        for source, info in self.data_sources.items():
            score = 0
            matched_elements = {
                "keywords": [],
                "entities": [],
                "metrics": [],
                "time_periods": [],
                "sites": [],
                "commodities": []
            }
            
            # Score keywords
            for keyword in info["keywords"]:
                import re
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    keyword_score = len(keyword.split()) * 2 if len(keyword.split()) > 1 else 1
                    score += keyword_score
                    matched_elements["keywords"].append(keyword)
            
            # Score entities
            for entity in info.get("entities", []):
                if entity.lower() in query_lower:
                    score += 2
                    matched_elements["entities"].append(entity)
            
            # Score metrics
            for metric in info.get("metrics", []):
                if metric.lower() in query_lower:
                    score += 3  # Metrics are highly specific
                    matched_elements["metrics"].append(metric)
            
            # Score time periods
            for period in info.get("time_periods", []):
                if period.lower() in query_lower:
                    score += 1
                    matched_elements["time_periods"].append(period)
            
            # Score sites (CSV specific)
            for site in info.get("sites", []):
                if site.lower() in query_lower:
                    score += 2
                    matched_elements["sites"].append(site)
            
            # Score commodities (CSV specific)
            for commodity in info.get("commodities", []):
                if commodity.lower() in query_lower:
                    score += 2
                    matched_elements["commodities"].append(commodity)
            
            # Score natural language patterns (CSV specific)
            for pattern in info.get("natural_language_patterns", []):
                if pattern.lower() in query_lower:
                    score += 4  # Higher score for natural language patterns
                    matched_elements["keywords"].append(pattern)
            
            scores[source] = {
                "score": score,
                "matched_elements": matched_elements
            }
        
        # Find the best matching source
        max_score = max(scores.values(), key=lambda x: x["score"])["score"]
        
        if max_score == 0:
            # No clear match, use LLM to decide
            return self._llm_route_decision(query)
        
        # Get the best source
        best_sources = [source for source, info in scores.items() if info["score"] == max_score]
        primary_source = best_sources[0]
        
        # Create detailed reasoning
        matched = scores[primary_source]["matched_elements"]
        reasoning_parts = []
        for element_type, elements in matched.items():
            if elements:
                reasoning_parts.append(f"{element_type}: {elements}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No specific matches found"
        
        return {
            "primary_source": primary_source,
            "alternative_sources": best_sources[1:] if len(best_sources) > 1 else [],
            "confidence": "high" if max_score >= 5 else "medium" if max_score >= 3 else "low",
            "reasoning": reasoning
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