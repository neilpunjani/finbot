#!/usr/bin/env python3
"""
Test script for Nova Scotia profit margin query
"""

import os
import sys
import json
from datetime import datetime
from unittest.mock import Mock, patch
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the required dependencies
class MockChatOpenAI:
    def __init__(self, model, temperature, api_key):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
    
    def invoke(self, messages):
        # Mock different responses based on the content
        content = messages[-1].content.lower()
        
        if "create a detailed plan" in content:
            # Planning response
            return MockResponse(json.dumps({
                "understanding": "User wants to find profit margin for Nova Scotia in 2023",
                "data_needed": "Financial data including revenue and costs for Nova Scotia in 2023",
                "approach": "Locate and analyze financial data from Excel or CSV sources",
                "steps": [
                    "Explore available data sources to find Nova Scotia data",
                    "Locate financial records for Nova Scotia in 2023",
                    "Extract revenue and cost data for profit margin calculation",
                    "Calculate profit margin and verify the result"
                ],
                "tool_selection_reasoning": "Excel exploration makes sense for financial data analysis",
                "potential_challenges": ["Data may be spread across multiple sheets", "Different fiscal year periods"],
                "success_criteria": "Accurate profit margin calculation with supporting data"
            }))
        
        elif "execute this step" in content:
            # Action planning response
            return MockResponse(json.dumps({
                "action_type": "excel_explorer",
                "reasoning": "Excel files typically contain financial data organized by regions and years",
                "parameters": {"focus": "Nova Scotia 2023 financial data"},
                "expected_outcome": "Find revenue and cost data for Nova Scotia in 2023",
                "confidence": 0.8
            }))
        
        elif "reflect on your progress" in content:
            # Reflection response
            return MockResponse(json.dumps({
                "progress_assessment": "Successfully found and analyzed Nova Scotia 2023 financial data",
                "key_findings": ["Revenue: $2.5M", "Costs: $2.1M", "Profit Margin: 16%"],
                "missing_information": [],
                "next_action": "finalize_answer",
                "reasoning": "All required data has been found and analyzed",
                "confidence": 0.9
            }))
        
        elif "analyze this dataframe" in content:
            # Data analysis response
            return MockResponse("""
            **Nova Scotia 2023 Financial Analysis**
            
            Based on the financial data found:
            - Total Revenue: $2,500,000
            - Total Costs: $2,100,000
            - Net Profit: $400,000
            - **Profit Margin: 16.0%**
            
            The profit margin is calculated as: (Net Profit / Total Revenue) × 100
            = ($400,000 / $2,500,000) × 100 = 16.0%
            
            This indicates healthy profitability for Nova Scotia operations in 2023.
            """)
        
        else:
            # Final response
            return MockResponse("""
            **Nova Scotia 2023 Profit Margin Analysis**
            
            After analyzing the available financial data, I found that Nova Scotia's profit margin for 2023 was **16.0%**.
            
            **Key Financial Metrics:**
            - Total Revenue: $2,500,000
            - Total Costs: $2,100,000
            - Net Profit: $400,000
            - Profit Margin: 16.0%
            
            **Calculation Method:**
            Profit Margin = (Net Profit ÷ Total Revenue) × 100
            = ($400,000 ÷ $2,500,000) × 100 = 16.0%
            
            **Context:**
            This profit margin indicates strong operational efficiency and profitability for Nova Scotia operations in 2023.
            """)

class MockResponse:
    def __init__(self, content):
        self.content = content

class MockDataFrame:
    def __init__(self):
        self.columns = ['Region', 'Year', 'Revenue', 'Costs', 'Profit']
        self.shape = (100, 5)
    
    def head(self, n=5):
        return self
    
    def to_dict(self, orient='records'):
        return [
            {'Region': 'Nova Scotia', 'Year': 2023, 'Revenue': 2500000, 'Costs': 2100000, 'Profit': 400000},
            {'Region': 'Ontario', 'Year': 2023, 'Revenue': 5000000, 'Costs': 4200000, 'Profit': 800000},
            {'Region': 'Alberta', 'Year': 2023, 'Revenue': 3200000, 'Costs': 2800000, 'Profit': 400000}
        ]

class MockExcelFile:
    def __init__(self, path):
        self.path = path
    
    @property
    def sheet_names(self):
        return ['Financial_Summary', 'Regional_Data', 'Quarterly_Reports']

# Apply patches
with patch('langchain_openai.ChatOpenAI', MockChatOpenAI), \
     patch('pandas.read_excel', lambda *args, **kwargs: MockDataFrame()), \
     patch('pandas.read_csv', lambda *args, **kwargs: MockDataFrame()), \
     patch('pandas.ExcelFile', MockExcelFile), \
     patch('os.path.exists', lambda path: True), \
     patch('os.listdir', lambda path: ['financial_data.csv', 'operations.csv']), \
     patch.dict(os.environ, {
         'OPENAI_API_KEY': 'test_key',
         'EXCEL_FILE_PATH': '/test/financial_data.xlsx',
         'CSV_DIRECTORY': '/test/csv'
     }):
    
    try:
        # Import the actual agent
        from agents.pure_agent import PureAgent
        
        print("🚀 Testing Pure Agent with Nova Scotia Query...")
        print("=" * 60)
        
        # Create and test the agent
        agent = PureAgent()
        
        # Test the specific query
        query = "What was the profit margin for Nova Scotia in 2023?"
        print(f"Query: {query}")
        print("-" * 60)
        
        # Process the query
        response = agent.solve(query)
        
        print("\n" + "=" * 60)
        print("🎯 AGENT RESPONSE:")
        print("=" * 60)
        print(response)
        
        print("\n" + "=" * 60)
        print("📊 AGENT MEMORY SUMMARY:")
        print("=" * 60)
        print(f"Thoughts: {len(agent.memory.thoughts)}")
        print(f"Actions: {len(agent.memory.actions)}")
        print(f"Observations: {len(agent.memory.observations)}")
        
        if agent.memory.actions:
            print(f"Actions taken: {[a.type for a in agent.memory.actions]}")
        
        if agent.memory.observations:
            success_count = sum(1 for o in agent.memory.observations if o.success)
            print(f"Successful observations: {success_count}/{len(agent.memory.observations)}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        traceback.print_exc()