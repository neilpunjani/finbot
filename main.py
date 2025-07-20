#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        # Try to set console to UTF-8
        os.system("chcp 65001 > nul")
        # Reconfigure stdout to use UTF-8 encoding
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # If that fails, try alternative approach
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

from dotenv import load_dotenv, find_dotenv
from src.agents.pure_workflow import PureAgenticWorkflow

# Force reload environment variables to avoid caching issues
load_dotenv(find_dotenv(), override=True, verbose=False)

def main():
    print("Initializing Adaptive ReAct Agent with Cross-Checking...")
    
    try:
        workflow = PureAgenticWorkflow()
        print("\n" + workflow.get_system_status())
        print("\n" + "="*60)
        print("🎯 Adaptive ReAct Agent - Discovery → Analysis → Cross-Check")
        print("="*60)
        print("🧠 Adaptive: Analyzes query complexity, selects optimal sheets (2-6)")
        print("🔄 ReAct: Cross-checks results across sheets for validation")
        print("🧮 Transparent: Shows exact calculations and verification")
        print("✅ Quality: Full dataset loading + data quality penalties")
        print("🎯 Fixed: No more 10-row limits, proper blank data detection") 
        print("Type 'help' for examples, 'status' for system info, or 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['help', 'examples']:
                print(workflow.get_available_commands())
                continue
                
            if user_input.lower() in ['status', 'system']:
                print(workflow.get_system_status())
                continue
                
            try:
                print("Processing your query...")
                response = workflow.process_query(user_input)
                print(f"\nAssistant: {response}\n")
                print("-" * 60 + "\n")
            except Exception as e:
                print(f"Error: {str(e)}\n")
                
    except Exception as e:
        print(f"❌ Failed to initialize adaptive ReAct agent with cross-checking: {str(e)}")
        print("\nPlease check your configuration:")
        print("1. Ensure all required environment variables are set in .env file")
        print("2. Verify Excel/CSV data source files exist and are accessible")
        print("3. Check OpenAI API key is valid for GPT-4o and GPT-4o-mini")
        print("4. Check that data files are in the correct directories")
        print("5. Ensure proper file permissions for data access")
        print("6. The adaptive ReAct features require pandas, openpyxl, and langchain")

if __name__ == "__main__":
    main()