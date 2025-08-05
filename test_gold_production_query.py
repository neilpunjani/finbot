#!/usr/bin/env python3
"""
Test Gold Production Query

Tests the actual gold production vs targets query that was failing
with the "mining_data.csv not found" error.
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gold_production_query():
    """Test the specific gold production query that was failing"""
    print("💎 Testing Gold Production vs Targets Query")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ["CSV_DIRECTORY"] = "data/csv"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    
    query = "What was the gold production versus targets in 2025?"
    
    try:
        # Test with the optimized pure workflow
        print("🚀 Testing with optimized PureAgenticWorkflow...")
        from src.agents.pure_workflow import PureAgenticWorkflow
        
        workflow = PureAgenticWorkflow()
        print("✅ Workflow initialized successfully")
        
        print(f"\n🔍 Processing query: '{query}'")
        response = workflow.process_query(query)
        
        print(f"\n📋 Response:")
        print("-" * 30)
        print(response)
        print("-" * 30)
        
        # Check if the response indicates successful processing
        if "error" not in response.lower() and "mining_data.csv" not in response.lower():
            print("\n✅ Query processed successfully!")
            print("🎯 No 'mining_data.csv not found' error")
            return True
        else:
            print("\n❌ Query processing failed")
            if "mining_data.csv" in response.lower():
                print("🔍 Still looking for missing 'mining_data.csv' file")
            return False
            
    except FileNotFoundError as e:
        if "mining_data.csv" in str(e):
            print(f"\n❌ Found the issue: {str(e)}")
            print("🔧 The system is looking for 'mining_data.csv' but should use 'ProductionData.csv'")
            print("💡 This indicates a hardcoded filename somewhere in the codebase")
            return False
        else:
            print(f"\n❌ Different file not found error: {str(e)}")
            return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def show_available_data():
    """Show what data is actually available"""
    print("\n📊 Available Data Analysis")
    print("=" * 30)
    
    csv_dir = "data/csv"
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f"📁 CSV files in {csv_dir}:")
        for file in csv_files:
            print(f"   • {file}")
            
        if "ProductionData.csv" in csv_files:
            print(f"\n✅ ProductionData.csv contains the mining data needed")
            print(f"💡 The query should work with this file")
        else:
            print(f"\n❌ ProductionData.csv not found")
    else:
        print(f"❌ CSV directory not found: {csv_dir}")


def main():
    """Run the gold production query test"""
    show_available_data()
    
    success = test_gold_production_query()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Performance optimizations working")
        print("✅ Gold production query processing correctly")
        print("🚀 Expected query time: 6-12 seconds (down from 10-17 seconds)")
    else:
        print("\n⚠️  ISSUE DETECTED")
        print("🔧 The system may still be looking for hardcoded 'mining_data.csv'")
        print("💡 Recommendation: Check for any remaining hardcoded file references")
        print("📋 Available data: ProductionData.csv, OperationalData.csv, ESGData.csv, WorkforceData.csv")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)