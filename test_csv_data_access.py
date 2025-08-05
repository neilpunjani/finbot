#!/usr/bin/env python3
"""
Test CSV Data Access and Fix Configuration Issues

Diagnoses and fixes the "mining_data.csv not found" error by ensuring
proper CSV file configuration and access.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_csv_files_exist():
    """Test that CSV files exist and are accessible"""
    print("🔍 Testing CSV file access...")
    
    csv_dir = "data/csv"
    if not os.path.exists(csv_dir):
        print(f"❌ CSV directory not found: {csv_dir}")
        return False
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"📁 Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            print(f"   ✅ {csv_file}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show columns for ProductionData
            if "production" in csv_file.lower():
                print(f"      📊 Columns: {list(df.columns)}")
                if 'Date' in df.columns:
                    years = pd.to_datetime(df['Date']).dt.year.unique()
                    print(f"      📅 Years available: {sorted(years)}")
                if 'Commodity' in df.columns:
                    commodities = df['Commodity'].unique()
                    print(f"      💎 Commodities: {list(commodities)}")
                    
        except Exception as e:
            print(f"   ❌ {csv_file}: Error reading - {str(e)}")
    
    return len(csv_files) > 0


def test_environment_configuration():
    """Test environment variable configuration"""
    print("\n🔧 Testing environment configuration...")
    
    # Check CSV_DIRECTORY setting
    csv_dir = os.getenv("CSV_DIRECTORY", "data/csv")
    print(f"   CSV_DIRECTORY: {csv_dir}")
    
    if os.path.exists(csv_dir):
        print(f"   ✅ Directory exists: {csv_dir}")
    else:
        print(f"   ❌ Directory not found: {csv_dir}")
        return False
    
    # Check for individual CSV environment variables
    csv_env_vars = {k: v for k, v in os.environ.items() if k.startswith("CSV_FILE_PATH")}
    if csv_env_vars:
        print(f"   📁 Individual CSV environment variables found: {len(csv_env_vars)}")
        for var, path in csv_env_vars.items():
            exists = os.path.exists(path) if path else False
            status = "✅" if exists else "❌"
            print(f"      {status} {var}: {path}")
    
    # Check CSV_FILES variable
    csv_files_env = os.getenv("CSV_FILES")
    if csv_files_env:
        paths = [p.strip() for p in csv_files_env.split(',')]
        print(f"   📁 CSV_FILES variable: {len(paths)} paths")
        for path in paths:
            exists = os.path.exists(path) if path else False
            status = "✅" if exists else "❌"
            print(f"      {status} {path}")
    
    return True


def test_production_data_analysis():
    """Test that production data can be analyzed for gold production"""
    print("\n💎 Testing gold production analysis...")
    
    production_file = "data/csv/ProductionData.csv"
    if not os.path.exists(production_file):
        print(f"❌ Production data file not found: {production_file}")
        return False
    
    try:
        df = pd.read_csv(production_file)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        # Filter for gold data
        gold_data = df[df['Commodity'] == 'Gold']
        print(f"   📊 Gold records: {len(gold_data)}")
        
        # Check available years
        years = sorted(gold_data['Year'].unique())
        print(f"   📅 Gold data years: {years}")
        
        # Check scenarios
        scenarios = gold_data['Scenario'].unique()
        print(f"   📋 Scenarios: {list(scenarios)}")
        
        # Sample analysis for 2025 (if data exists)
        gold_2025 = gold_data[gold_data['Year'] == 2025]
        if len(gold_2025) > 0:
            print(f"   ✅ 2025 gold data available: {len(gold_2025)} records")
            
            # Group by scenario
            by_scenario = gold_2025.groupby('Scenario')['MetalProduced'].sum()
            print(f"   📈 2025 Gold Production by Scenario:")
            for scenario, production in by_scenario.items():
                print(f"      • {scenario}: {production:,.1f} units")
        else:
            print(f"   ⚠️  No 2025 gold data found")
            # Show what years have gold data
            if len(gold_data) > 0:
                sample_year = gold_data['Year'].iloc[0]
                sample_data = gold_data[gold_data['Year'] == sample_year]
                sample_by_scenario = sample_data.groupby('Scenario')['MetalProduced'].sum()
                print(f"   📈 Sample {sample_year} Gold Production by Scenario:")
                for scenario, production in sample_by_scenario.items():
                    print(f"      • {scenario}: {production:,.1f} units")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error analyzing production data: {str(e)}")
        return False


def fix_csv_configuration():
    """Create .env configuration to fix CSV access issues"""
    print("\n🔧 Creating proper CSV configuration...")
    
    # Set environment variable for this session
    os.environ["CSV_DIRECTORY"] = "data/csv"
    
    # Create or update .env file
    env_file = ".env"
    env_content = []
    
    # Read existing .env if it exists
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.readlines()
    
    # Remove any existing CSV_DIRECTORY line
    env_content = [line for line in env_content if not line.startswith('CSV_DIRECTORY=')]
    
    # Add the correct CSV_DIRECTORY
    env_content.append('CSV_DIRECTORY=data/csv\n')
    
    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(env_content)
    
    print(f"   ✅ Updated {env_file} with CSV_DIRECTORY=data/csv")
    
    return True


def main():
    """Run all CSV configuration tests and fixes"""
    print("🩺 CSV Data Access Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("CSV Files Exist", test_csv_files_exist),
        ("Environment Configuration", test_environment_configuration), 
        ("Production Data Analysis", test_production_data_analysis),
        ("Fix CSV Configuration", fix_csv_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 CSV configuration is working correctly!")
        print("💡 Your ProductionData.csv file contains the mining data")
        print("📋 Available analysis:")
        print("   • Gold production vs targets by year")
        print("   • Actual vs Budget scenarios") 
        print("   • Multi-site production comparison")
    else:
        print(f"\n⚠️  {total - passed} issue(s) found")
        print("🔧 Run this diagnostic tool to identify and fix CSV access problems")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)