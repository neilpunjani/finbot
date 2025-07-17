#!/usr/bin/env python3
"""
Test that the agent doesn't crash with datetime serialization errors
"""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any

# Copy the serialization function from the agent
def serialize_for_json(obj):
    """Convert dataclass objects to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # Handle dataclass objects
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                result[key] = serialize_for_json(value)
            else:
                result[key] = value
        return result
    else:
        return obj

@dataclass
class Thought:
    content: str
    reasoning: str
    confidence: float
    timestamp: datetime

@dataclass
class Action:
    type: str
    parameters: Dict[str, Any]
    reasoning: str
    expected_outcome: str

@dataclass
class Observation:
    result: Any
    success: bool
    insights: List[str]
    confidence: float
    issues: List[str]

def test_agent_memory_serialization():
    """Test that agent memory with datetime objects can be serialized"""
    print("🔍 Testing agent memory serialization...")
    
    # Create realistic agent memory objects
    now = datetime.now()
    
    thoughts = [
        Thought(
            content="User wants Nova Scotia profit margin",
            reasoning="This requires financial data analysis",
            confidence=0.8,
            timestamp=now
        ),
        Thought(
            content="Found relevant Excel data",
            reasoning="Excel file contains financial records",
            confidence=0.9,
            timestamp=now
        )
    ]
    
    actions = [
        Action(
            type="excel_explorer",
            parameters={"focus": "Nova Scotia 2023 data"},
            reasoning="Excel contains financial data by region",
            expected_outcome="Find Nova Scotia 2023 revenue and costs"
        ),
        Action(
            type="csv_explorer", 
            parameters={"filter": "Nova Scotia", "year": 2023},
            reasoning="CSV may have additional operational data",
            expected_outcome="Find supporting operational metrics"
        )
    ]
    
    observations = [
        Observation(
            result="Found Nova Scotia 2023 data: Revenue $2.5M, Costs $2.1M",
            success=True,
            insights=["Profit margin = 16%", "Strong performance vs other regions"],
            confidence=0.9,
            issues=[]
        ),
        Observation(
            result="CSV data confirms operational efficiency",
            success=True,
            insights=["High productivity", "Low waste ratios"],
            confidence=0.8,
            issues=[]
        )
    ]
    
    # Test the specific JSON serialization patterns used in the agent
    try:
        # Test pattern 1: Action serialization (used in execute_action)
        action_json = json.dumps([serialize_for_json(a) for a in actions], indent=2)
        print("✅ Action serialization successful")
        
        # Test pattern 2: Observation serialization (used in execute_action)
        observation_json = json.dumps([serialize_for_json(o) for o in observations], indent=2)
        print("✅ Observation serialization successful")
        
        # Test pattern 3: Thought serialization (used in reflection)
        thought_json = json.dumps([serialize_for_json(t) for t in thoughts], indent=2)
        print("✅ Thought serialization successful")
        
        # Test pattern 4: Mixed serialization (used in final response)
        combined_json = json.dumps({
            "thoughts": [serialize_for_json(t) for t in thoughts],
            "actions": [serialize_for_json(a) for a in actions],
            "observations": [serialize_for_json(o) for o in observations]
        }, indent=2)
        print("✅ Combined serialization successful")
        
        # Verify we can parse it back
        parsed = json.loads(combined_json)
        print(f"✅ Parsed {len(parsed['thoughts'])} thoughts, {len(parsed['actions'])} actions, {len(parsed['observations'])} observations")
        
        # Check timestamp format
        timestamp = parsed['thoughts'][0]['timestamp']
        print(f"✅ Timestamp format: {timestamp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {str(e)}")
        return False

def test_error_scenarios():
    """Test edge cases that might cause serialization issues"""
    print("\n🔍 Testing error scenarios...")
    
    # Test with None values
    try:
        observation_with_none = Observation(
            result=None,
            success=True,
            insights=[],
            confidence=0.5,
            issues=[]
        )
        
        serialized = json.dumps([serialize_for_json(observation_with_none)], indent=2)
        print("✅ None value serialization successful")
        
    except Exception as e:
        print(f"❌ None value serialization failed: {str(e)}")
        return False
    
    # Test with empty lists
    try:
        empty_lists = json.dumps({
            "thoughts": [],
            "actions": [],
            "observations": []
        }, indent=2)
        print("✅ Empty lists serialization successful")
        
    except Exception as e:
        print(f"❌ Empty lists serialization failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Testing Agent Memory Serialization")
    print("=" * 50)
    
    test1 = test_agent_memory_serialization()
    test2 = test_error_scenarios()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print("=" * 50)
    print(f"✅ Agent memory serialization: {'Working' if test1 else 'Failed'}")
    print(f"✅ Error scenarios handled: {'Yes' if test2 else 'No'}")
    
    if test1 and test2:
        print("\n🎉 Agent should no longer crash with datetime serialization errors!")
        print("🚀 The agent can now properly serialize its memory for LLM prompts!")
    else:
        print("\n❌ Some serialization issues still exist")