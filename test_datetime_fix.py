#!/usr/bin/env python3
"""
Test the datetime serialization fix
"""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any

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

def test_datetime_serialization():
    """Test that datetime objects are properly serialized"""
    print("🔍 Testing datetime serialization...")
    
    # Create test objects with datetime
    now = datetime.now()
    
    thought = Thought(
        content="Test thought",
        reasoning="Test reasoning",
        confidence=0.8,
        timestamp=now
    )
    
    action = Action(
        type="test_action",
        parameters={"key": "value"},
        reasoning="Test action reasoning",
        expected_outcome="Test outcome"
    )
    
    observation = Observation(
        result="Test result",
        success=True,
        insights=["Test insight"],
        confidence=0.9,
        issues=[]
    )
    
    # Test serialization
    thoughts = [thought]
    actions = [action]
    observations = [observation]
    
    try:
        # Test the new serialization
        serialized_thoughts = json.dumps([serialize_for_json(t) for t in thoughts], indent=2)
        serialized_actions = json.dumps([serialize_for_json(a) for a in actions], indent=2)
        serialized_observations = json.dumps([serialize_for_json(o) for o in observations], indent=2)
        
        print("✅ Serialization successful!")
        print(f"Thoughts JSON: {serialized_thoughts}")
        print(f"Actions JSON: {serialized_actions}")
        print(f"Observations JSON (first 100 chars): {serialized_observations[:100]}...")
        
        # Test that the serialized data is valid JSON
        parsed_thoughts = json.loads(serialized_thoughts)
        parsed_actions = json.loads(serialized_actions)
        parsed_observations = json.loads(serialized_observations)
        
        print("✅ Deserialization successful!")
        print(f"Parsed thought timestamp: {parsed_thoughts[0]['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {str(e)}")
        return False

def test_old_serialization():
    """Test the old serialization method to show it fails"""
    print("\n🔍 Testing old serialization method (should fail)...")
    
    from dataclasses import asdict
    
    now = datetime.now()
    thought = Thought(
        content="Test thought",
        reasoning="Test reasoning", 
        confidence=0.8,
        timestamp=now
    )
    
    try:
        # This should fail with datetime serialization error
        serialized = json.dumps([asdict(thought)], indent=2)
        print("❌ Old serialization unexpectedly succeeded!")
        return False
        
    except TypeError as e:
        print(f"✅ Old serialization correctly failed: {str(e)}")
        return True

if __name__ == "__main__":
    print("🚀 Testing Datetime Serialization Fix")
    print("=" * 50)
    
    test1 = test_datetime_serialization()
    test2 = test_old_serialization()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print("=" * 50)
    print(f"✅ New serialization: {'Working' if test1 else 'Failed'}")
    print(f"✅ Old serialization fails as expected: {'Yes' if test2 else 'No'}")
    
    if test1 and test2:
        print("\n🎉 Datetime serialization fix is working correctly!")
    else:
        print("\n❌ Datetime serialization fix needs attention")