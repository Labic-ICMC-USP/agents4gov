#!/usr/bin/env python3
"""
Simple test script for CNPq/Lattes Navigator tool
"""
import json
from lattes_navigator import Tools

def test_basic_functionality():
    """Test the tool with sample data"""
    print("=" * 60)
    print("Testing CNPq/Lattes Navigator Tool")
    print("=" * 60)
    
    # Initialize the tool
    tool = Tools()
    print("\n✓ Tool initialized successfully")
    
    # Sample input (using anonymized data)
    researchers_data = [
        {"name": "Ana Silva Santos", "lattes_id": "1234567890123456"},
        {"name": "Carlos Oliveira Lima", "lattes_id": "2345678901234567"}
    ]
    
    # Convert to JSON string
    researchers_json = json.dumps(researchers_data)
    
    print(f"\n✓ Testing with {len(researchers_data)} researchers")
    print(f"  - {researchers_data[0]['name']}")
    print(f"  - {researchers_data[1]['name']}")
    
    # Call the tool
    print("\n⏳ Running analysis...")
    result = tool.analyze_researchers_coi(
        researchers_json=researchers_json,
        time_window=5,
        coi_rules_config='{"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}'
    )
    
    # Parse the result
    result_data = json.loads(result)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result_data['status'] == 'success':
        print(f"\n✓ Status: {result_data['status']}")
        print(f"✓ Execution date: {result_data['execution_metadata']['execution_date']}")
        print(f"✓ Time window: {result_data['execution_metadata']['time_window_years']} years")
        print(f"✓ Researchers analyzed: {result_data['execution_metadata']['num_researchers']}")
        
        # Show warnings
        print("\n📋 Warnings:")
        for researcher in result_data['researchers']:
            name = researcher['person']['name']
            warnings = researcher.get('warnings', [])
            if warnings:
                print(f"\n  {name}:")
                for warning in warnings:
                    print(f"    ⚠️  {warning}")
        
        # Show COI results
        print(f"\n🔍 COI Analysis:")
        coi_pairs = result_data['coi_matrix']['pairs']
        if coi_pairs:
            print(f"  Found {len(coi_pairs)} potential conflict(s) of interest")
            for pair in coi_pairs:
                print(f"\n  • {pair['a_name']} ↔ {pair['b_name']}")
                print(f"    Rules: {', '.join(pair['rules_triggered'])}")
                print(f"    Confidence: {pair['confidence']}")
        else:
            print("  No conflicts of interest detected")
        
        # Show summary
        print(f"\n📊 Summary:")
        print(f"  {result_data['summary_text']}")
        
        # Save full result
        output_file = "test_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to: {output_file}")
        
    else:
        print(f"\n❌ Error: {result_data['error_type']}")
        print(f"   Message: {result_data['message']}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return result_data

if __name__ == "__main__":
    try:
        result = test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

