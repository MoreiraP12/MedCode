#!/usr/bin/env python3
"""
Quick test script to validate baseline evaluation setup
"""

import pandas as pd
import sys
import traceback

def test_medcalc_data_loading():
    """Test loading the MedCalc dataset"""
    try:
        print("Testing MedCalc-Bench data loading...")
        df = pd.read_csv('data/medcalc_bench/dataset/test_data.csv')
        print(f"✓ Successfully loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check first row
        if len(df) > 0:
            first_row = df.iloc[0]
            print(f"✓ First row patient note length: {len(str(first_row['Patient Note']))}")
            print(f"✓ First row question: {str(first_row['Question'])[:100]}...")
            print(f"✓ First row ground truth: {first_row['Ground Truth Answer']}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading MedCalc data: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    required_modules = [
        'pandas', 'numpy', 'tqdm', 're', 'json'
    ]
    
    optional_modules = [
        'transformers', 'torch'
    ]
    
    print("Testing required dependencies...")
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"✗ {module} not available")
            return False
    
    print("\nTesting optional dependencies...")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"⚠ {module} not available (needed for model inference)")
    
    return True

def test_prompt_template():
    """Test the prompt template creation"""
    try:
        print("\nTesting prompt template...")
        note = "Patient is a 65-year-old male with diabetes."
        question = "What is the patient's age?"
        
        prompt = f"### Instruction:\n<calc>\n{note}\n\n{question}\n### Response:\n"
        print(f"✓ Prompt template created successfully")
        print(f"✓ Prompt length: {len(prompt)} characters")
        return True
    except Exception as e:
        print(f"✗ Error creating prompt template: {e}")
        return False

def test_pandasplotbench_config():
    """Test PandasPlotBench configuration"""
    try:
        print("\nTesting PandasPlotBench setup...")
        
        # Check if config file exists and has been updated
        import yaml
        with open('data/pandasplotbench/configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        models = config['model_plot_gen']['names']
        if 'huggingface/Qwen/Qwen2.5-3B-Instruct' in models:
            print("✓ Config updated to use Qwen model")
        else:
            print(f"⚠ Config models: {models}")
        
        temp = config['model_plot_gen']['parameters']['temperature']
        print(f"✓ Temperature set to: {temp}")
        
        if 'max_tokens' in config['model_plot_gen']['parameters']:
            max_tokens = config['model_plot_gen']['parameters']['max_tokens']
            print(f"✓ Max tokens set to: {max_tokens}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking PandasPlotBench config: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== Baseline Evaluation Setup Test ===\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("MedCalc Data Loading", test_medcalc_data_loading), 
        ("Prompt Template", test_prompt_template),
        ("PandasPlotBench Config", test_pandasplotbench_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Summary ===")
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run baseline evaluations.")
    else:
        print("\n⚠ Some tests failed. Please address issues before running evaluations.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 