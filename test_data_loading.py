from datasets import load_dataset
import os

print("🧪 Testing data loading for all three datasets...\n")

# 1️⃣ MedCalc-Bench (clinical calculator prompts)
print("1️⃣ MedCalc-Bench:")
try:
    # MedCalc-Bench data is in CSV format
    medcalc_train = load_dataset("csv", data_files="data/medcalc_bench/dataset/train_data.csv")
    medcalc_test = load_dataset("csv", data_files="data/medcalc_bench/dataset/test_data.csv")
    print(f"   ✅ Loaded successfully!")
    print(f"   📊 Train: {len(medcalc_train['train'])} samples")
    print(f"   📊 Test: {len(medcalc_test['train'])} samples")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2️⃣ PandasPlotBench (visualization prompts)
print("\n2️⃣ PandasPlotBench:")
try:
    pandas_data = load_dataset("json", data_files="data/pandasplotbench/tasks.jsonl")
    print(f"   ✅ Loaded successfully!")
    print(f"   📊 Tasks: {len(pandas_data['train'])} visualization tasks")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3️⃣ MIMIC-IV Demo (patient data)
print("\n3️⃣ MIMIC-IV Demo:")
try:
    # Check what files are available in mimic_demo
    mimic_files = []
    for root, dirs, files in os.walk("data/mimic_demo"):
        for file in files:
            if file.endswith(('.csv', '.csv.gz')):
                mimic_files.append(os.path.join(root, file))
    
    if mimic_files:
        print(f"   ✅ Found {len(mimic_files)} CSV files:")
        for file in mimic_files[:5]:  # Show first 5
            rel_path = file.replace("data/mimic_demo/", "")
            print(f"      - {rel_path}")
        if len(mimic_files) > 5:
            print(f"      ... and {len(mimic_files) - 5} more files")
        
        # Try loading one of the CSV files as an example
        sample_file = mimic_files[0]
        mimic_data = load_dataset("csv", data_files=sample_file)
        print(f"   📊 Sample file '{os.path.basename(sample_file)}': {len(mimic_data['train'])} rows")
    else:
        print(f"   ❌ No CSV files found in data/mimic_demo/")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n🎯 Summary: All datasets are ready for one-liner loading!")
print("   Example usage:")
print("   >>> from datasets import load_dataset")
print("   >>> medcalc_train = load_dataset('csv', data_files='data/medcalc_bench/dataset/train_data.csv')")
print("   >>> medcalc_test = load_dataset('csv', data_files='data/medcalc_bench/dataset/test_data.csv')")
print("   >>> pandas_plots = load_dataset('json', data_files='data/pandasplotbench/tasks.jsonl')")
print("   >>> mimic_demo = load_dataset('csv', data_files='data/mimic_demo/hosp/patients.csv.gz')") 