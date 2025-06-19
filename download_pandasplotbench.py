from datasets import load_dataset
import os

print("Downloading PandasPlotBench dataset...")
dataset = load_dataset('JetBrains-Research/PandasPlotBench', split='test')
dataset.to_json('data/pandasplotbench/tasks.jsonl')
print(f'Downloaded {len(dataset)} tasks to data/pandasplotbench/tasks.jsonl') 