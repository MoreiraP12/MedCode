#!/usr/bin/env python3
"""
Multi-task Dataset Preprocessing Pipeline

This script implements the walkthrough for turning three raw datasets 
(MedCalc-Bench, PandasPlotBench, MIMIC-IV) into one shuffled, tag-aware JSONL file
for multi-task fine-tuning of Qwen-3-4B model.

Task tags: <calc> for calculation tasks, <plot> for plotting tasks
"""

import pandas as pd
import json
import random
import gzip
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split


class MultiTaskPreprocessor:
    """Preprocessor for creating multi-task fine-tuning dataset"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "prompts", calc_plot_ratio: float = 0.7):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.calc_plot_ratio = calc_plot_ratio
        self.output_dir.mkdir(exist_ok=True)
        
        # Common prompt schema
        self.instruction_template = """### Instruction:
{tag}
{task_input}

### Response:
{expected_answer}"""
    
    def load_medcalc_bench(self) -> List[Dict[str, Any]]:
        """Load and process MedCalc-Bench dataset"""
        print("Loading MedCalc-Bench dataset...")
        
        csv_path = self.data_dir / "medcalc_bench" / "dataset" / "train_data.csv"
        df = pd.read_csv(csv_path)
        
        processed_data = []
        
        for _, row in df.iterrows():
            # Combine patient note and question as input
            patient_note = row['Patient Note'].strip()
            question = row['Question'].strip()
            
            # Create task input for <calc> tag
            task_input = f"{patient_note}\n\n{question}"
            
            # Use ground truth answer - keep numeric answer and optionally reasoning
            ground_truth = str(row['Ground Truth Answer'])
            
            # Create the final prompt
            prompt = self.instruction_template.format(
                tag="<calc>",
                task_input=task_input,
                expected_answer=ground_truth
            )
            
            processed_data.append({
                "input": prompt.split("### Response:")[0].strip(),
                "output": "### Response:\n" + ground_truth,
                "task": "calc"
            })
        
        print(f"Processed {len(processed_data)} MedCalc-Bench examples")
        return processed_data
    
    def load_pandasplotbench(self) -> List[Dict[str, Any]]:
        """Load and process PandasPlotBench dataset"""
        print("Loading PandasPlotBench dataset...")
        
        jsonl_path = self.data_dir / "pandasplotbench" / "tasks.jsonl"
        processed_data = []
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= 100:  # Limit for demo purposes
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Extract relevant fields
                    plot_description = data.get('task__plot_description', '').strip()
                    data_csv = data.get('data_csv', '').strip()
                    ground_truth_code = data.get('code_plot', '').strip()
                    
                    if not all([plot_description, data_csv, ground_truth_code]):
                        continue
                    
                    # Create CSV preview (first few lines)
                    csv_lines = data_csv.split('\n')[:5]  # First 5 lines
                    csv_peek = "CSV_PEEK:\n" + '\n'.join(f"{i},{line}" for i, line in enumerate(csv_lines) if line.strip())
                    
                    # Create task input for <plot> tag
                    task_input = f"{csv_peek}\nTask: {plot_description}"
                    
                    # Clean up the ground truth code (remove any reference to images)
                    clean_code = self.clean_plot_code(ground_truth_code)
                    
                    # Create the final prompt
                    prompt = self.instruction_template.format(
                        tag="<plot>",
                        task_input=task_input,
                        expected_answer=clean_code
                    )
                    
                    processed_data.append({
                        "input": prompt.split("### Response:")[0].strip(),
                        "output": "### Response:\n" + clean_code,
                        "task": "plot"
                    })
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
                print(f"Processed {len(processed_data)} PandasPlotBench examples")
        return processed_data

    def generate_synthetic_plot_data(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate synthetic plotting tasks for medical/scientific contexts"""
        print(f"Generating {num_examples} synthetic plotting examples...")
        
        processed_data = []
        
        # Define plot types and medical contexts
        plot_types = [
            "line", "scatter", "bar", "histogram", "box", "violin", 
            "heatmap", "area", "subplots", "dual_axis"
        ]
        
        medical_contexts = [
            {
                "name": "vital_signs",
                "variables": ["heart_rate", "blood_pressure", "temperature", "oxygen_saturation", "respiratory_rate"],
                "units": ["bpm", "mmHg", "°C", "%", "breaths/min"],
                "time_series": True
            },
            {
                "name": "lab_values", 
                "variables": ["glucose", "creatinine", "hemoglobin", "white_blood_cells", "platelets"],
                "units": ["mg/dL", "mg/dL", "g/dL", "cells/μL", "cells/μL"],
                "time_series": True
            },
            {
                "name": "medication_dosing",
                "variables": ["dose_amount", "plasma_concentration", "clearance", "half_life"],
                "units": ["mg", "μg/mL", "L/hr", "hours"],
                "time_series": True
            },
            {
                "name": "patient_demographics",
                "variables": ["age", "bmi", "diagnosis_count", "length_of_stay"],
                "units": ["years", "kg/m²", "count", "days"],
                "time_series": False
            },
            {
                "name": "imaging_metrics",
                "variables": ["tumor_size", "lesion_count", "contrast_enhancement", "volume"],
                "units": ["mm", "count", "HU", "cm³"],
                "time_series": False
            }
        ]
        
        for i in range(num_examples):
            # Choose random context and plot type
            context = random.choice(medical_contexts)
            plot_type = random.choice(plot_types)
            
            # Generate synthetic data
            if context["time_series"]:
                data_points = random.randint(10, 50)
                time_var = "time" if plot_type != "scatter" else "day"
                y_var = random.choice(context["variables"])
                y_unit = context["units"][context["variables"].index(y_var)]
                x_var = time_var  # Set x_var for consistency
                
                # Create CSV data
                csv_lines = [f"{time_var},{y_var}"]
                for j in range(data_points):
                    time_val = j if time_var == "day" else f"2024-01-{j+1:02d}"
                    if y_var == "heart_rate":
                        val = random.randint(60, 120) + random.gauss(0, 5)
                    elif y_var == "blood_pressure":
                        val = random.randint(90, 160) + random.gauss(0, 10)
                    elif y_var == "temperature":
                        val = round(36.5 + random.gauss(0, 0.8), 1)
                    elif y_var == "glucose":
                        val = random.randint(80, 200) + random.gauss(0, 15)
                    else:
                        val = round(100 + random.gauss(0, 20), 2)
                    
                    csv_lines.append(f"{time_val},{val:.1f}")
                
            else:
                # Categorical/distribution data
                data_points = random.randint(20, 100)
                x_var = random.choice(context["variables"])
                y_var = "count" if plot_type in ["histogram", "bar"] else random.choice(context["variables"])
                y_unit = "count" if y_var == "count" else context["units"][context["variables"].index(y_var) if y_var in context["variables"] else 0]
                
                csv_lines = [f"{x_var},{y_var}"]
                for j in range(data_points):
                    if x_var == "age":
                        x_val = random.randint(18, 90)
                    elif x_var == "bmi":
                        x_val = round(18 + random.expovariate(0.1), 1)
                    else:
                        x_val = round(random.gauss(50, 15), 1)
                    
                    if y_var == "count":
                        y_val = random.randint(1, 20)
                    else:
                        y_val = round(random.gauss(100, 30), 1)
                    
                    csv_lines.append(f"{x_val},{y_val}")
            
            # Create CSV peek
            csv_peek = "CSV_PEEK:\n" + '\n'.join(f"{i},{line}" for i, line in enumerate(csv_lines[:6]))
            
            # Generate task description
            if plot_type == "line":
                task_desc = f"Create a line plot showing {y_var} trends over {time_var}. Add appropriate title and axis labels."
            elif plot_type == "scatter":
                task_desc = f"Create a scatter plot of {x_var} vs {y_var}. Include trend line if correlation exists."
            elif plot_type == "bar":
                task_desc = f"Create a bar chart showing {y_var} distribution across {x_var} categories."
            elif plot_type == "histogram":
                task_desc = f"Create a histogram of {x_var} distribution. Use appropriate bin size."
            elif plot_type == "box":
                task_desc = f"Create a box plot showing {y_var} distribution grouped by {x_var} categories."
            elif plot_type == "heatmap":
                task_desc = f"Create a heatmap correlation matrix for the given variables."
            else:
                task_desc = f"Create a {plot_type} plot of the data with proper medical context styling."
            
            task_input = f"{csv_peek}\nTask: {task_desc}"
            
            # Generate appropriate matplotlib code
            code_lines = [
                "import matplotlib.pyplot as plt",
                "import pandas as pd",
                "df = pd.read_csv('data.csv')"
            ]
            
            if plot_type == "line":
                code_lines.extend([
                    f"plt.plot(df['{time_var}'], df['{y_var}'])",
                    f"plt.title('{y_var.replace('_', ' ').title()} Over Time')",
                    f"plt.xlabel('{time_var.replace('_', ' ').title()}')",
                    f"plt.ylabel('{y_var.replace('_', ' ').title()} ({y_unit})')"
                ])
            elif plot_type == "scatter":
                code_lines.extend([
                    f"plt.scatter(df['{x_var}'], df['{y_var}'])",
                    f"plt.title('{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}')",
                    f"plt.xlabel('{x_var.replace('_', ' ').title()}')",
                    f"plt.ylabel('{y_var.replace('_', ' ').title()}')"
                ])
            elif plot_type == "bar":
                code_lines.extend([
                    f"plt.bar(df['{x_var}'], df['{y_var}'])",
                    f"plt.title('{y_var.replace('_', ' ').title()} by {x_var.replace('_', ' ').title()}')",
                    f"plt.xlabel('{x_var.replace('_', ' ').title()}')",
                    f"plt.ylabel('{y_var.replace('_', ' ').title()}')"
                ])
            elif plot_type == "histogram":
                code_lines.extend([
                    f"plt.hist(df['{x_var}'], bins=20, alpha=0.7)",
                    f"plt.title('Distribution of {x_var.replace('_', ' ').title()}')",
                    f"plt.xlabel('{x_var.replace('_', ' ').title()}')",
                    "plt.ylabel('Frequency')"
                ])
            else:
                # Generic plot
                code_lines.extend([
                    f"plt.plot(df.iloc[:, 0], df.iloc[:, 1])",
                    "plt.title('Medical Data Visualization')",
                    "plt.xlabel('X Variable')",
                    "plt.ylabel('Y Variable')"
                ])
            
            code_lines.extend([
                "plt.grid(True, alpha=0.3)",
                "plt.tight_layout()",
                "plt.show()"
            ])
            
            clean_code = '\n'.join(code_lines)
            
            # Create the final prompt
            prompt = self.instruction_template.format(
                tag="<plot>",
                task_input=task_input,
                expected_answer=clean_code
            )
            
            processed_data.append({
                "input": prompt.split("### Response:")[0].strip(),
                "output": "### Response:\n" + clean_code,
                "task": "plot"
            })
        
        print(f"Generated {len(processed_data)} synthetic plotting examples")
        return processed_data

    def clean_plot_code(self, code: str) -> str:
        """Clean plotting code by removing problematic elements"""
        # Remove image references and other problematic elements
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines with image hashes or binary data
            if 'plots_gt' in line or 'iVBORw0KGgo' in line:
                continue
            # Remove excessive whitespace
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def load_mimic_demo(self) -> List[Dict[str, Any]]:
        """Load and process MIMIC-IV demo dataset (optional)"""
        print("Loading MIMIC-IV demo dataset...")
        
        # For demo purposes, we'll create a few synthetic examples
        # In practice, you'd implement the APACHE II / SOFA calculator logic
        
        processed_data = []
        
        # Example synthetic MIMIC data based on the walkthrough
        synthetic_examples = [
            {
                "age": 67, "sex": "M", "hr": 145, "map": 55, "fio2": 0.6, 
                "pao2": 60, "temp": 38.2, "creatinine": 2.1, "gcs": 6,
                "apache": 28, "sofa": 11
            },
            {
                "age": 72, "sex": "F", "hr": 120, "map": 65, "fio2": 0.4,
                "pao2": 80, "temp": 37.5, "creatinine": 1.8, "gcs": 8,
                "apache": 22, "sofa": 8
            },
            {
                "age": 58, "sex": "M", "hr": 110, "map": 70, "fio2": 0.3,
                "pao2": 90, "temp": 36.8, "creatinine": 1.2, "gcs": 12,
                "apache": 15, "sofa": 5
            }
        ]
        
        for example in synthetic_examples:
            # Create task input
            task_input = f"""AGE: {example['age']}  SEX: {example['sex']}
HR: {example['hr']}  MAP: {example['map']} mmHg
FiO2: {example['fio2']}   PaO2: {example['pao2']} mmHg
Temp: {example['temp']} °C
Creatinine: {example['creatinine']} mg/dL
GCS: {example['gcs']}
Compute APACHE II and SOFA. Reply as:
APACHE=<integer> ; SOFA=<integer>"""
            
            expected_answer = f"APACHE={example['apache']} ; SOFA={example['sofa']}"
            
            # Create the final prompt
            prompt = self.instruction_template.format(
                tag="<calc>",
                task_input=task_input,
                expected_answer=expected_answer
            )
            
            processed_data.append({
                "input": prompt.split("### Response:")[0].strip(),
                "output": "### Response:\n" + expected_answer,
                "task": "calc"
            })
        
        print(f"Processed {len(processed_data)} MIMIC-IV demo examples")
        return processed_data
    
    def create_balanced_dataset(self, calc_data: List[Dict], plot_data: List[Dict]) -> List[Dict]:
        """Create a balanced dataset with specified calc:plot ratio"""
        
        # Calculate target sizes
        total_plot = len(plot_data)
        target_calc = int(total_plot * self.calc_plot_ratio / (1 - self.calc_plot_ratio))
        
        # Sample data to achieve target ratio
        if len(calc_data) > target_calc:
            calc_data = random.sample(calc_data, target_calc)
        
        # Combine datasets
        combined_data = calc_data + plot_data
        
        # Shuffle the combined dataset
        random.shuffle(combined_data)
        
        print(f"Created balanced dataset: {len(calc_data)} calc + {len(plot_data)} plot = {len(combined_data)} total")
        print(f"Actual ratio - calc: {len(calc_data)/(len(combined_data)):.1%}, plot: {len(plot_data)/(len(combined_data)):.1%}")
        
        return combined_data
    
    def split_and_save(self, data: List[Dict], train_split: float = 0.9):
        """Split data and save to JSONL files"""
        
        # Stratified split to maintain task balance
        X = [item['input'] for item in data]
        y = [item['task'] for item in data]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=1-train_split, stratify=y, random_state=42
        )
        
        # Reconstruct full items
        train_data = []
        val_data = []
        
        train_indices = set(range(len(X_train)))
        
        train_map = {X_train[i]: i for i in range(len(X_train))}
        val_map = {X_val[i]: i for i in range(len(X_val))}
        
        for item in data:
            if item['input'] in train_map:
                train_data.append(item)
            else:
                val_data.append(item)
        
        # Save train data
        train_path = self.output_dir / "multi_train.jsonl"
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        # Save validation data
        val_path = self.output_dir / "multi_val.jsonl"
        with open(val_path, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(train_data)} training examples to {train_path}")
        print(f"Saved {len(val_data)} validation examples to {val_path}")
        
        return train_data, val_data
    
    def run_sanity_checks(self, train_data: List[Dict], val_data: List[Dict]):
        """Run sanity checks on the processed data"""
        print("\n=== SANITY CHECKS ===")
        
        # Check 1: Tag balance
        all_data = train_data + val_data
        task_counts = {}
        for item in all_data:
            task = item['task']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print(f"Task distribution: {task_counts}")
        calc_ratio = task_counts.get('calc', 0) / len(all_data)
        plot_ratio = task_counts.get('plot', 0) / len(all_data)
        print(f"Ratios - calc: {calc_ratio:.1%}, plot: {plot_ratio:.1%}")
        
        # Check 2: Max tokens per example (rough estimate)
        print("\nToken length analysis (rough estimate):")
        token_lengths = []
        for item in random.sample(all_data, min(200, len(all_data))):
            # Rough token count: ~4 chars per token
            input_tokens = len(item['input']) // 4
            output_tokens = len(item['output']) // 4
            total_tokens = input_tokens + output_tokens
            token_lengths.append(total_tokens)
        
        max_tokens = max(token_lengths)
        avg_tokens = sum(token_lengths) / len(token_lengths)
        print(f"Max tokens: {max_tokens}, Avg tokens: {avg_tokens:.0f}")
        
        if max_tokens > 1024:
            print(f"WARNING: {sum(1 for t in token_lengths if t > 1024)} examples exceed 1024 tokens")
        
        # Check 3: Leakage check (sample)
        print("\nLeakage check (sample):")
        calc_examples = [item for item in all_data if item['task'] == 'calc']
        if calc_examples:
            sample_calc = random.choice(calc_examples)
            input_text = sample_calc['input']
            output_text = sample_calc['output']
            
            # Look for numeric answers in input
            numbers_in_output = re.findall(r'\d+\.?\d*', output_text)
            numbers_in_input = re.findall(r'\d+\.?\d*', input_text)
            
            leaked_numbers = set(numbers_in_output) & set(numbers_in_input)
            if leaked_numbers:
                print(f"WARNING: Potential answer leakage detected: {leaked_numbers}")
            else:
                print("No obvious answer leakage detected in sample")
        
        print("\n=== SANITY CHECKS COMPLETE ===\n")
    
    def process_all(self):
        """Main processing pipeline"""
        print("Starting multi-task dataset preprocessing...")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Step 1: Load each dataset
        medcalc_data = self.load_medcalc_bench()
        plot_data = self.load_pandasplotbench()
        mimic_data = self.load_mimic_demo()
        
        # Generate synthetic plot data to increase dataset size
        synthetic_plot_data = self.generate_synthetic_plot_data(num_examples=1000)
        
        # Combine calc data (MedCalc + MIMIC)
        calc_data = medcalc_data + mimic_data
        
        # Combine plot data (PandasPlotBench + Synthetic)
        plot_data = plot_data + synthetic_plot_data
        
        # Step 2: Create balanced dataset
        balanced_data = self.create_balanced_dataset(calc_data, plot_data)
        
        # Step 3: Split and save
        train_data, val_data = self.split_and_save(balanced_data)
        
        # Step 4: Run sanity checks
        self.run_sanity_checks(train_data, val_data)
        
        print("Processing complete!")
        return train_data, val_data


def main():
    """Main function"""
    preprocessor = MultiTaskPreprocessor()
    train_data, val_data = preprocessor.process_all()
    
    print(f"\nDataset ready for training!")
    print(f"Load with: datasets.load_dataset('json', data_files='prompts/multi_train.jsonl')")


if __name__ == "__main__":
    main() 