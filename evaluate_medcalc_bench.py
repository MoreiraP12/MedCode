#!/usr/bin/env python3
"""
MedCalc-Bench Baseline Evaluation Script
Evaluates Qwen-3-4B model on medical calculation benchmark.
"""

import pandas as pd
import re
import json
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse

class MedCalcEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        """Initialize the evaluator with the specified model."""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = self.model.device if hasattr(self.model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Model loaded on device: {self.device}")
    
    def create_prompt(self, note: str, question: str) -> str:
        """Create the prompt template as specified."""
        prompt = f"### Instruction:\n<calc>\n{note}\n\n{question}\n### Response:\n"
        return prompt
    
    def extract_numeric_answer(self, text: str) -> Optional[float]:
        """Extract the first numeric token that parses as float or date."""
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Look for numbers (including decimals)
        numeric_patterns = [
            r'^\d+\.?\d*',  # Numbers at the start
            r'(?:^|\s)(\d+\.?\d*)(?:\s|$)',  # Numbers surrounded by whitespace
            r'(\d+\.?\d*)',  # Any numbers
        ]
        
        for pattern in numeric_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # Get the first captured group or the whole match
                    num_str = match.group(1) if match.groups() else match.group(0)
                    return float(num_str)
                except ValueError:
                    continue
        
        return None
    
    def generate_response(self, prompt: str, max_new_tokens: int = 20000) -> str:
        """Generate response using the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response
    
    def evaluate_single(self, row: dict) -> dict:
        """Evaluate a single test case."""
        note = row['Patient Note']
        question = row['Question']
        ground_truth = row['Ground Truth Answer']
        
        # Create prompt and generate response
        prompt = self.create_prompt(note, question)
        response = self.generate_response(prompt)
        
        # Extract numeric answer
        predicted_answer = self.extract_numeric_answer(response)
        
        # Check exact match
        is_correct = False
        if predicted_answer is not None:
            try:
                gt_float = float(ground_truth)
                is_correct = abs(predicted_answer - gt_float) < 1e-6
            except ValueError:
                # Ground truth might not be numeric
                is_correct = str(predicted_answer) == str(ground_truth)
        
        return {
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'response': response,
            'is_correct': is_correct
        }
    
    def evaluate_dataset(self, test_file: str, output_file: str = "medcalc_results.json", 
                        max_samples: Optional[int] = None) -> dict:
        """Evaluate the entire test dataset."""
        print(f"Loading test data from: {test_file}")
        df = pd.read_csv(test_file)
        
        if max_samples:
            df = df.head(max_samples)
            print(f"Evaluating first {max_samples} samples")
        
        print(f"Total samples to evaluate: {len(df)}")
        
        results = []
        correct_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            try:
                result = self.evaluate_single(row)
                result['row_id'] = idx
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'row_id': idx,
                    'predicted_answer': None,
                    'ground_truth': row['Ground Truth Answer'],
                    'response': "",
                    'is_correct': False,
                    'error': str(e)
                })
        
        # Calculate accuracy
        accuracy = correct_count / len(results) if results else 0
        
        # Summary statistics
        summary = {
            'total_samples': len(results),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'accuracy_percent': accuracy * 100
        }
        
        # Save results
        output_data = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nEvaluation Results:")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Correct predictions: {summary['correct_predictions']}")
        print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
        print(f"Results saved to: {output_file}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen model on MedCalc-Bench')
    parser.add_argument('--model', default='Qwen/Qwen3-4B', 
                       help='Model name or path')
    parser.add_argument('--test_file', default='data/medcalc_bench/dataset/test_data.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output', default='medcalc_baseline_results.json',
                       help='Output file for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MedCalcEvaluator(args.model)
    
    # Run evaluation
    summary = evaluator.evaluate_dataset(
        args.test_file, 
        args.output,
        args.max_samples
    )
    
    return summary

if __name__ == "__main__":
    main() 