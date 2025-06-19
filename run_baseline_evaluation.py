#!/usr/bin/env python3
"""
Unified baseline evaluation runner for both MedCalc-Bench and PandasPlotBench
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
import argparse

def log_message(message, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def run_command(command, cwd=None, timeout=None):
    """Run command with proper error handling and live output streaming."""
    log_message(f"Running command: `{command}` in `{cwd or os.getcwd()}`")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait(timeout=timeout)

        full_output = "".join(output_lines)

        if process.returncode == 0:
            log_message("Command completed successfully")
            return True, full_output
        else:
            log_message(f"Command failed with code {process.returncode}", "ERROR")
            return False, full_output

    except subprocess.TimeoutExpired:
        log_message(f"Command timed out after {timeout} seconds, terminating process.", "ERROR")
        process.kill()
        remaining_output = process.stdout.read()
        print(remaining_output, end='')
        output_lines.append(remaining_output)
        return False, "".join(output_lines)
    except Exception as e:
        log_message(f"Command failed with exception: {e}", "ERROR")
        return False, str(e)

def test_setup():
    """Test if the setup is ready"""
    log_message("Testing baseline evaluation setup...")
    
    success, output = run_command("python test_baseline_setup.py")
    if success:
        log_message("Setup test passed")
        return True
    else:
        log_message("Setup test failed", "ERROR")
        print(output)
        return False

def run_medcalc_evaluation(max_samples=None, quick_test=False):
    """Run MedCalc-Bench evaluation"""
    log_message("Starting MedCalc-Bench evaluation...")
    
    if quick_test:
        max_samples = min(max_samples or 100, 100)
        output_file = "medcalc_quick_test_results.json"
        log_message(f"Running quick test with {max_samples} samples")
    else:
        output_file = "medcalc_baseline_results.json"
        log_message("Running full MedCalc-Bench evaluation (this may take 2-3 hours)")
    
    # Build command
    cmd = "python evaluate_medcalc_bench.py"
    cmd += f" --output {output_file}"
    if max_samples:
        cmd += f" --max_samples {max_samples}"
    
    # Set timeout based on sample size
    if quick_test:
        timeout = 1800  # 30 minutes for quick test
    else:
        timeout = 14400  # 4 hours for full evaluation
    
    start_time = time.time()
    success, output = run_command(cmd, timeout=timeout)
    end_time = time.time()
    
    if success:
        duration = end_time - start_time
        log_message(f"MedCalc-Bench evaluation completed in {duration:.1f} seconds")
        
        # Try to load and display results
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            summary = results['summary']
            log_message(f"MedCalc-Bench Results:")
            log_message(f"  - Total samples: {summary['total_samples']}")
            log_message(f"  - Correct predictions: {summary['correct_predictions']}")
            log_message(f"  - Accuracy: {summary['accuracy_percent']:.2f}%")
            
            return True, summary
        except Exception as e:
            log_message(f"Could not load results file: {e}", "ERROR")
            return True, None
    else:
        log_message("MedCalc-Bench evaluation failed", "ERROR")
        print(output)
        return False, None

def run_pandasplotbench_evaluation(limit=None):
    """Run PandasPlotBench evaluation"""
    log_message("Starting PandasPlotBench evaluation...")
    
    # Navigate to PandasPlotBench directory
    ppb_dir = "data/pandasplotbench"
    
    if not os.path.exists(ppb_dir):
        log_message(f"PandasPlotBench directory not found: {ppb_dir}", "ERROR")
        return False, None
    
    # Build command
    cmd = "python run_benchmark.py"
    if limit:
        cmd += f" --limit {limit}"
        log_message(f"Running with limit: {limit} tasks")
    else:
        log_message("Running full PandasPlotBench evaluation")
    
    timeout = 7200 if limit else 14400  # 2-4 hours
    
    start_time = time.time()
    success, output = run_command(cmd, cwd=ppb_dir, timeout=timeout)
    end_time = time.time()
    
    if success:
        duration = end_time - start_time
        log_message(f"PandasPlotBench evaluation completed in {duration:.1f} seconds")
        
        # Try to find and display results
        results_dir = os.path.join(ppb_dir, "out_results")
        stats_file = os.path.join(results_dir, "benchmark_stat.jsonl")
        
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_result = json.loads(lines[-1])
                        log_message(f"PandasPlotBench Results:")
                        for key, value in last_result.items():
                            log_message(f"  - {key}: {value}")
                        return True, last_result
            except Exception as e:
                log_message(f"Could not load results file: {e}", "ERROR")
        
        log_message("PandasPlotBench completed but could not parse results")
        return True, None
    else:
        log_message("PandasPlotBench evaluation failed", "ERROR")
        print(output)
        return False, None

def main():
    parser = argparse.ArgumentParser(description='Run baseline evaluations')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with limited samples')
    parser.add_argument('--medcalc-only', action='store_true',
                       help='Run only MedCalc-Bench evaluation')
    parser.add_argument('--plotbench-only', action='store_true',
                       help='Run only PandasPlotBench evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for MedCalc (for testing)')
    parser.add_argument('--plot-limit', type=int, default=None,
                       help='Limit for PandasPlotBench tasks (for testing)')
    parser.add_argument('--skip-setup-test', action='store_true',
                       help='Skip initial setup test')
    
    args = parser.parse_args()
    
    log_message("=== Baseline Evaluation Runner ===")
    
    # Test setup unless skipped
    if not args.skip_setup_test:
        if not test_setup():
            log_message("Setup test failed. Please fix issues before continuing.", "ERROR")
            return 1
    
    results = {}
    
    # Determine what to run
    run_medcalc = not args.plotbench_only
    run_plotbench = not args.medcalc_only
    
    # Run MedCalc-Bench
    if run_medcalc:
        success, medcalc_results = run_medcalc_evaluation(
            max_samples=args.max_samples,
            quick_test=args.quick_test
        )
        results['medcalc'] = {
            'success': success,
            'results': medcalc_results
        }
        
        if not success:
            log_message("MedCalc-Bench evaluation failed", "ERROR")
            if not run_plotbench:  # If this is the only evaluation, exit
                return 1
    
    # Run PandasPlotBench
    if run_plotbench:
        plot_limit = args.plot_limit
        if args.quick_test and plot_limit is None:
            plot_limit = 10
            
        success, plotbench_results = run_pandasplotbench_evaluation(limit=plot_limit)
        results['plotbench'] = {
            'success': success,
            'results': plotbench_results
        }
        
        if not success:
            log_message("PandasPlotBench evaluation failed", "ERROR")
    
    # Summary
    log_message("=== Evaluation Summary ===")
    
    if 'medcalc' in results:
        if results['medcalc']['success']:
            if results['medcalc']['results']:
                acc = results['medcalc']['results']['accuracy_percent']
                log_message(f"✓ MedCalc-Bench: {acc:.2f}% accuracy")
            else:
                log_message("✓ MedCalc-Bench: Completed (check output file)")
        else:
            log_message("✗ MedCalc-Bench: Failed")
    
    if 'plotbench' in results:
        if results['plotbench']['success']:
            log_message("✓ PandasPlotBench: Completed (check out_results/)")
        else:
            log_message("✗ PandasPlotBench: Failed")
    
    # Save combined results
    summary_file = "baseline_evaluation_summary.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        log_message(f"Summary saved to: {summary_file}")
    except Exception as e:
        log_message(f"Could not save summary: {e}", "ERROR")
    
    # Determine exit code
    all_success = all(
        result['success'] for result in results.values()
    )
    
    if all_success:
        log_message("🎉 All evaluations completed successfully!")
        return 0
    else:
        log_message("⚠ Some evaluations failed. Check logs above.", "ERROR")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 