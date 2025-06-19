# Baseline Evaluation Guide for Qwen-3-4B

This guide provides step-by-step instructions to obtain baseline scores for the untuned Qwen-3-4B model on MedCalc-Bench and PandasPlotBench.

## Prerequisites

### Environment Setup

1. **Python Version**: Use Python 3.8+ (recommended 3.9+)
2. **GPU**: Preferably A10G or similar for faster inference (2-3h wall-clock time)
3. **Dependencies**: Install requirements from `baseline_evaluation_requirements.txt`

```bash
# Create virtual environment (recommended)
python -m venv baseline_env
source baseline_env/bin/activate  # On Windows: baseline_env\Scripts\activate

# Install dependencies
pip install -r baseline_evaluation_requirements.txt
```

### Data Verification

Both datasets are already available in the `data/` directory:
- MedCalc-Bench: `data/medcalc_bench/dataset/test_data.csv`
- PandasPlotBench: `data/pandasplotbench/`

## 1. MedCalc-Bench Evaluation

### Expected Results
- **Metric**: Exact-match accuracy
- **Expected Baseline**: ~40-45% (Qwen-2.5-7B achieved 38% in MedHELM)
- **Common failure modes**: Unit conversion errors, picking wrong vitals

### Running the Evaluation

```bash
# Test setup first
python test_baseline_setup.py

# Run quick test with 100 samples
python evaluate_medcalc_bench.py --max_samples 100 --output medcalc_test_results.json

# Run full evaluation (takes 2-3 hours)
python evaluate_medcalc_bench.py --output medcalc_baseline_results.json
```

### Command Line Options

```bash
python evaluate_medcalc_bench.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --test_file "data/medcalc_bench/dataset/test_data.csv" \
    --output "medcalc_baseline_results.json" \
    --max_samples 1000  # Optional: limit for testing
```

### Evaluation Process

The script follows the exact specification:

1. **Prompt Template**: 
   ```
   ### Instruction:
   <calc>
   {patient_note}
   
   {question}
   ### Response:
   ```

2. **Model Settings**:
   - Temperature: 0.0
   - Max new tokens: 32
   - Model: Qwen-3-4B

3. **Post-processing**: Extract first numeric token that parses as float
4. **Scoring**: Exact match against "Ground Truth Answer" column

### Output Format

Results are saved as JSON with:
```json
{
  "summary": {
    "total_samples": 10454,
    "correct_predictions": 4200,
    "accuracy": 0.402,
    "accuracy_percent": 40.2
  },
  "results": [...]
}
```

## 2. PandasPlotBench Evaluation

### Expected Results
- **Metric**: Plot pass@1 (plot hash matches)
- **Expected Baseline**: ~25-30% (VisCoder-7B baseline is 34%)
- **Common failure modes**: Multi-axis plots, gradient-fill tasks

### Configuration

The config has been updated in `data/pandasplotbench/configs/config.yaml`:
```yaml
model_plot_gen:
  names: ["huggingface/Qwen/Qwen2.5-3B-Instruct"]
  parameters:
    temperature: 0.0
    max_tokens: 256
```

### Running the Evaluation

```bash
# Navigate to PandasPlotBench directory
cd data/pandasplotbench

# Install additional requirements if needed
pip install -r requirements.txt  # or use poetry

# Run the benchmark
python run_benchmark.py

# For limited testing (first 10 tasks)
python run_benchmark.py --limit 10
```

### Evaluation Process

The harness automatically:
1. Feeds each plotting task to the model
2. Runs generated code in a sandbox
3. Records pass@1 based on plot hash matching
4. Generates comprehensive statistics

### Output

Results are saved in `data/pandasplotbench/out_results/`:
- `results.json`: Detailed responses
- `benchmark_stat.jsonl`: Final statistics

## 3. Monitoring and Troubleshooting

### Common Issues

1. **Memory Issues**: 
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Use CPU offloading: `device_map="auto"`

2. **Dependency Conflicts**:
   - Use fresh virtual environment
   - Check Python version compatibility
   - Install specific transformers version: `pip install transformers==4.35.0`

3. **Model Loading Issues**:
   - Ensure sufficient disk space (>20GB for model)
   - Check internet connection for model download
   - Use Hugging Face token if needed

### Performance Monitoring

```bash
# Monitor GPU usage
watch nvidia-smi

# Monitor CPU and memory
htop

# Check disk space
df -h
```

### Progress Tracking

Both scripts include progress bars and periodic saving:
- MedCalc: Progress shown via tqdm
- PandasPlotBench: Built-in progress tracking

## 4. Expected Timeline

### MedCalc-Bench
- Setup: 15-30 minutes
- Test run (100 samples): 5-10 minutes
- Full evaluation: 2-3 hours on A10G

### PandasPlotBench
- Setup: 10-15 minutes  
- Full evaluation: 1-2 hours

## 5. Results Interpretation

### MedCalc-Bench Baseline
- **Target**: ~40-45% exact match
- **Analysis**: Check error patterns in failed cases
- **Common errors**: Unit conversions, vital sign selection

### PandasPlotBench Baseline
- **Target**: ~25-30% pass@1
- **Analysis**: Identify plot type failure patterns
- **Common errors**: Complex multi-axis plots, styling

## 6. Next Steps

After obtaining baselines:
1. **Record numbers** as yard-stick for QLoRA training
2. **Analyze error patterns** for targeted improvements
3. **Compare** with published benchmarks
4. **Document** specific failure modes for training data curation

## Troubleshooting

If you encounter issues:
1. Run `python test_baseline_setup.py` first
2. Check logs for specific error messages
3. Verify model and data paths
4. Ensure sufficient resources (GPU memory, disk space)
5. Try with smaller sample sizes first

The baseline numbers will serve as the foundation for measuring improvements from QLoRA fine-tuning. 