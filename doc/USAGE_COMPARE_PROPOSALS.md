# Usage Guide: compare_proposals_parallel Updates

## Summary of Changes

The `compare_proposals_parallel` function has been updated with two major improvements:

### 1. Save LLM Data to `data/LLM_data` Folder
- Raw LLM outputs are now saved to `data/LLM_data/{timestamp}_{model_name}/` folder
- Each comparison is saved as: `statement_of_resch{IPTS_a}_vs_{IPTS_b}_llm_output.txt`
- Timestamp format: `YYYYMMDD_HHMMSS`
- This allows you to keep a record of all LLM queries and avoid re-querying when improving parsing logic

### 2. Separate LLM Querying from Output Processing
- **New function**: `parse_comparison_output(output)` - Parses raw LLM output into structured results
- **New function**: `process_llm_outputs(folder, llm_model, llm_data_folder)` - Processes saved LLM outputs
- The `compare_proposals_parallel` function now has a `process_outputs` parameter (default: `True`)

## Usage Examples

### Example 1: Standard Usage (Query + Process)
```python
from compare_proposal import compare_proposals_parallel

# This will query LLM and immediately process the outputs
IPTSs = [34014, 34962, 34123]
folder = "data/2025B"
llm_model = "google/gemini-2.5-flash-preview-09-2025"

llm_data_folder = compare_proposals_parallel(IPTSs, folder, llm_model)
# Output: data/LLM_data/20251102_123456_google_gemini-2.5-flash-preview-09-2025/
```

### Example 2: Query LLM Only (Skip Processing)
```python
from compare_proposal import compare_proposals_parallel

# This will only query LLM and save raw outputs, skip processing
IPTSs = [34014, 34962, 34123]
folder = "data/2025B"
llm_model = "google/gemini-2.5-flash-preview-09-2025"

llm_data_folder = compare_proposals_parallel(IPTSs, folder, llm_model, process_outputs=False)
# Output: data/LLM_data/20251102_123456_google_gemini-2.5-flash-preview-09-2025/
```

### Example 3: Reprocess Existing LLM Outputs
```python
from compare_proposal import process_llm_outputs

# After improving the parsing logic, reprocess saved LLM outputs
folder = "data/2025B"
llm_model = "google/gemini-2.5-flash-preview-09-2025"
llm_data_folder = "data/LLM_data/20251102_123456_google_gemini-2.5-flash-preview-09-2025"

process_llm_outputs(folder, llm_model, llm_data_folder)
# This will regenerate comparison files and CSV from saved LLM outputs
```

### Example 4: Improve Parsing Logic
```python
from compare_proposal import parse_comparison_output

# Test your improved parsing logic on a single output
with open("data/LLM_data/20251102_123456_google_gemini-2.5-flash-preview-09-2025/statement_of_resch34014_vs_34962_llm_output.txt", 'r') as f:
    raw_output = f.read()

parsed = parse_comparison_output(raw_output)
print(f"Winner: {parsed['winner']}")
print(f"Results: {parsed['results']}")
print(f"Comparison Table: {parsed['comparison_table']}")
print(f"Recommendation: {parsed['recommendation']}")
```

## Benefits

1. **Cost Savings**: Avoid re-querying expensive LLM APIs when improving parsing logic
2. **Debugging**: Easy to inspect raw LLM outputs when parsing fails
3. **Flexibility**: Can test different parsing strategies on the same LLM outputs
4. **Data Preservation**: Keep a historical record of all LLM queries with timestamps
5. **Reproducibility**: Can regenerate results from saved outputs at any time

## File Structure

```
data/
  LLM_data/
    20251102_123456_google_gemini-2.5-flash-preview-09-2025/
      statement_of_resch34014_vs_34962_llm_output.txt
      statement_of_resch34014_vs_34123_llm_output.txt
      statement_of_resch34962_vs_34123_llm_output.txt
  2025B/
    comparisons/
      google_gemini-2.5-flash-preview-09-2025/
        statement_of_resch34014_vs_34962_comparison_google_gemini-2.5-flash-preview-09-2025.txt
        ...
    proposal_comparisons_google_gemini-2.5-flash-preview-09-2025.csv
    proposal_elo_scores_google_gemini-2.5-flash-preview-09-2025.csv
```

## Workflow

1. **Query LLM**: Run `compare_proposals_parallel()` to query LLM and save raw outputs
2. **Process Outputs**: Outputs are automatically processed (or set `process_outputs=False` to skip)
3. **Inspect Results**: Check the comparison files and CSV
4. **If Parsing Fails**:
   - Improve the `parse_comparison_output()` function
   - Rerun `process_llm_outputs()` with the saved LLM data folder
   - No need to re-query LLM!
5. **Calculate Elo**: Use `calculate_Elo_score()` on the generated CSV
6. **Plot Results**: Use `plot_SRCMRating_vs_Elo()` to visualize correlations
