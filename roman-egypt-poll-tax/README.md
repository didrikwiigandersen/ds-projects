# Poll Tax Payment Patterns in Roman Egypt

## Project Overview

This project analyzes poll tax receipts from Roman Egypt (circa 50 BCE - 250 CE) to investigate temporal patterns in tax payment using statistical methods. The analysis examines whether poll tax payments were distributed uniformly throughout the year or concentrated in specific periods, providing insights into the economic and agricultural cycles of ancient Roman Egypt.

**Course**: CLST 1201: The Ancient Economy  
**Author**: Didrik Wiig-Andersen, W'26, SEAS'26

## Research Questions

1. Were poll tax payments distributed uniformly throughout the year?
2. If not uniform, were payments concentrated in fall months (September, October, November)?
3. Did payment patterns remain consistent across different time periods?

## Dataset

The dataset consists of poll tax receipt records from Roman Egypt, containing:
- **Time Period**: Approximately 50 BCE - 250 CE
- **Variables**: Receipt identifier, year, month, day, and era (CE/BCE)
- **Records**: Filtered dataset includes only records with complete month and year information

## Project Structure

```
roman-egypt-poll-tax/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── analysis/
│   └── roman_egypt_poll_tax_analysis.ipynb  # Main analysis notebook
├── data/
│   └── data_cleaned.xlsx             # Poll tax receipt data
└── papers/
    └── final_paper.pdf                # Final research paper
```

## Methodology

The analysis employs Chi-square goodness-of-fit tests to examine the distribution of poll tax payments across months:

1. **Test 1: Uniform Distribution** - Tests whether receipts are uniformly distributed across all 12 months
2. **Test 2: Fall Months Concentration** - Tests whether payments were concentrated in September, October, and November
3. **Test 3: Late Summer/Early Fall Concentration** - Tests whether payments were concentrated in July, August, and September
4. **Test 4: Temporal Stability** - Tests whether payment patterns changed significantly over time using 25-year intervals

All tests use a significance level of α = 0.05.

## Key Findings

### Test 1: Uniform Distribution
- **Result**: Rejected null hypothesis (p ≈ 0.002)
- **Conclusion**: Poll tax payments were **not** uniformly distributed throughout the year

### Test 2: Fall Months Concentration
- **Result**: Failed to reject null hypothesis (p ≈ 0.93)
- **Conclusion**: **No evidence** that payments were concentrated in fall months (September-November)

### Test 3: Late Summer/Early Fall Concentration
- **Result**: Rejected null hypothesis (p < 0.0001)
- **Conclusion**: Strong evidence that payments were concentrated in **July, August, and September**
- **Interpretation**: This likely corresponds to the harvest season when agricultural income was available

### Test 4: Temporal Stability
- **Result**: Failed to reject null hypothesis (p ≈ 0.34)
- **Conclusion**: Payment timing patterns remained relatively **consistent** from 50 BCE to 250 CE

## Installation

1. Navigate to the project directory:
```bash
cd roman-egypt-poll-tax
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. Navigate to the project directory:
```bash
cd roman-egypt-poll-tax
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

## Dependencies

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0
- Jupyter >= 1.0.0
- openpyxl >= 3.0.0 (for reading Excel files)

## Results Summary

The analysis reveals that poll tax payments in Roman Egypt were:
- **Non-uniformly distributed** across the year
- **Concentrated in late summer/early fall** (July-September), likely corresponding to harvest season
- **Stable over time** across the 300-year period studied

These findings suggest that tax collection was synchronized with agricultural cycles, when farmers had income from harvests to pay their taxes.

## Limitations

1. **Data Completeness**: Analysis limited to records with complete month and year information
2. **Temporal Coverage**: Dataset spans approximately 300 years, but coverage may vary by period
3. **Geographic Scope**: Analysis focuses on Roman Egypt; patterns may differ in other regions
4. **Missing Context**: Additional historical context about tax collection policies would strengthen interpretation

## Future Work

1. Geographic analysis: Examine regional variations in payment patterns
2. Socioeconomic factors: Investigate correlations with income levels or social status
3. Comparative analysis: Compare with other tax types or other regions of the Roman Empire
4. Advanced statistical methods: Apply time series analysis or clustering techniques

## References

See `papers/final_paper.pdf` for detailed academic discussion, historical context, and bibliography.
