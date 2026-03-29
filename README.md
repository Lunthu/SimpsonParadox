# Hidden Patterns Dashboard with Simpson's Paradox Detection

**A Python application for discovering hidden patterns in your data that standard correlation analysis misses.**

---

## Key Features

### Core Capabilities
- **Automatic Pattern Detection**: Finds Simpson's Paradox, confounding, interactions, and reversals
- **Interactive Dashboard**: Professional Plotly/Dash interface with multiple tabs
- **Visual + Text Integration**: Side-by-side visualizations with detailed explanations
- **Complete Analysis**: Shows ALL significant correlations and patterns (no artificial limits)
- **Production Ready**: Robust error handling for any dataset

### Analysis Features
- **Statistical Rigor**: Pearson correlation with p-value testing
- **Automatic Classification**: Metrics vs dimensions auto-detected
- **Multi-Format Support**: CSV, XLSX, Parquet, JSON
- **Pattern Prioritization**: Most important patterns highlighted first
- **Group-Level Analysis**: Correlation variation across dimension groups

---

## Quick Start

### Installation

```bash
# Navigate to project directory
cd correlation-dashboard

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

This creates `sample_data.csv` with **TWO engineered Simpson's Paradoxes**:
1. **Spend vs Quality_Score by Device** (positive in groups, negative overall)
2. **CTR vs Engagement_Rate by Traffic_Type** (positive in groups, negative overall)

### Run the Dashboard

```bash
python main.py sample_data.csv
```

Open your browser to: **http://localhost:8050**

### Analyze Your Own Data

```bash
python main.py your_data.csv
```

Optional parameters:
```bash
python main.py data.csv --correlation-threshold 0.7 --port 8080
```

## Project Structure

```
correlation-dashboard/
│
├── main.py                      # CLI entry point - start here
├── analyzer.py                  # Core correlation analysis engine
├── paradox_detector.py          # Simpson's Paradox & pattern detection
├── visualizer.py                # Standard visualizations (scatter, heatmap)
├── paradox_visualizer.py        # Specialized pattern visualizations
├── dashboard.py                 # Dash web interface
│
├── generate_sample_data.py      # Sample data generator with paradoxes
├── requirements.txt             # Python dependencies
└── setup.sh                     # Installation script
```

### Core Modules Description

**main.py** (~5KB)
- Command-line interface
- Argument parsing
- Dashboard launcher
- Entry point for all analyses

**analyzer.py** (~12KB)
- `CorrelationAnalyzer` class
- Data loading (CSV, XLSX, Parquet, JSON)
- Column classification (metrics vs dimensions)
- Correlation calculation with p-values
- Integration with pattern detection
- Summary statistics

**paradox_detector.py** (~20KB)
- `ParadoxDetector` class
- Simpson's Paradox detection algorithm
- Confounding variable identification
- Interaction effect analysis
- Subgroup reversal detection
- Pattern priority scoring
- Statistical significance testing

**visualizer.py** (~11KB)
- `CorrelationVisualizer` class
- Scatter plots with trend lines
- Correlation matrix heatmaps
- Pattern summary charts
- Distribution plots (box plots)
- Color palette management

**paradox_visualizer.py** (~15KB)
- `ParadoxVisualizer` class
- Simpson's Paradox side-by-side plots
- Interaction effect visualizations
- Confounding variable comparisons
- Subgroup reversal highlighting (RED)
- Subplot creation and annotations

**dashboard.py** (~18KB)
- `CorrelationDashboard` class
- Multi-tab Dash application
- Integrated card layouts (70% viz / 30% text)
- Responsive Bootstrap design
- Tab navigation and callbacks
- Pattern display organization

**generate_sample_data.py** (~7KB)
- Synthetic marketing dataset generator
- Engineered Simpson's Paradoxes
- Realistic business metrics
- Controllable parameters
- CSV output

---

## 🔧 Configuration & Usage

### Command Line Arguments

```bash
python main.py <filepath> [options]

Required:
  filepath                        Path to data file (CSV, XLSX, Parquet, JSON)

Optional:
  --correlation-threshold FLOAT   Minimum |r| to consider (default: 0.5)
  --significance-level FLOAT      P-value threshold (default: 0.05)
  --dimensions TEXT [TEXT ...]    Specific dimensions to analyze
  --metrics TEXT [TEXT ...]       Specific metrics to analyze
  --port INT                      Dashboard port (default: 8050)
  --debug                         Enable debug mode
```

### Examples

```bash
# Basic usage
python main.py data.csv

# Custom thresholds
python main.py data.csv --correlation-threshold 0.3 --significance-level 0.01

# Specific columns
python main.py data.csv --dimensions Region Channel --metrics Revenue Profit

# Custom port
python main.py data.csv --port 8080

# Debug mode
python main.py data.csv --debug
```

## Data Requirements

### Supported File Formats
- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **Parquet** (`.parquet`)
- **JSON** (`.json`)

### Data Structure

**Required:**
- Tabular format with rows and columns
- Column headers in first row
- Mix of categorical and numeric columns

**Column Types:**

**Dimensions (Categorical)** - for grouping:
- Text/string values
- Examples: Region, Channel, Product, Customer_Type
- Auto-detected: object dtype with <50% unique values

**Metrics (Numeric)** - for correlation:
- Numeric values (int or float)
- Examples: Revenue, Count, Rate, Score, Amount
- Auto-detected: numeric dtype or high-cardinality object


## 🔬 Detection Algorithms

### Simpson's Paradox Detection

**Algorithm Steps:**
1. Calculate overall correlation for metric pair (X, Y)
2. Group data by dimension (e.g., by Region)
3. Calculate correlation within each group
4. Check conditions:
   - All groups agree on sign (all positive OR all negative)
   - Overall has opposite sign
   - Both overall and groups are significant (p < 0.05)
   - Reversal magnitude > threshold
5. Calculate severity (high if magnitude > 0.5)

**Mathematical Condition:**
```
r_overall × sign(mean(r_groups)) < 0
AND all groups have same sign
AND |r_overall| > threshold
AND |mean(r_groups)| > threshold
```

### Confounding Variable Detection

**Algorithm Steps:**
1. Measure overall correlation strength (X, Y)
2. For each potential confounder Z:
   - Group by Z
   - Calculate correlation within each group
   - Average within-group correlations
3. Calculate attenuation: |r_overall| - |r_within_group|
4. Flag if strong overall but weak within-group

**Condition:**
```
|r_overall| > 0.4
AND |r_within_group| < 0.2
AND attenuation > 0.3
```

### Interaction Effect Detection

**Algorithm Steps:**
1. Calculate correlation for each group
2. Measure variation:
   - Standard deviation of correlations
   - Range (max - min)
3. Flag if variation is high

**Condition:**
```
std(group_correlations) > 0.3
OR range(group_correlations) > 0.6
```


### Subgroup Reversal Detection

**Algorithm Steps:**
1. Calculate overall correlation (must be significant)
2. For each group:
   - Calculate group correlation
   - Check if sign differs from overall
   - Verify significance
3. Return groups with reversed correlation

**Condition:**
```
sign(r_group) ≠ sign(r_overall)
AND p_group < 0.05
AND p_overall < 0.05
```

## Dependencies

### Required Python Packages

```txt
pandas>=2.0.0                   # Data manipulation and analysis
scipy>=1.10.0                   # Statistical functions (pearsonr, stats)
plotly>=5.14.0                  # Interactive visualizations
dash>=2.10.0                    # Web dashboard framework
dash-bootstrap-components>=1.4.0 # UI components and themes
numpy>=1.24.0                   # Numerical operations
openpyxl>=3.1.0                 # Excel file support (.xlsx)
```

### Python Version

**Minimum:** Python 3.8
**Recommended:** Python 3.10 or 3.11
**Tested on:** Python 3.10, 3.11


## Additional Resources

### Statistical Concepts

**Simpson's Paradox:**
- [Wikipedia Article](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
- Classic Example: UC Berkeley admission bias (1973)
- Book: "The Book of Why" by Judea Pearl

**Confounding Variables:**
- Essential in causal inference
- Key to experimental design
- Bradford Hill criteria for causation

**Interaction Effects:**
- Also called "moderation" or "effect modification"
- Common in psychology and social sciences
- Important for personalization

