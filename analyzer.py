"""
Correlation Analysis Engine
Handles data loading, correlation detection, and pattern identification
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings
from paradox_detector import ParadoxDetector

warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """Main class for correlation analysis and pattern detection"""
    
    def __init__(self, 
                 filepath: str, 
                 correlation_threshold: float = 0.5,
                 significance_level: float = 0.05,
                 dimensions: List[str] = None,
                 metrics: List[str] = None,
                 detection_sensitivity: str = 'moderate',
                 max_plot_points: int = 5000,
                 custom_detection_threshold: float = None):
        """
        Initialize the analyzer
        
        Args:
            filepath: Path to dataset
            correlation_threshold: Minimum correlation coefficient for display
            significance_level: P-value threshold
            dimensions: Optional dimension columns
            metrics: Optional metric columns
            detection_sensitivity: 'low', 'moderate', 'high', or 'custom'
            max_plot_points: Max points in visualizations
            custom_detection_threshold: Custom threshold when sensitivity='custom'
        """
        self.filepath = filepath
        self.correlation_threshold = correlation_threshold
        self.significance_level = significance_level
        self.detection_sensitivity = detection_sensitivity
        self.max_plot_points = max_plot_points
        self.custom_detection_threshold = custom_detection_threshold
        self.specified_dimensions = dimensions
        self.specified_metrics = metrics
        self.data = None
        self.metrics = []
        self.dimensions = []
        self.correlations = {}
        self.patterns = []
        self.paradox_detector = None


    def load_data(self) -> pd.DataFrame:
        """Load dataset from various file formats"""
        file_ext = self.filepath.lower().split('.')[-1]
        
        try:
            if file_ext == 'csv':
                self.data = pd.read_csv(self.filepath)
            elif file_ext in ['xlsx', 'xls']:
                self.data = pd.read_excel(self.filepath)
            elif file_ext == 'parquet':
                self.data = pd.read_parquet(self.filepath)
            elif file_ext == 'json':
                self.data = pd.read_json(self.filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            print(f"✓ Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            
            # Inform about sampling if dataset is large
            if len(self.data) > self.max_plot_points:
                print(f"  📊 Large dataset detected: Visualizations will sample {self.max_plot_points:,} points")
                print(f"  ✓ Correlations and trend lines use full {len(self.data):,} rows for accuracy")
            
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def identify_columns(self, 
                        metric_patterns: Optional[List[str]] = None,
                        dimension_patterns: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Automatically identify metrics (numeric) and dimensions (categorical)
        
        Args:
            metric_patterns: List of column name patterns to identify as metrics
            dimension_patterns: List of column name patterns to identify as dimensions
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Identify numeric columns as potential metrics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identify categorical/object columns as dimensions
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Apply pattern matching if provided
        if metric_patterns:
            self.metrics = [col for col in numeric_cols 
                          if any(pattern.lower() in col.lower() for pattern in metric_patterns)]
        else:
            self.metrics = numeric_cols
        
        if dimension_patterns:
            self.dimensions = [col for col in categorical_cols 
                             if any(pattern.lower() in col.lower() for pattern in dimension_patterns)]
        else:
            self.dimensions = categorical_cols
        
        print(f"✓ Identified {len(self.metrics)} metrics: {self.metrics}")
        print(f"✓ Identified {len(self.dimensions)} dimensions: {self.dimensions}")
        
        return self.metrics, self.dimensions
    
    def calculate_correlations(self) -> Dict[Tuple[str, str], Dict]:
        """
        Calculate correlations between all metric pairs
        Returns dictionary with correlation coefficient, p-value, and strength
        """
        if not self.metrics:
            raise ValueError("No metrics identified. Call identify_columns() first.")
        
        self.correlations = {}
        
        for i, metric1 in enumerate(self.metrics):
            for metric2 in self.metrics[i+1:]:
                # Remove NaN values for correlation calculation
                valid_data = self.data[[metric1, metric2]].dropna()
                
                if len(valid_data) < 3:
                    continue
                
                # Calculate Pearson correlation
                corr_coef, p_value = stats.pearsonr(valid_data[metric1], valid_data[metric2])
                
                # Only store if correlation exceeds threshold
                if abs(corr_coef) >= self.correlation_threshold:
                    strength = self._get_correlation_strength(abs(corr_coef))
                    
                    self.correlations[(metric1, metric2)] = {
                        'coefficient': corr_coef,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level,
                        'strength': strength,
                        'n_samples': len(valid_data)
                    }
        
        print(f"✓ Found {len(self.correlations)} significant correlations")
        return self.correlations
    
    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength"""
        if abs_corr >= 0.8:
            return 'Very Strong'
        elif abs_corr >= 0.6:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        else:
            return 'Weak'
    
    def detect_patterns(self, dimension_col: str) -> List[Dict]:
        """
        Detect patterns and anomalies within dimension groups
        
        Args:
            dimension_col: Dimension column to group by
        """
        patterns = []
        
        if dimension_col not in self.dimensions:
            print(f"Warning: {dimension_col} not in dimensions list")
            return patterns
        
        # For each correlated pair, check for patterns across dimension groups
        for (metric1, metric2), corr_info in self.correlations.items():
            grouped_data = self.data.groupby(dimension_col)[[metric1, metric2]].agg(['mean', 'std', 'count'])
            
            # Calculate group-specific correlations
            group_correlations = {}
            for group_name, group_df in self.data.groupby(dimension_col):
                if len(group_df) < 3:
                    continue
                    
                valid_group = group_df[[metric1, metric2]].dropna()
                if len(valid_group) >= 3:
                    group_corr, group_p = stats.pearsonr(valid_group[metric1], valid_group[metric2])
                    group_correlations[group_name] = {
                        'correlation': group_corr,
                        'p_value': group_p,
                        'n': len(valid_group)
                    }
            
            # Detect significant differences between groups
            if len(group_correlations) >= 2:
                corr_values = [v['correlation'] for v in group_correlations.values()]
                corr_std = np.std(corr_values)
                
                # If correlation varies significantly across groups, it's a pattern
                if corr_std > 0.3:
                    patterns.append({
                        'type': 'correlation_variation',
                        'metrics': (metric1, metric2),
                        'dimension': dimension_col,
                        'group_correlations': group_correlations,
                        'overall_correlation': corr_info['coefficient'],
                        'variation': corr_std,
                        'description': f"Correlation between {metric1} and {metric2} varies by {dimension_col}"
                    })
            
            # Detect outlier groups based on metric means
            for metric in [metric1, metric2]:
                means = grouped_data[(metric, 'mean')]
                std_all = means.std()
                mean_all = means.mean()
                
                for group in means.index:
                    z_score = abs((means[group] - mean_all) / std_all) if std_all > 0 else 0
                    
                    if z_score > 2:  # Significant outlier
                        patterns.append({
                            'type': 'outlier_group',
                            'metric': metric,
                            'dimension': dimension_col,
                            'group': group,
                            'value': means[group],
                            'mean': mean_all,
                            'z_score': z_score,
                            'description': f"{group} shows unusual {metric} (z-score: {z_score:.2f})"
                        })
        
        self.patterns.extend(patterns)
        print(f"✓ Detected {len(patterns)} patterns for dimension '{dimension_col}'")
        return patterns
    
    def get_top_correlations(self, n: int = 10) -> List[Tuple[Tuple[str, str], Dict]]:
        """Get top N correlations by absolute coefficient"""
        sorted_corr = sorted(self.correlations.items(), 
                           key=lambda x: abs(x[1]['coefficient']), 
                           reverse=True)
        return sorted_corr[:n]
    
    def get_summary(self) -> Dict:
        """Get analysis summary statistics"""
        return {
            'total_rows': len(self.data) if self.data is not None else 0,
            'total_columns': len(self.data.columns) if self.data is not None else 0,
            'metrics_count': len(self.metrics),
            'dimensions_count': len(self.dimensions),
            'correlations_found': len(self.correlations),
            'patterns_detected': len(self.patterns),
            'simpsons_paradox_count': len(self.hidden_patterns.get('simpsons_paradox', [])),
            'confounding_count': len(self.hidden_patterns.get('confounding', [])),
            'interaction_count': len(self.hidden_patterns.get('interactions', [])),
            'top_correlation': max(self.correlations.values(), 
                                 key=lambda x: abs(x['coefficient'])) if self.correlations else None
        }
    
    def detect_hidden_patterns(self) -> Dict:
        """
        Detect Simpson's Paradox and other hidden statistical patterns
        """
        if not self.metrics or not self.dimensions:
            print("⚠️  No metrics or dimensions to analyze")
            return {}
        
        print("\n" + "="*60)
        print("🔍 ADVANCED PATTERN DETECTION")
        print("="*60)
        
        # Initialize paradox detector
        self.paradox_detector = ParadoxDetector(
            data=self.data,
            metrics=self.metrics,
            dimensions=self.dimensions,
            significance_level=self.significance_level,
            detection_sensitivity=self.detection_sensitivity,
            custom_threshold=self.custom_detection_threshold
        )
        
        # Run all detection methods
        self.hidden_patterns = self.paradox_detector.get_all_patterns()
        
        # Print summary
        print(f"\n📊 HIDDEN PATTERNS FOUND:")
        print(f"   🚨 Simpson's Paradoxes: {len(self.hidden_patterns['simpsons_paradox'])}")
        print(f"   🔍 Confounding Variables: {len(self.hidden_patterns['confounding'])}")
        print(f"   ⚡ Interaction Effects: {len(self.hidden_patterns['interactions'])}")
        print(f"   🔄 Subgroup Reversals: {len(self.hidden_patterns['reversals'])}")
        print(f"   📈 Total Hidden Patterns: {self.hidden_patterns['total_patterns']}")
        
        return self.hidden_patterns
    
    def get_priority_hidden_patterns(self, n: int = 5) -> List[Dict]:
        """Get top N most important hidden patterns"""
        if not self.paradox_detector:
            return []
        
        return self.paradox_detector.get_priority_patterns(n)
