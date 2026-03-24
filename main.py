"""
Main Entry Point
Run correlation analysis and launch dashboard
"""

import sys
import argparse
import os
from analyzer import CorrelationAnalyzer
from dashboard import CorrelationDashboard


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Correlation Analysis Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.csv
  python main.py data.csv --correlation-threshold 0.85 --sensitivity custom
  python main.py data.csv --sensitivity high --port 8080
  python main.py data.csv --dimensions "Traffic Type" "Campaign"
        """
    )
    
    parser.add_argument('filepath', help='Path to dataset file (CSV, XLSX, etc.)')
    parser.add_argument('--correlation-threshold', type=float, default=0.5,
                       help='Minimum correlation coefficient (default: 0.5)')
    parser.add_argument('--significance-level', type=float, default=0.05,
                       help='P-value threshold (default: 0.05)')
    parser.add_argument('--dimensions', nargs='*',
                       help='Specific dimensions to analyze (optional)')
    parser.add_argument('--metrics', nargs='*',
                       help='Specific metrics to analyze (optional)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Dashboard port (default: 8050)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug mode')
    parser.add_argument('--sensitivity', '-s', 
                       choices=['low', 'moderate', 'high', 'custom'],
                       default='moderate',
                       help='Pattern detection sensitivity: low (r≥0.5, p<0.01), moderate (r≥0.3, p<0.05), high (r≥0.2, p<0.10), custom (uses --correlation-threshold) (default: moderate)')
    parser.add_argument('--max-plot-points', type=int, default=5000,
                       help='Maximum points for visualizations (default: 5000)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 CORRELATION ANALYSIS DASHBOARD")
    print("=" * 60)
    
    # Step 1: Initialize analyzer
    print(f"\n📁 Loading data from: {args.filepath}")
    
    # Smart threshold alignment:
    # If user sets --correlation-threshold but not --sensitivity explicitly,
    # automatically use that threshold for pattern detection too
    custom_threshold = None
    detection_sensitivity = args.sensitivity
    
    # Check if correlation threshold is non-default
    if args.correlation_threshold != 0.5:
        # Use 'custom' mode to align detection with correlation threshold
        custom_threshold = args.correlation_threshold
        detection_sensitivity = 'custom'
        print(f"  🎯 Aligning pattern detection threshold to {args.correlation_threshold:.2f}")
    
    analyzer = CorrelationAnalyzer(
        filepath=args.filepath,
        correlation_threshold=args.correlation_threshold,
        significance_level=args.significance_level,
        detection_sensitivity=detection_sensitivity,
        max_plot_points=args.max_plot_points,
        custom_detection_threshold=custom_threshold
    )
    
    # Step 2: Load data
    try:
        analyzer.load_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Step 3: Identify columns
    print("\n🔍 Identifying metrics and dimensions...")
    analyzer.identify_columns(
        metric_patterns=args.metrics,
        dimension_patterns=args.dimensions
    )
    
    if not analyzer.metrics:
        print("❌ No metrics found in dataset")
        sys.exit(1)
    
    # Step 4: Calculate correlations
    print("\n📊 Calculating correlations...")
    analyzer.calculate_correlations()
    
    if not analyzer.correlations:
        print("⚠️  No significant correlations found")
        print("💡 Try lowering --correlation-threshold")
    
    # Step 5: Detect patterns for each dimension
    print("\n⚡ Detecting patterns...")
    for dimension in analyzer.dimensions:
        analyzer.detect_patterns(dimension)
    
    # Step 6: Detect hidden patterns (Simpson's Paradox, etc.)
    print("\n🔍 ADVANCED PATTERN DETECTION")
    if detection_sensitivity == 'custom':
        print(f"   Detection threshold: r ≥ {custom_threshold:.2f} (aligned with correlation threshold)")
    else:
        print(f"   Detection sensitivity: {detection_sensitivity.upper()}")
    print(f"   Detecting hidden patterns (Simpson's Paradox, confounding, etc.)...")
    analyzer.detect_hidden_patterns()
    
    # Step 7: Display summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    summary = analyzer.get_summary()
    print(f"Total Rows:          {summary['total_rows']:,}")
    print(f"Total Columns:       {summary['total_columns']}")
    print(f"Metrics:             {summary['metrics_count']}")
    print(f"Dimensions:          {summary['dimensions_count']}")
    print(f"Correlations Found:  {summary['correlations_found']}")
    print(f"Patterns Detected:   {summary['patterns_detected']}")
    print(f"\n🚨 HIDDEN PATTERNS:")
    print(f"Simpson's Paradoxes:  {summary.get('simpsons_paradox_count', 0)}")
    print(f"Confounding Vars:     {summary.get('confounding_count', 0)}")
    print(f"Interaction Effects:  {summary.get('interaction_count', 0)}")
    print(f"Subgroup Reversals:   {summary.get('subgroup_reversal_count', 0)}")
    
    if summary['top_correlation']:
        top = summary['top_correlation']
        print(f"\nStrongest Correlation: {abs(top['coefficient']):.3f} ({top['strength']})")
    
    # Display top correlations
    print("\n📈 TOP 5 CORRELATIONS:")
    print("-" * 60)
    for i, ((m1, m2), info) in enumerate(analyzer.get_top_correlations(n=5), 1):
        sig = "✓" if info['significant'] else "✗"
        print(f"{i}. {m1} ↔ {m2}")
        print(f"   Correlation: {info['coefficient']:+.3f} | {info['strength']} | Sig: {sig}")
    
    # Display top hidden patterns
    priority_patterns = analyzer.get_priority_hidden_patterns(n=3)
    if priority_patterns:
        print("\n🚨 TOP 3 HIDDEN PATTERNS:")
        print("-" * 60)
        for i, pattern in enumerate(priority_patterns, 1):
            print(f"{i}. {pattern['type'].upper().replace('_', ' ')}")
            desc_lines = pattern['description'].split('\n')
            print(f"   {desc_lines[0]}")
    
    print("\n" + "=" * 60)
    
    # Step 8: Launch dashboard
    print("\n🚀 Launching interactive dashboard...")
    dashboard = CorrelationDashboard(analyzer)
    dashboard.run(debug=not args.no_debug, port=args.port)


if __name__ == "__main__":
    main()
