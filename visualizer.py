"""
Visualization Module
Creates interactive scatter plots with dimension mapping
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class CorrelationVisualizer:
    """Create interactive visualizations for correlation analysis"""
    
    def __init__(self, data: pd.DataFrame, color_palette: Optional[List[str]] = None):
        """
        Initialize visualizer
        
        Args:
            data: DataFrame with analysis data
            color_palette: Custom color palette for dimensions
        """
        self.data = data
        self.color_palette = color_palette or px.colors.qualitative.Set2
        

    def _sample_for_plot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample data for visualization if too large, maintaining distribution.
        
        Args:
            df: DataFrame to potentially sample
            
        Returns:
            Sampled or original DataFrame
        """
        if len(df) <= self.max_plot_points:
            return df  # Small enough, use all data
        
        # Sample maintaining distribution
        sample_fraction = self.max_plot_points / len(df)
        sampled = df.sample(n=self.max_plot_points, random_state=42)
        
        print(f"  📊 Sampling {self.max_plot_points:,} of {len(df):,} points for visualization ({sample_fraction:.1%})")
        
        return sampled

    def create_scatter_plot(self, 
                           metric_x: str, 
                           metric_y: str,
                           dimension: str,
                           correlation_info: Dict,
                           highlight_patterns: Optional[List[Dict]] = None,
                           title: Optional[str] = None) -> go.Figure:
        """
        Create scatter plot with dimension mapping and pattern highlights
        
        Args:
            metric_x: X-axis metric
            metric_y: Y-axis metric
            dimension: Dimension for color mapping
            correlation_info: Correlation statistics
            highlight_patterns: Patterns to highlight on plot
            title: Custom plot title
        """
        # Clean data
        plot_data = self.data[[metric_x, metric_y, dimension]].dropna()
        
        # Create base scatter plot
        fig = px.scatter(
            plot_data,
            x=metric_x,
            y=metric_y,
            color=dimension,
            color_discrete_sequence=self.color_palette,
            opacity=0.7,
            hover_data={
                metric_x: ':.2f',
                metric_y: ':.2f',
                dimension: True
            }
        )
        
        # Add regression line
        if len(plot_data) >= 2:
            try:
                if plot_data[metric_x].nunique() > 1 and plot_data[metric_y].nunique() > 1:
                    z = np.polyfit(plot_data[metric_x], plot_data[metric_y], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_data[metric_x].min(), plot_data[metric_x].max(), 100)
                    
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='rgba(0,0,0,0.3)', dash='dash', width=2),
                        showlegend=True
                    ))
            except (np.linalg.LinAlgError, ValueError) as e:
                # Skip trend line if fitting fails
                print(f"Warning: Could not fit trend line for {metric_x} vs {metric_y}: {e}")
                pass
        
        # Highlight patterns if provided
        if highlight_patterns:
            self._add_pattern_highlights(fig, plot_data, metric_x, metric_y, 
                                        dimension, highlight_patterns)
        
        # Update layout
        corr_coef = correlation_info['coefficient']
        p_value = correlation_info['p_value']
        strength = correlation_info['strength']
        
        if title is None:
            title = (f"{metric_x} vs {metric_y}<br>"
                    f"<sub>Correlation: {corr_coef:.3f} ({strength}) | "
                    f"p-value: {p_value:.4f}</sub>")
        
        fig.update_layout(
            title=title,
            xaxis_title=metric_x,
            yaxis_title=metric_y,
            hovermode='closest',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def _add_pattern_highlights(self, 
                                fig: go.Figure,
                                data: pd.DataFrame,
                                metric_x: str,
                                metric_y: str,
                                dimension: str,
                                patterns: List[Dict]):
        """Add visual highlights for detected patterns"""
        for pattern in patterns:
            if pattern['type'] == 'outlier_group':
                # Highlight outlier group points
                outlier_group = pattern['group']
                outlier_data = data[data[dimension] == outlier_group]
                
                if len(outlier_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=outlier_data[metric_x],
                        y=outlier_data[metric_y],
                        mode='markers',
                        name=f'⚠️ {outlier_group} (Outlier)',
                        marker=dict(
                            size=12,
                            symbol='star',
                            line=dict(color='red', width=2)
                        ),
                        showlegend=True
                    ))
    
    def create_correlation_matrix(self, 
                                  correlations: Dict[Tuple[str, str], Dict],
                                  metrics: List[str]) -> go.Figure:
        """
        Create correlation matrix heatmap
        
        Args:
            correlations: Dictionary of correlations
            metrics: List of metric names
        """
        # Build correlation matrix
        n = len(metrics)
        matrix = np.eye(n)
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i != j:
                    key = (metric1, metric2) if (metric1, metric2) in correlations else (metric2, metric1)
                    if key in correlations:
                        matrix[i, j] = correlations[key]['coefficient']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=metrics,
            y=metrics,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_pattern_summary(self, patterns: List[Dict]) -> go.Figure:
        """
        Create visualization summarizing detected patterns
        
        Args:
            patterns: List of detected patterns
        """
        if not patterns:
            # Empty state
            fig = go.Figure()
            fig.add_annotation(
                text="No significant patterns detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=300, template='plotly_white')
            return fig
        
        # Count pattern types
        pattern_counts = {}
        for p in patterns:
            ptype = p['type']
            pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(pattern_counts.keys()),
                y=list(pattern_counts.values()),
                marker_color='indianred'
            )
        ])
        
        fig.update_layout(
            title=f"Pattern Detection Summary ({len(patterns)} patterns found)",
            xaxis_title="Pattern Type",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_dimension_distribution(self, 
                                     dimension: str,
                                     metric: str) -> go.Figure:
        """
        Create box plot showing metric distribution across dimension groups
        
        Args:
            dimension: Dimension to group by
            metric: Metric to analyze
        """
        fig = px.box(
            self.data,
            x=dimension,
            y=metric,
            color=dimension,
            color_discrete_sequence=self.color_palette,
            points='outliers'
        )
        
        fig.update_layout(
            title=f"{metric} Distribution by {dimension}",
            xaxis_title=dimension,
            yaxis_title=metric,
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_multi_scatter_grid(self,
                                 correlations: List[Tuple[Tuple[str, str], Dict]],
                                 dimension: str,
                                 max_plots: int = 6) -> go.Figure:
        """
        Create grid of scatter plots for top correlations
        
        Args:
            correlations: List of correlation pairs and their info
            dimension: Dimension for color mapping
            max_plots: Maximum number of plots to show
        """
        n_plots = min(len(correlations), max_plots)
        
        if n_plots == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlations to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate grid dimensions
        cols = 2
        rows = (n_plots + 1) // 2
        
        # Create subplots
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[
                f"{pair[0]} vs {pair[1]}<br>r={info['coefficient']:.2f}"
                for pair, info in correlations[:n_plots]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add scatter plots
        for idx, ((metric_x, metric_y), corr_info) in enumerate(correlations[:n_plots]):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            plot_data = self.data[[metric_x, metric_y, dimension]].dropna()
            
            # Add traces for each dimension group
            for group in plot_data[dimension].unique():
                group_data = plot_data[plot_data[dimension] == group]
                
                fig.add_trace(
                    go.Scatter(
                        x=group_data[metric_x],
                        y=group_data[metric_y],
                        mode='markers',
                        name=str(group),
                        legendgroup=str(group),
                        showlegend=(idx == 0),  # Show legend only for first plot
                        marker=dict(opacity=0.6)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=300 * rows,
            template='plotly_white',
            title_text="Top Correlations Overview"
        )
        
        return fig
