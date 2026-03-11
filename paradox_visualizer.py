"""
Paradox Visualization Module
Specialized visualizations for Simpson's Paradox and hidden patterns
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional


class ParadoxVisualizer:
    """Create visualizations specifically for paradoxes and hidden patterns"""
    
    def __init__(self, data: pd.DataFrame, max_plot_points: int = 1000, 
                 color_palette: Optional[List[str]] = None):
        """
        Initialize paradox visualizer
        
        Args:
            data: DataFrame with analysis data
            max_plot_points: Maximum points to show in visualizations
            color_palette: Custom color palette
        """
        self.data = data
        self.max_plot_points = max_plot_points
        self.color_palette = color_palette or px.colors.qualitative.Bold
    

    def _sample_for_plot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample data for visualization if too large.
        Maintains distribution for accurate visual representation.
        
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
        
        print(f"  📊 Sampling {self.max_plot_points:,} of {len(df):,} points for paradox visualization ({sample_fraction:.1%})")
        
        return sampled

    def visualize_simpsons_paradox(self, paradox: Dict) -> go.Figure:
        """
        Create visualization showing Simpson's Paradox
        
        Shows overall trend vs individual group trends
        """
        metric_x = paradox['metric_x']
        metric_y = paradox['metric_y']
        dimension = paradox['dimension']
        
        # Get full data for accurate calculations
        full_data = self.data[[metric_x, metric_y, dimension]].dropna()
        
        # Sample for visualization
        plot_data = self._sample_for_plot(full_data)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"By {dimension} (Group Trends)",
                "Overall (Aggregate Trend)"
            ),
            horizontal_spacing=0.15
        )
        
        # Left plot: Individual groups with trend lines
        for i, (group_name, group_df) in enumerate(plot_data.groupby(dimension)):
            # Group scatter
            fig.add_trace(
                go.Scatter(
                    x=group_df[metric_x],
                    y=group_df[metric_y],
                    mode='markers',
                    name=str(group_name),
                    marker=dict(size=8, opacity=0.6),
                    legendgroup=str(group_name),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Group trend line (calculated on full group data for accuracy)
            full_group_df = full_data[full_data[dimension] == group_name]
            if len(full_group_df) >= 2:
                try:
                    # Check for valid data (no NaN, sufficient variance)
                    if full_group_df[metric_x].nunique() > 1 and full_group_df[metric_y].nunique() > 1:
                        z = np.polyfit(full_group_df[metric_x], full_group_df[metric_y], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(full_group_df[metric_x].min(), full_group_df[metric_x].max(), 100)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=p(x_line),
                                mode='lines',
                                name=f'{group_name} trend',
                                line=dict(width=2),
                                legendgroup=str(group_name),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                    # Skip trend line if fitting fails
                    print(f"Warning: Could not fit trend line for {group_name}: {e}")
                    pass
        
        # Right plot: Overall trend
        fig.add_trace(
            go.Scatter(
                x=plot_data[metric_x],
                y=plot_data[metric_y],
                mode='markers',
                name='All Data',
                marker=dict(size=8, opacity=0.4, color='gray'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Overall trend line (use full data for accuracy)
        try:
            if full_data[metric_x].nunique() > 1 and full_data[metric_y].nunique() > 1:
                z_overall = np.polyfit(full_data[metric_x], full_data[metric_y], 1)
                p_overall = np.poly1d(z_overall)
                x_line_overall = np.linspace(full_data[metric_x].min(), full_data[metric_x].max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line_overall,
                        y=p_overall(x_line_overall),
                        mode='lines',
                        name='Overall trend',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=2
                )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Skip overall trend line if fitting fails
            print(f"Warning: Could not fit overall trend line: {e}")
            pass
        
        # Update layout
        overall_corr = paradox['overall_correlation']
        avg_group_corr = paradox['average_group_correlation']
        
        title_text = (
            f"🚨 Simpson's Paradox: {metric_x} vs {metric_y}<br>"
            f"<sub>Group correlations avg: {avg_group_corr:+.3f} | "
            f"Overall correlation: {overall_corr:+.3f} (REVERSED!)</sub>"
        )
        
        fig.update_layout(
            title=title_text,
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text=metric_x, row=1, col=1)
        fig.update_xaxes(title_text=metric_x, row=1, col=2)
        fig.update_yaxes(title_text=metric_y, row=1, col=1)
        fig.update_yaxes(title_text=metric_y, row=1, col=2)
        
        # Add annotation explaining the paradox
        fig.add_annotation(
            text=(
                f"⚠️ Each {dimension} group shows {'+' if avg_group_corr > 0 else '-'}ve trend<br>"
                f"But overall data shows {'+' if overall_corr > 0 else '-'}ve trend!"
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="red"),
            align="center"
        )
        
        return fig
    
    def visualize_interaction_effect(self, interaction: Dict) -> go.Figure:
        """
        Visualize interaction effect where relationship changes across groups
        """
        metric_x = interaction['metric_x']
        metric_y = interaction['metric_y']
        moderator = interaction['moderator']
        
        # Get full data for accurate calculations
        full_data = self.data[[metric_x, metric_y, moderator]].dropna()
        
        # Sample for visualization
        plot_data = self._sample_for_plot(full_data)
        
        # Create scatter with trend lines for each group
        fig = go.Figure()
        
        for group_name, group_df in plot_data.groupby(moderator):
            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=group_df[metric_x],
                    y=group_df[metric_y],
                    mode='markers',
                    name=str(group_name),
                    marker=dict(size=8, opacity=0.6),
                    legendgroup=str(group_name)
                )
            )
            
            # Trend line
            full_group_df = full_data[full_data[moderator] == group_name]
            if len(full_group_df) >= 2:
                try:
                    if full_group_df[metric_x].nunique() > 1 and full_group_df[metric_y].nunique() > 1:
                        z = np.polyfit(full_group_df[metric_x], full_group_df[metric_y], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(full_group_df[metric_x].min(), full_group_df[metric_x].max(), 100)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=p(x_line),
                                mode='lines',
                                name=f'{group_name} trend',
                                line=dict(width=3),
                                legendgroup=str(group_name),
                                showlegend=False
                            )
                        )
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"Warning: Could not fit trend line for {group_name}: {e}")
                    pass
        
        # Get correlation info
        group_corrs = interaction['group_correlations']
        strongest = interaction['strongest_group']
        weakest = interaction['weakest_group']
        
        title_text = (
            f"⚡ Interaction Effect: {metric_x} vs {metric_y}<br>"
            f"<sub>Moderated by {moderator} | "
            f"Strongest in {strongest}, Weakest in {weakest}</sub>"
        )
        
        fig.update_layout(
            title=title_text,
            xaxis_title=metric_x,
            yaxis_title=metric_y,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def visualize_confounding(self, confounder_info: Dict) -> go.Figure:
        """
        Visualize confounding variable effect
        """
        metric_x = confounder_info['metric_x']
        metric_y = confounder_info['metric_y']
        confounder = confounder_info['confounder']
        
        plot_data = self.data[[metric_x, metric_y, confounder]].dropna()
        
        # Create side-by-side comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Controlling for {confounder}",
                "Overall (Confounded)"
            )
        )
        
        # Left: By confounder groups
        for group_name, group_df in plot_data.groupby(confounder):
            fig.add_trace(
                go.Scatter(
                    x=group_df[metric_x],
                    y=group_df[metric_y],
                    mode='markers',
                    name=str(group_name),
                    marker=dict(size=8, opacity=0.6),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Right: Overall
        fig.add_trace(
            go.Scatter(
                x=plot_data[metric_x],
                y=plot_data[metric_y],
                mode='markers',
                name='All Data',
                marker=dict(size=8, opacity=0.4, color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Overall trend
        try:
            if plot_data[metric_x].nunique() > 1 and plot_data[metric_y].nunique() > 1:
                z = np.polyfit(plot_data[metric_x], plot_data[metric_y], 1)
                p = np.poly1d(z)
                x_line = np.linspace(plot_data[metric_x].min(), plot_data[metric_x].max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        name='Confounded trend',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=2
                )
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: Could not fit confounded trend line: {e}")
            pass
        
        overall_corr = confounder_info['overall_correlation']
        within_corr = confounder_info['within_group_correlation']
        
        title_text = (
            f"🔍 Confounding Variable: {confounder}<br>"
            f"<sub>Overall r={overall_corr:.3f} | Within-group r={within_corr:.3f} "
            f"(Attenuation: {confounder_info['attenuation']:.3f})</sub>"
        )
        
        fig.update_layout(
            title=title_text,
            height=500,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text=metric_x)
        fig.update_yaxes(title_text=metric_y)
        
        return fig
    
    def create_paradox_summary_dashboard(self, hidden_patterns: Dict) -> go.Figure:
        """
        Create comprehensive dashboard showing all hidden patterns
        """
        simpsons = hidden_patterns.get('simpsons_paradox', [])
        confounders = hidden_patterns.get('confounding', [])
        interactions = hidden_patterns.get('interactions', [])
        reversals = hidden_patterns.get('reversals', [])
        
        # Create summary bar chart
        pattern_counts = {
            "Simpson's<br>Paradox": len(simpsons),
            'Confounding<br>Variables': len(confounders),
            'Interaction<br>Effects': len(interactions),
            'Subgroup<br>Reversals': len(reversals)
        }
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(pattern_counts.keys()),
                y=list(pattern_counts.values()),
                marker_color=colors,
                text=list(pattern_counts.values()),
                textposition='auto',
                textfont=dict(size=16, color='white')
            )
        ])
        
        total = sum(pattern_counts.values())
        
        fig.update_layout(
            title=f"🔍 Hidden Patterns Detected: {total} Total",
            xaxis_title="Pattern Type",
            yaxis_title="Count",
            template='plotly_white',
            height=400,
            font=dict(size=14)
        )
        
        # Add annotations for each bar
        annotations = []
        for i, (pattern, count) in enumerate(pattern_counts.items()):
            if count > 0:
                annotations.append(
                    dict(
                        x=i,
                        y=count,
                        text=f"{count}",
                        showarrow=False,
                        font=dict(size=20, color='white', family='Arial Black'),
                        yshift=0
                    )
                )
        
        return fig
    
    def create_paradox_detail_cards(self, hidden_patterns: Dict, max_items: int = 5) -> List[Dict]:
        """
        Create detailed information cards for each paradox
        """
        cards = []
        
        # Simpson's Paradoxes
        for paradox in hidden_patterns.get('simpsons_paradox', [])[:max_items]:
            cards.append({
                'type': 'simpsons_paradox',
                'title': f"🚨 Simpson's Paradox",
                'subtitle': f"{paradox['metric_x']} vs {paradox['metric_y']}",
                'description': paradox['description'],
                'severity': paradox['severity'],
                'details': {
                    'Dimension': paradox['dimension'],
                    'Overall Correlation': f"{paradox['overall_correlation']:+.3f}",
                    'Avg Group Correlation': f"{paradox['average_group_correlation']:+.3f}",
                    'Reversal Magnitude': f"{paradox['reversal_magnitude']:.3f}"
                }
            })
        
        # Confounding Variables
        for conf in hidden_patterns.get('confounding', [])[:max_items]:
            cards.append({
                'type': 'confounding',
                'title': f"🔍 Confounding Variable",
                'subtitle': f"{conf['confounder']} confounds {conf['metric_x']}-{conf['metric_y']}",
                'description': conf['description'],
                'severity': 'high' if conf['attenuation'] > 0.5 else 'moderate',
                'details': {
                    'Overall Correlation': f"{conf['overall_correlation']:.3f}",
                    'Within-Group Correlation': f"{conf['within_group_correlation']:.3f}",
                    'Attenuation': f"{conf['attenuation']:.3f}"
                }
            })
        
        # Interaction Effects
        for inter in hidden_patterns.get('interactions', [])[:max_items]:
            cards.append({
                'type': 'interaction',
                'title': f"⚡ Interaction Effect",
                'subtitle': f"{inter['moderator']} moderates {inter['metric_x']}-{inter['metric_y']}",
                'description': inter['description'],
                'severity': 'high' if inter['correlation_range'] > 0.8 else 'moderate',
                'details': {
                    'Correlation Range': f"{inter['correlation_range']:.3f}",
                    'Strongest Group': f"{inter['strongest_group']} (r={inter['strongest_correlation']:.3f})",
                    'Weakest Group': f"{inter['weakest_group']} (r={inter['weakest_correlation']:.3f})"
                }
            })
        
        return cards

    def visualize_subgroup_reversal(self, reversal: Dict) -> go.Figure:
        """
        Visualize subgroup reversal where some groups show opposite correlation
        """
        metric_x = reversal['metric_x']
        metric_y = reversal['metric_y']
        dimension = reversal['dimension']
        
        # Full data for calculations
        full_data = self.data[[metric_x, metric_y, dimension]].dropna()
        plot_data = self._sample_for_plot(full_data)
        
        # Create figure with subplots: individual groups on left, overall on right
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"By {dimension} (Individual Groups)",
                "Overall (All Data)"
            ),
            horizontal_spacing=0.15
        )
        
        # Get reversed group names
        reversed_groups = {rg['group'] for rg in reversal['reversed_groups']}
        
        # Left plot: Individual groups with highlighting for reversed ones
        for i, (group_name, group_df) in enumerate(plot_data.groupby(dimension)):
            is_reversed = group_name in reversed_groups
            
            # Choose color: red for reversed, normal colors for others
            if is_reversed:
                color = 'red'
                marker_size = 10
                opacity = 0.8
            else:
                color = self.color_palette[i % len(self.color_palette)]
                marker_size = 8
                opacity = 0.6
            
            # Group scatter
            fig.add_trace(
                go.Scatter(
                    x=group_df[metric_x],
                    y=group_df[metric_y],
                    mode='markers',
                    name=f"{group_name}{' ⚠️ REVERSED' if is_reversed else ''}",
                    marker=dict(size=marker_size, opacity=opacity, color=color),
                    legendgroup=str(group_name),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Group trend line (use full group data for accuracy)
            full_group_df = full_data[full_data[dimension] == group_name]
            if len(full_group_df) >= 2:
                try:
                    if full_group_df[metric_x].nunique() > 1 and full_group_df[metric_y].nunique() > 1:
                        z = np.polyfit(full_group_df[metric_x], full_group_df[metric_y], 1)
                        x_line = np.linspace(full_group_df[metric_x].min(), full_group_df[metric_x].max(), 100)
                        p = np.poly1d(z)
                        
                        # Thicker line for reversed groups
                        line_width = 4 if is_reversed else 2
                        line_style = 'solid' if is_reversed else 'solid'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=p(x_line),
                                mode='lines',
                                name=f'{group_name} trend',
                                line=dict(width=line_width, color=color, dash=line_style),
                                legendgroup=str(group_name),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"Warning: Could not fit trend line for {group_name}: {e}")
                    pass
        
        # Right plot: Overall data and trend
        fig.add_trace(
            go.Scatter(
                x=plot_data[metric_x],
                y=plot_data[metric_y],
                mode='markers',
                name='All Data',
                marker=dict(size=8, opacity=0.4, color='gray'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Overall trend line (use full data for accuracy)
        try:
            if full_data[metric_x].nunique() > 1 and full_data[metric_y].nunique() > 1:
                z_overall = np.polyfit(full_data[metric_x], full_data[metric_y], 1)
                p_overall = np.poly1d(z_overall)
                x_line_overall = np.linspace(full_data[metric_x].min(), full_data[metric_x].max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line_overall,
                        y=p_overall(x_line_overall),
                        mode='lines',
                        name='Overall trend',
                        line=dict(color='green', width=3, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=2
                )
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: Could not fit overall trend line: {e}")
            pass
        
        # Update layout
        overall_corr = reversal['overall_correlation']
        
        title_text = (
            f"🔄 Subgroup Reversal: {metric_x} vs {metric_y}<br>"
            f"<sub>Overall: r={overall_corr:+.3f} | "
            f"{len(reversed_groups)} group(s) show opposite direction</sub>"
        )
        
        fig.update_layout(
            title=title_text,
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        fig.update_xaxes(title_text=metric_x, row=1, col=1)
        fig.update_xaxes(title_text=metric_x, row=1, col=2)
        fig.update_yaxes(title_text=metric_y, row=1, col=1)
        fig.update_yaxes(title_text=metric_y, row=1, col=2)
        
        # Add annotation
        fig.add_annotation(
            text=(
                f"⚠️ Red groups show REVERSED correlation<br>"
                f"(opposite direction from overall trend)"
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11, color="red"),
            align="center"
        )
        
        return fig
