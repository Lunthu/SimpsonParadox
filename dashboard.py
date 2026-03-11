"""
Dashboard Application
Interactive Dash dashboard for correlation analysis
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from analyzer import CorrelationAnalyzer
from visualizer import CorrelationVisualizer
from paradox_visualizer import ParadoxVisualizer
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


class CorrelationDashboard:
    """Main dashboard application"""
    
    def __init__(self, analyzer: CorrelationAnalyzer, theme: str = 'FLATLY'):
        """
        Initialize dashboard
        
        Args:
            analyzer: CorrelationAnalyzer instance with loaded data
            theme: Bootstrap theme name
        """
        self.analyzer = analyzer
        self.visualizer = CorrelationVisualizer(analyzer.data)
        self.paradox_visualizer = ParadoxVisualizer(analyzer.data)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[getattr(dbc.themes, theme, dbc.themes.FLATLY)],
            suppress_callback_exceptions=True
        )
        
        self.app.title = "Correlation Analysis Dashboard"
        
        # Build layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
    
    def _create_layout(self):
        """Create dashboard layout"""
        
        # Summary statistics
        summary = self.analyzer.get_summary()
        top_correlations = self.analyzer.get_top_correlations(n=10)
        
        # Header
        header = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Correlation Analysis Dashboard", 
                           className="text-primary mb-3"),
                    html.P("Automated correlation detection and pattern analysis",
                          className="lead text-muted")
                ])
            ])
        ], fluid=True, className="bg-light py-4 mb-4")
        
        # Summary cards
        summary_cards = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(summary['total_rows'], className="text-primary"),
                            html.P("Total Rows", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(summary['metrics_count'], className="text-success"),
                            html.P("Metrics", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(summary['correlations_found'], className="text-info"),
                            html.P("Correlations Found", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(summary['patterns_detected'], className="text-warning"),
                            html.P("Patterns Detected", className="mb-0")
                        ])
                    ])
                ], width=3)
            ])
        ], fluid=True, className="mb-4")
        
        # Tabs for different views
        tabs = dbc.Container([
            dbc.Tabs([
                dbc.Tab(label="🚨 Hidden Patterns", tab_id="hidden", class_name="fw-bold"),
                dbc.Tab(label="📈 Correlation Matrix", tab_id="matrix"),
                dbc.Tab(label="🔍 Scatter Plots", tab_id="scatter"),
                dbc.Tab(label="⚡ Pattern Analysis", tab_id="patterns"),
                dbc.Tab(label="📊 Distribution Analysis", tab_id="distribution"),
            ], id="tabs", active_tab="hidden", className="mb-3"),
            
            html.Div(id="tab-content")
        ], fluid=True)
        
        # Footer
        footer = dbc.Container([
            html.Hr(),
            html.P("Built with Pandas, SciPy, Plotly, and Dash", 
                  className="text-center text-muted")
        ], fluid=True, className="mt-4")
        
        return html.Div([header, summary_cards, tabs, footer])
    
    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        
        @self.app.callback(
            Output("tab-content", "children"),
            [Input("tabs", "active_tab")]
        )
        def render_tab_content(active_tab):
            """Render content based on selected tab"""
            
            if active_tab == "hidden":
                return self._render_hidden_patterns_tab()
            
            elif active_tab == "matrix":
                return self._render_matrix_tab()
            
            elif active_tab == "scatter":
                return self._render_scatter_tab()
            
            elif active_tab == "patterns":
                return self._render_patterns_tab()
            
            elif active_tab == "distribution":
                return self._render_distribution_tab()
            
            return html.Div("Select a tab")
        
        @self.app.callback(
            Output("scatter-plots-container", "children"),
            [Input("scatter-dimension-dropdown", "value")]
        )
        def update_scatter_plots(dimension):
            """Update scatter plots when dimension changes"""
            if not dimension:
                return html.Div("Please select a dimension", 
                              className="alert alert-warning")
            return self._render_scatter_plots_for_dimension(dimension)
        
        @self.app.callback(
            Output("distribution-container", "children"),
            [Input("distribution-dimension-dropdown", "value")]
        )
        def update_distribution(dimension):
            """Update distribution plots when dimension changes"""
            if not dimension:
                return html.Div("Please select a dimension", 
                              className="alert alert-warning")
            return self._render_distribution_for_dimension(dimension)
    
    def _render_hidden_patterns_tab(self):
        """Render hidden patterns and Simpson's Paradox view"""
        hidden_patterns = self.analyzer.hidden_patterns
        
        if not hidden_patterns or hidden_patterns.get('total_patterns', 0) == 0:
            return dbc.Container([
                dbc.Alert([
                    html.H4("🔍 No Hidden Patterns Detected", className="alert-heading"),
                    html.P("No Simpson's Paradoxes or confounding variables found in this dataset."),
                    html.Hr(),
                    html.P("This could mean:"),
                    html.Ul([
                        html.Li("The data relationships are straightforward"),
                        html.Li("Sample size may be too small for detection"),
                        html.Li("Try analyzing with different dimension combinations")
                    ])
                ], color="info")
            ], fluid=True)
        
        
        # Summary figure
        summary_fig = self.paradox_visualizer.create_paradox_summary_dashboard(hidden_patterns)
        
        # Create integrated pattern sections (visualization + details combined)
        integrated_sections = []
        
        # Process Simpson's Paradoxes
        simpsons = hidden_patterns.get('simpsons_paradox', [])
        for i, paradox in enumerate(simpsons, 1):
            # Create visualization
            fig = self.paradox_visualizer.visualize_simpsons_paradox(paradox)
            
            # Create detail information
            severity_color = 'danger' if paradox.get('severity') == 'high' else 'warning'
            
            details = {
                'Dimension': paradox['dimension'],
                'Overall Correlation': f"{paradox['overall_correlation']:+.3f}",
                'Avg Group Correlation': f"{paradox['average_group_correlation']:+.3f}",
                'Reversal Magnitude': f"{paradox['reversal_magnitude']:.3f}",
                'Groups Analyzed': len(paradox['group_correlations'])
            }
            
            # Group correlations details
            group_details = []
            for group, info in paradox['group_correlations'].items():
                group_details.append(
                    html.Li(f"{group}: r={info['correlation']:+.3f} (n={info['n']})")
                )
            
            # Create enriched explanation combining description and key metrics
            enriched_explanation = (
                f"{paradox['description']}\n\n"
                f"📊 Analysis Details:\n"
                f"• Dimension analyzed: {paradox['dimension']}\n"
                f"• Overall correlation: {paradox['overall_correlation']:+.3f}\n"
                f"• Average group correlation: {paradox['average_group_correlation']:+.3f}\n"
                f"• Reversal magnitude: {paradox['reversal_magnitude']:.3f}\n"
                f"• Number of groups: {len(paradox['group_correlations'])}"
            )
            
            integrated_sections.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.Span("🚨 ", className=f"text-{severity_color}"),
                            f"Simpson's Paradox #{i}: {paradox['metric_x']} vs {paradox['metric_y']}"
                        ], className="mb-0")
                    ], className=f"bg-{severity_color} bg-opacity-10"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=fig)
                            ], width=12, lg=7),
                            dbc.Col([
                                html.Div([
                                    html.H5("📝 Pattern Analysis", className="mb-3"),
                                    html.Pre(
                                        enriched_explanation,
                                        style={'white-space': 'pre-wrap', 'font-size': '0.85rem', 'line-height': '1.5'},
                                        className="mb-3 p-3 bg-light rounded border"
                                    ),
                                    html.H6("🔢 Group Correlations", className="mb-2"),
                                    html.Ul(group_details, style={'font-size': '0.9rem'})
                                ], className="h-100")
                            ], width=12, lg=5)
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            )
        
        # Process Interaction Effects
        interactions = hidden_patterns.get('interactions', [])
        for i, interaction in enumerate(interactions, 1):
            # Create visualization
            fig = self.paradox_visualizer.visualize_interaction_effect(interaction)
            
            # Create enriched explanation combining description and key metrics
            enriched_explanation = (
                f"{interaction['description']}\n\n"
                f"📊 Analysis Details:\n"
                f"• Moderator variable: {interaction['moderator']}\n"
                f"• Correlation range: {interaction['correlation_range']:.3f}\n"
                f"• Correlation std dev: {interaction['correlation_std']:.3f}\n"
                f"• Strongest in {interaction['strongest_group']}: r={interaction['strongest_correlation']:+.3f}\n"
                f"• Weakest in {interaction['weakest_group']}: r={interaction['weakest_correlation']:+.3f}"
            )
            
            integrated_sections.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.Span("⚡ ", className="text-warning"),
                            f"Interaction Effect #{i}: {interaction['metric_x']} vs {interaction['metric_y']}"
                        ], className="mb-0")
                    ], className="bg-warning bg-opacity-10"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=fig)
                            ], width=12, lg=7),
                            dbc.Col([
                                html.Div([
                                    html.H5("📝 Pattern Analysis", className="mb-3"),
                                    html.Pre(
                                        enriched_explanation,
                                        style={'white-space': 'pre-wrap', 'font-size': '0.85rem', 'line-height': '1.5'},
                                        className="mb-3 p-3 bg-light rounded border"
                                    )
                                ], className="h-100")
                            ], width=12, lg=5)
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            )
        
        # Process Confounding Variables
        confounders = hidden_patterns.get('confounding', [])
        for i, conf in enumerate(confounders, 1):
            # Create visualization
            fig = self.paradox_visualizer.visualize_confounding(conf)
            
            # Create enriched explanation combining description and key metrics
            enriched_explanation = (
                f"{conf['description']}\n\n"
                f"📊 Analysis Details:\n"
                f"• Confounding variable: {conf['confounder']}\n"
                f"• Overall correlation: {conf['overall_correlation']:.3f}\n"
                f"• Within-group correlation: {conf['within_group_correlation']:.3f}\n"
                f"• Attenuation: {conf['attenuation']:.3f}\n"
                f"• Effect: Strong overall correlation weakens substantially when controlling for the confounder"
            )
            
            integrated_sections.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.Span("🔍 ", className="text-info"),
                            f"Confounding Variable #{i}: {conf['metric_x']} vs {conf['metric_y']}"
                        ], className="mb-0")
                    ], className="bg-info bg-opacity-10"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=fig)
                            ], width=12, lg=7),
                            dbc.Col([
                                html.Div([
                                    html.H5("📝 Pattern Analysis", className="mb-3"),
                                    html.Pre(
                                        enriched_explanation,
                                        style={'white-space': 'pre-wrap', 'font-size': '0.85rem', 'line-height': '1.5'},
                                        className="mb-3 p-3 bg-light rounded border"
                                    )
                                ], className="h-100")
                            ], width=12, lg=5)
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            )
        
        # Process Subgroup Reversals
        reversals = hidden_patterns.get('reversals', [])
        for i, reversal in enumerate(reversals, 1):
            # Create visualization
            fig = self.paradox_visualizer.visualize_subgroup_reversal(reversal)
            
            # Create enriched explanation
            enriched_explanation = (
                f"{reversal['description']}\n\n"
                f"📊 Analysis Details:\n"
                f"• Dimension: {reversal['dimension']}\n"
                f"• Overall correlation: {reversal['overall_correlation']:+.3f}\n"
                f"• Overall p-value: {reversal['overall_p_value']:.4f}\n"
                f"• Total groups analyzed: {len(reversal.get('all_groups', reversal['reversed_groups']))}\n"
                f"• Groups showing reversal: {len(reversal['reversed_groups'])}"
            )
            
            # Create list of reversed groups
            reversed_details = []
            for rg in reversal['reversed_groups']:
                reversed_details.append(
                    html.Li(
                        f"{rg['group']}: r={rg['correlation']:+.3f} (n={rg['n']}, p={rg['p_value']:.4f})",
                        className="text-danger fw-bold"
                    )
                )
            
            # Create card with side-by-side layout
            integrated_sections.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.Span("🔄 ", className="text-secondary"),
                            f"Subgroup Reversal #{i}: {reversal['metric_x']} vs {reversal['metric_y']}"
                        ], className="mb-0")
                    ], className="bg-secondary bg-opacity-10"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(figure=fig)
                            ], width=12, lg=7),
                            dbc.Col([
                                html.Div([
                                    html.H5("📝 Pattern Analysis", className="mb-3"),
                                    html.Pre(
                                        enriched_explanation,
                                        style={'white-space': 'pre-wrap', 'font-size': '0.85rem', 'line-height': '1.5'},
                                        className="mb-3 p-3 bg-light rounded border"
                                    ),
                                    html.H6("⚠️ Groups Showing Reversal", className="mb-2 mt-4 text-danger"),
                                    html.Ul(reversed_details, style={'font-size': '0.9rem'})
                                ], className="h-100")
                            ], width=12, lg=5)
                        ])
                    ])
                ], className="mb-4 shadow-sm")
            )
        
        return dbc.Container([
            dbc.Alert([
                html.Strong(f"🚨 Displaying ALL {hidden_patterns.get('total_patterns', 0)} hidden patterns"),
                html.Br(),
                html.Small(f"Simpson's Paradoxes: {len(simpsons)}, "
                          f"Confounding: {len(confounders)}, "
                          f"Interactions: {len(interactions)}, "
                          f"Reversals: {len(reversals)}")
            ], color="danger", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.H3("🚨 Hidden Patterns & Simpson's Paradoxes", className="mb-2"),
                    html.P(
                        "Each pattern below combines visualization with detailed explanation. "
                        "These counterintuitive patterns show where aggregate trends differ from group-specific trends.",
                        className="text-muted mb-4"
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=summary_fig)
                ])
            ], className="mb-4"),
            
            html.Div(integrated_sections)
        ], fluid=True)
    
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Top Correlations", className="mt-4"),
                    self._create_correlation_table()
                ])
            ])
        ], fluid=True)
    
    def _render_matrix_tab(self):
        """Render correlation matrix heatmap and table"""
        # Create correlation matrix heatmap
        matrix_fig = self.visualizer.create_correlation_matrix(
            self.analyzer.correlations,
            self.analyzer.metrics
        )
        
        # Create correlation table
        corr_data = []
        for (metric_x, metric_y), info in self.analyzer.get_top_correlations(n=100):
            corr_data.append({
                'Metric X': metric_x,
                'Metric Y': metric_y,
                'Correlation': f"{info['coefficient']:+.3f}",
                'P-value': f"{info['p_value']:.4f}",
                'Strength': info['strength'],
                'Significant': '✓' if info['significant'] else '✗'
            })
        
        import pandas as pd
        corr_df = pd.DataFrame(corr_data)
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("📈 Correlation Matrix", className="mb-3"),
                    html.P(
                        "Overview of correlations between all metric pairs. "
                        "Darker colors indicate stronger correlations.",
                        className="text-muted"
                    )
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=matrix_fig)
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Correlation Table", className="mb-3"),
                    dbc.Table.from_dataframe(
                        corr_df,
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="table-sm"
                    )
                ])
            ])
        ], fluid=True)

    def _render_scatter_tab(self):
        """Render scatter plots for all significant correlations with dimension selector"""
        # Get all correlations (not limited)
        all_correlations = self.analyzer.get_top_correlations(n=100)  # Get up to 100
        
        if not all_correlations:
            return html.Div("No correlations found", 
                          className="alert alert-info")
        
        # Dimension selector control
        dimension_control = dbc.Card([
            dbc.CardBody([
                html.H5("📍 Dimension Selection", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Dimension for Color Mapping:", 
                                 className="fw-bold"),
                        dcc.Dropdown(
                            id='scatter-dimension-dropdown',
                            options=[{'label': d, 'value': d} 
                                    for d in self.analyzer.dimensions],
                            value=self.analyzer.dimensions[0] if self.analyzer.dimensions else None,
                            className="mb-0"
                        )
                    ], width=6)
                ])
            ])
        ], className="mb-4")
        
        return dbc.Container([
            dbc.Alert([
                html.Strong(f"📊 Showing all {len(all_correlations)} significant correlations"),
                html.Br(),
                html.Small(f"Correlation threshold: {self.analyzer.correlation_threshold}")
            ], color="info", className="mb-3"),
            dimension_control,
            html.Div(id='scatter-plots-container')
        ], fluid=True)
    
    def _render_scatter_plots_for_dimension(self, dimension: str):
        """Generate scatter plots for selected dimension - shows all significant correlations"""
        if not dimension:
            return html.Div("Please select a dimension", 
                          className="alert alert-warning")
        
        # Get all significant correlations
        all_correlations = self.analyzer.get_top_correlations(n=100)  # Up to 100
        
        # Detect patterns for this dimension
        patterns = [p for p in self.analyzer.patterns 
                   if p.get('dimension') == dimension]
        
        scatter_plots = []
        for (metric_x, metric_y), corr_info in all_correlations:
            # Filter relevant patterns
            relevant_patterns = [p for p in patterns 
                               if p.get('metrics') == (metric_x, metric_y) or
                                  p.get('metric') in [metric_x, metric_y]]
            
            fig = self.visualizer.create_scatter_plot(
                metric_x, metric_y, dimension, corr_info,
                highlight_patterns=relevant_patterns
            )
            
            scatter_plots.append(
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig)
                    ])
                ], className="mb-4")
            )
        
        return scatter_plots
    
    def _render_patterns_tab(self):
        """Render pattern analysis view - shows patterns from all dimensions"""
        # Get all patterns (from all dimensions)
        all_patterns = self.analyzer.patterns
        
        if not all_patterns:
            return dbc.Container([
                dbc.Alert([
                    html.H4("⚡ No Patterns Detected", className="alert-heading"),
                    html.P("No basic patterns found in this dataset."),
                    html.Hr(),
                    html.P("Check the 'Hidden Patterns' tab for advanced pattern detection.")
                ], color="info")
            ], fluid=True)
        
        # Group patterns by dimension for better organization
        patterns_by_dimension = {}
        for pattern in all_patterns:
            dim = pattern.get('dimension', 'Unknown')
            if dim not in patterns_by_dimension:
                patterns_by_dimension[dim] = []
            patterns_by_dimension[dim].append(pattern)
        
        # Pattern summary figure (all patterns)
        summary_fig = self.visualizer.create_pattern_summary(all_patterns)
        
        # Create sections for each dimension
        dimension_sections = []
        for dimension, patterns in patterns_by_dimension.items():
            # Create pattern cards for this dimension
            pattern_cards = []
            for i, pattern in enumerate(patterns[:10], 1):  # Show top 10 per dimension
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6(f"Pattern {i}: {pattern['type']}", 
                               className="text-primary"),
                        html.P(pattern['description'], className="mb-2"),
                        html.Small(self._format_pattern_details(pattern),
                                 className="text-muted")
                    ])
                ], className="mb-2")
                pattern_cards.append(card)
            
            dimension_sections.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H5(f"📊 {dimension}", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(pattern_cards)
                    ])
                ], className="mb-4")
            )
        
        return dbc.Container([
            dbc.Alert([
                html.Strong(f"⚡ Found {len(all_patterns)} patterns across all dimensions")
            ], color="info", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=summary_fig)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Patterns by Dimension", className="mt-4 mb-3"),
                    html.Div(dimension_sections)
                ])
            ])
        ], fluid=True)
    
    
    def _render_distribution_tab(self):
        """Render distribution analysis view with dimension selector"""
        # Dimension selector control
        dimension_control = dbc.Card([
            dbc.CardBody([
                html.H5("📍 Dimension Selection", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Dimension for Distribution Analysis:", 
                                 className="fw-bold"),
                        dcc.Dropdown(
                            id='distribution-dimension-dropdown',
                            options=[{'label': d, 'value': d} 
                                    for d in self.analyzer.dimensions],
                            value=self.analyzer.dimensions[0] if self.analyzer.dimensions else None,
                            className="mb-0"
                        )
                    ], width=6)
                ])
            ])
        ], className="mb-4")
        
        return dbc.Container([
            dimension_control,
            html.Div(id='distribution-container')
        ], fluid=True)
    
    def _render_distribution_for_dimension(self, dimension: str):
        """Generate distribution plots for selected dimension"""
        if not dimension:
            return html.Div("Please select a dimension", 
                          className="alert alert-warning")
        
        distribution_plots = []
        
        for metric in self.analyzer.metrics[:6]:  # Limit to 6 metrics
            fig = self.visualizer.create_dimension_distribution(dimension, metric)
            distribution_plots.append(
                dbc.Col([
                    dcc.Graph(figure=fig)
                ], width=6, className="mb-4")
            )
        
        return dbc.Row(distribution_plots)
    
    def _create_correlation_table(self):
        """Create table of top correlations"""
        top_corr = self.analyzer.get_top_correlations(n=10)
        
        if not top_corr:
            return html.Div("No correlations to display")
        
        table_rows = []
        for (m1, m2), info in top_corr:
            row = html.Tr([
                html.Td(m1),
                html.Td(m2),
                html.Td(f"{info['coefficient']:.3f}"),
                html.Td(info['strength']),
                html.Td("✓" if info['significant'] else "✗"),
                html.Td(f"{info['p_value']:.4f}")
            ])
            table_rows.append(row)
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Metric 1"),
                html.Th("Metric 2"),
                html.Th("Correlation"),
                html.Th("Strength"),
                html.Th("Significant"),
                html.Th("P-Value")
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, responsive=True, striped=True)
        
        return table
    
    def _format_pattern_details(self, pattern: dict) -> str:
        """Format pattern details for display"""
        if pattern['type'] == 'outlier_group':
            return f"Group: {pattern['group']} | Z-score: {pattern['z_score']:.2f}"
        elif pattern['type'] == 'correlation_variation':
            return f"Variation: {pattern['variation']:.3f}"
        return ""
    
    def run(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard server
        
        Args:
            debug: Enable debug mode
            port: Port number
        """
        print(f"\n🚀 Starting dashboard on http://localhost:{port}")
        print(f"📊 Analyzing {len(self.analyzer.data)} rows with {len(self.analyzer.metrics)} metrics")
        print(f"🔍 Found {len(self.analyzer.correlations)} correlations")
        print(f"⚡ Detected {len(self.analyzer.patterns)} patterns\n")
        
        # Disable reloader to prevent double execution of analysis
        # Dev tools will still work for debugging UI
        self.app.run(debug=debug, port=port, host='0.0.0.0', use_reloader=False)
