
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class VisualizationModule:
    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    def create_animated_scatter(self, data, x_col, y_col, animation_col=None):
        """Create animated scatter plot"""
        if animation_col:
            fig = px.scatter(data, x=x_col, y=y_col, 
                            animation_frame=animation_col,
                           title=f"Animated {y_col} vs {x_col}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], 
                                    mode='markers'))

        fig.update_layout(height=500)
        return fig

    def create_3d_surface(self, x, y, z, title="3D Surface Plot"):
        """Create 3D surface plot"""
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=600
        )

        return fig

    def create_correlation_heatmap(self, data, title="Correlation Matrix"):
        """Create correlation heatmap"""
        corr_matrix = data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title=title,
            height=600,
            width=700
        )

        return fig

    def create_interactive_time_series(self, data, date_col, value_cols):
        """Create interactive time series with range slider"""
        fig = go.Figure()

        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data[col],
                mode='lines',
                name=col
            ))

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(
            title="Interactive Time Series",
            height=500,
            hovermode='x unified'
        )

        return fig
