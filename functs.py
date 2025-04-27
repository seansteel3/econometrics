#!/usr/bin/env python3

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def make_lineplot_with_yoybars(df, main_data, yoy_data, title, yaxis_label, color_scale, line_color = 'royalblue', cmin = None, cmax = None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        #subplot_titles=(f"{title}", " ")
    )
    

    #production
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[main_data],
            name=f"{title}",
            mode='lines',
            line=dict(color=line_color)
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = df[yoy_data]
    
    if cmin is None:
        cmin = bar_colors.min()
        cmax = bar_colors.max()
    else:
        cmin = cmin
        cmax = cmax
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df[yoy_data],
            marker=dict(
                color=bar_colors,
                colorscale=color_scale,  # red-yellow-green
                cmin=cmin ,
                cmax=cmax,
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        title=f"{title} with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title=yaxis_label),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    return fig

