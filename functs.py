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


def generate_basic_heatmap(df, columns, new_column_names, width=800, height=800, title = True):
    df = df.copy()
    df = df[columns]
    
    df.columns = new_column_names
    
    corr_matrix = df.corr(numeric_only=True)
    heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))
    
    if title:
        heatmap.update_layout(
            title="Pearson Correlations",
            #xaxis_title="Variables",
            #yaxis_title="Variables",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            width=width,
            height=height
        )
    else:
        heatmap.update_layout(
            #title="Pearson Correlations",
            #xaxis_title="Variables",
            #yaxis_title="Variables",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            width=width,
            height=height
        )
    return heatmap

def linreg_beta(x,y, window = 12):
    mean_x = x.rolling(window=window).mean()
    mean_y = y.rolling(window=window).mean()
    cov_xy = (x * y).rolling(window=window).mean() - mean_x * mean_y

    # Rolling variance
    var_x = (x * x).rolling(window=window).mean() - mean_x * mean_x
    
    # Rolling beta
    beta = cov_xy / var_x
    return beta



