#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        subplot_titles=(f"{title}", " ")
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




def get_data(all_monthly):
    emp_cols = ['manufact_employee', 'all_employees', 'job_opening', 'layoffs', 
                'agriculture_employees', 'mining_employees', 'wholesale_employees', 
                'truck_employees', 'it_employees', 'retail_employees', 'leisure_employees', 
                'finace_employees', 'construction_employees', 'federal_employees',
                'buisness_employees', 'health_employees', 'other_employees']  
    
    rate_cols = ['unemployment', 'participation_rate', 'participation_25_54_rate', 'participation_55+_rate', 
                 'participation_16_19_rate', 'participation_20_24_rate', 
                 'participation_women_rate', 'participation_men_rate', 
                 'participation_immigrant_rate']


    emp_df = all_monthly.copy()
    emp_df = emp_df[emp_cols + rate_cols]
    emp_df.reset_index(inplace=True)
    emp_df = emp_df[emp_df['date'] >= '1990-12-31']
    
    
    # Derived perc changes
    for col in emp_cols:
        emp_df[f'{col}_perc_change'] = 100 * (emp_df[col] - emp_df[col].shift(12))/ emp_df[col].shift(12)
    
    for col in rate_cols:
        emp_df[f'{col}_perc_change'] = 100 * (emp_df[col] - emp_df[col].shift(12))/ emp_df[col].shift(12)
    
    #emp_df = emp_df.fillna(0)
    #emp_df = emp_df.iloc[:, :-1]
    #emp_df.dropna(inplace = True)
    
    return emp_df

def get_layout(all_monthly):
    
    emp_df = get_data(all_monthly)


    layout = html.Div([
        html.H2("Employee Dashboard"),
    
        dcc.DatePickerRange(
            id='date-picker-employee',
            min_date_allowed=emp_df['date'].min(),
            max_date_allowed=emp_df['date'].max(),
        ),
        
        html.Button("Reset", id="reset-button-employee", n_clicks=0, style={'margin': '20px 0'}),
        
        html.Div([
            html.Div([dcc.Graph(id='unemp-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='allemp-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='jobopen-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='layoff-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='fed-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='buisness-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='agremp-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='mineemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='wholeemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='truckemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='constructemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='manufact-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='retailemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='leisureemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='itemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='financeemp-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='health-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='other-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='participation-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='women_part-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='men_part-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='part1619-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='part2024-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='part2555-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='part55-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='partimmg-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Div([
            html.Div([dcc.Graph(id='heatmap-employ-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
    ])
    
    return layout
    

def register_callbacks(app, all_monthly):
    
    emp_df = get_data(all_monthly)
    @app.callback(
        [
         Output('unemp-fig', 'figure'),
         Output('allemp-fig', 'figure'),
         Output('jobopen-fig', 'figure'),
         Output('layoff-fig', 'figure'),
         Output('agremp-fig', 'figure'),
         Output('mineemp-fig', 'figure'),
         Output('wholeemp-fig', 'figure'),
         Output('truckemp-fig', 'figure'),
         Output('itemp-fig', 'figure'),
         Output('retailemp-fig', 'figure'),
         Output('leisureemp-fig', 'figure'),
         Output('financeemp-fig', 'figure'),
         Output('constructemp-fig', 'figure'),
         Output('manufact-fig', 'figure'),
         Output('fed-fig', 'figure'),
         Output('buisness-fig', 'figure'),
         Output('health-fig', 'figure'),
         Output('other-fig', 'figure'),         
         Output('participation-fig', 'figure'),
         Output('women_part-fig', 'figure'),
         Output('men_part-fig', 'figure'),
         Output('part1619-fig', 'figure'),
         Output('part2024-fig', 'figure'),
         Output('part2555-fig', 'figure'),
         Output('part55-fig', 'figure'),
         Output('partimmg-fig', 'figure'),
         
         Output('heatmap-employ-fig', 'figure'),
        

         Output('date-picker-employee', 'start_date'),
         Output('date-picker-employee', 'end_date'),
         ],
        
        [Input('date-picker-employee', 'start_date'),
         Input('date-picker-employee', 'end_date'),
         Input('reset-button-employee', 'n_clicks')
         
         ]
    )
    def update_graphs(start_date, end_date, reset_clicks):
        #set defaults for reset
        triggered_id = ctx.triggered_id
        
        # Set default values
        default_start = emp_df['date'].min()
        default_end = emp_df['date'].max()
        if start_date is None:
            start_date = default_start
        if end_date is None:
            end_date = default_end
    
        if triggered_id == 'reset-button-employee':
            start_date = default_start
            end_date = default_end
    
        # Filter data
        mask = (emp_df['date'] >= start_date) & (emp_df['date'] <= end_date)
        filtered_df = emp_df.loc[mask]
        date_col = filtered_df['date']
        
        unemp = make_lineplot_with_yoybars(filtered_df, main_data= 'unemployment',
                                               yoy_data = 'unemployment_perc_change',
                                               title = 'Unemployment Rate',
                                               yaxis_label= 'Rate',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        allemp = make_lineplot_with_yoybars(filtered_df, main_data= 'all_employees',
                                               yoy_data = 'all_employees_perc_change',
                                               title = 'All Employees (nonfarm)',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        jobopen = make_lineplot_with_yoybars(filtered_df, main_data= 'job_opening',
                                               yoy_data = 'job_opening_perc_change',
                                               title = 'All Job Openings',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        layoff = make_lineplot_with_yoybars(filtered_df, main_data= 'layoffs',
                                               yoy_data = 'layoffs_perc_change',
                                               title = 'All Layoffs',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        agremp = make_lineplot_with_yoybars(filtered_df, main_data= 'agriculture_employees',
                                               yoy_data = 'agriculture_employees_perc_change',
                                               title = 'Agriculture Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        mineemp = make_lineplot_with_yoybars(filtered_df, main_data= 'mining_employees',
                                               yoy_data = 'mining_employees_perc_change',
                                               title = 'Mining/Forestry Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        wholeemp = make_lineplot_with_yoybars(filtered_df, main_data= 'wholesale_employees',
                                               yoy_data = 'wholesale_employees_perc_change',
                                               title = 'Wholesale Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        truckemp = make_lineplot_with_yoybars(filtered_df, main_data= 'truck_employees',
                                               yoy_data = 'truck_employees_perc_change',
                                               title = 'Trucker Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        itemp = make_lineplot_with_yoybars(filtered_df, main_data= 'it_employees',
                                               yoy_data = 'it_employees_perc_change',
                                               title = 'IT Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
        retailemp = make_lineplot_with_yoybars(filtered_df, main_data= 'retail_employees',
                                               yoy_data = 'retail_employees_perc_change',
                                               title = 'Retail Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        leisureemp = make_lineplot_with_yoybars(filtered_df, main_data= 'leisure_employees',
                                               yoy_data = 'leisure_employees_perc_change',
                                               title = 'Leisure & Hospitality Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        financeemp = make_lineplot_with_yoybars(filtered_df, main_data= 'finace_employees',
                                               yoy_data = 'finace_employees_perc_change',
                                               title = 'Finace Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
    
        constructemp = make_lineplot_with_yoybars(filtered_df, main_data= 'construction_employees',
                                               yoy_data = 'construction_employees_perc_change',
                                               title = 'Construction Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        manufact = make_lineplot_with_yoybars(filtered_df, main_data= 'manufact_employee',
                                               yoy_data = 'manufact_employee_perc_change',
                                               title = 'Manufacturing Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        fedemp = make_lineplot_with_yoybars(filtered_df, main_data= 'federal_employees',
                                               yoy_data = 'federal_employees_perc_change',
                                               title = 'Federal Government Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        buisnessemp = make_lineplot_with_yoybars(filtered_df, main_data= 'buisness_employees',
                                               yoy_data = 'buisness_employees_perc_change',
                                               title = 'Buisness Services Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        healthemp = make_lineplot_with_yoybars(filtered_df, main_data= 'health_employees',
                                               yoy_data = 'health_employees_perc_change',
                                               title = 'Healthcare Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None), 
        otheremp  = make_lineplot_with_yoybars(filtered_df, main_data= 'other_employees',
                                               yoy_data = 'other_employees_perc_change',
                                               title = 'Other Services Employees',
                                               yaxis_label= 'Thousands',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None), 
    
        participation = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_rate',
                                               yoy_data = 'participation_rate_perc_change',
                                               title = 'Labor Force Participation Rate',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        women_part = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_women_rate',
                                               yoy_data = 'participation_women_rate_perc_change',
                                               title = 'Labor Force Participation Rate: Women',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
        men_part = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_men_rate',
                                               yoy_data = 'participation_men_rate_perc_change',
                                               title = 'Labor Force Participation Rate: Men',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        part_16_19 = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_16_19_rate',
                                               yoy_data = 'participation_16_19_rate_perc_change',
                                               title = 'Labor Force Participation Rate: 16-19yr old',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        part_20_24 = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_20_24_rate',
                                               yoy_data = 'participation_20_24_rate_perc_change',
                                               title = 'Labor Force Participation Rate: 20-24yr old',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        part_25_55 = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_25_54_rate',
                                               yoy_data = 'participation_25_54_rate_perc_change',
                                               title = 'Labor Force Participation Rate: 25-54yr old',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        part_55 = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_55+_rate',
                                               yoy_data = 'participation_55+_rate_perc_change',
                                               title = 'Labor Force Participation Rate: 55+ yr old',
                                               yaxis_label= 'Rate (%)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        part_immg = make_lineplot_with_yoybars(filtered_df, main_data= 'participation_immigrant_rate',
                                               yoy_data = 'participation_immigrant_rate_perc_change',
                                               title = 'Labor Force Participation Rate: 1st Gen Immigrants',
                                               yaxis_label= 'Rate',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
         
        ### Correlation Heatmap ###
        
        filtered_df2 = filtered_df.copy()
        filtered_df2 = filtered_df2[['all_employees', 'job_opening',
                       'layoffs', 'unemployment', 'participation_rate', 
                        'participation_16_19_rate', 'participation_20_24_rate', 
                        'participation_25_54_rate','participation_55+_rate', 
                        'truck_employees', 'it_employees', 'wholesale_employees',
                        'finace_employees', 'health_employees', 'federal_employees']]
        
        filtered_df2.columns = ['All Emp', 'Job Openings','Layoffs',
                                'Unemployment', 'total partrate', '16-19 partrate',
                                '20-24 partrate', '25-55 partrate', '55+ partrate',
                                'Truck Emp', 'IT Emp', 'Wholesale Emp', 'Finance Emp',
                                'Health Emp', 'Gov Emp']
        
        corr_matrix = filtered_df2.corr(numeric_only=True)
        heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))
    
        heatmap.update_layout(
            title="Pearson Correlations",
            #xaxis_title="Variables",
            #yaxis_title="Variables",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            width=800,
            height=800
        )
        
        return (unemp, allemp, jobopen, layoff, agremp, mineemp, wholeemp, truckemp,
                itemp, retailemp, leisureemp, financeemp, constructemp, manufact,
                fedemp, buisnessemp, healthemp, otheremp, participation,
                women_part, men_part, part_16_19, part_20_24, part_25_55, part_55,
                part_immg, heatmap,
                start_date, end_date)
    
    
    