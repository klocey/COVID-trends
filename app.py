#from importlib.metadata import version
#print('flask:', version('flask'))
#print('gunicorn:', version('gunicorn'))
#print('dash:', version('dash'))
#print('dash_bootstrap_components:', version('dash_bootstrap_components'))
#print('pandas:', version('pandas'))
#print('plotly:', version('plotly'))
#print('statsmodels:', version('statsmodels'))
#print('scipy:', version('scipy'))
#print('numpy:', version('numpy'))

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import warnings
import sys
import os
import re
import numpy as np
from scipy import stats
import random
import statsmodels.api as sm

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

#tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################

warnings.filterwarnings('ignore')
#pd.set_option('display.max_columns', None)

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server


#########################################################################################
################################# LOAD DATA #############################################
#########################################################################################

mydir = (os.getcwd()).replace('\\','/')+'/'
sys.path.append(mydir)

#counties_df = pd.read_pickle(mydir + 'data/dat_for_app.pkl')
main_df = pd.read_pickle(mydir + 'data/dat_for_app.pkl')

#tdf = main_df.filter(items=['date', 'Confirmed'], axis=1)

#########################################################################################
######################## Define static variables ########################################
#########################################################################################

features = list(main_df)
x1_features = list(features)
x2_features = features[1:]
y_features = features[1:]

operators = ['/', '*', '+', '-']


#########################################################################################
########################### CUSTOM FUNCTIONS ############################################
#########################################################################################

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


#########################################################################################
################# DASH APP CONTROL FUNCTIONS ############################################
#########################################################################################

def description_card1():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("IL COVID Trends",
                    style={
            'textAlign': 'left',
        }),
            html.P("Examine trends in COVID cases, hospitalization, hospital utilization, testing, and vaccination. Most data pertain to Illinois. That that aren't pertain to the US.",
                    style={
            'textAlign': 'left',
        }),
        ],
    )
    

def control_card1():

    return html.Div(
        id="control-card1",
        children=[
            html.Div(id='control-card1-container',
                children=[
                    html.H5("Design your x-variable", style={'display': 'inline-block', 'width': '55%'}),
                    html.I(className="fas fa-question-circle fa-lg", id="target1",
                        style={'display': 'inline-block', 'width': '20px', 'color':'#99ccff'},
                        ),
                    dbc.Tooltip("If you want a simple variable, then choose None for the second feature. If you choose 'date' for your first feature, the app will ignore the second feature.",
                        target="target1", style = {'font-size': 12},
                        ),
                    dcc.Dropdown(
                        id="x1",
                        options=[{"label": i, "value": i} for i in x1_features],
                        multi=False,
                        value='date',
                        style={
                            #'font-size': "100%",
                            'width': '420px',
                            'display': 'inline-block',
                            #'border-radius': '15px',
                            #'box-shadow': '1px 1px 1px grey',
                            #'background-color': '#f0f0f0',
                            #'padding': '10px',
                            #'margin-bottom': '10px',
                            'margin-right': '10px',
                            }
                    ),
                    dcc.Dropdown(
                        id="operator1",
                        options=[{"label": i, "value": i} for i in operators],
                        multi=False,
                        value='/',
                        style={
                            'font-size': "100%",
                            'width': '30px',
                            'display': 'inline-block',
                            'margin-right': '10px',
                            }
                    ),
                    
                    dcc.Dropdown(
                        id="x2",
                        options=[{"label": i, "value": i} for i in ['None'] + x2_features],
                        multi=False,
                        value='None',
                        style={
                            'font-size': "100%",
                            'width': '380px',
                            'display': 'inline-block',
                            }
                    ),
                    ],
                ),
            ],
    )
    
def control_card2():

    return html.Div(
        id="control-card2",
        children=[
            html.Div(id='control-card2-container',
                children=[
                    html.H5("Design your y-variables", style={'display': 'inline-block', 'width': '57%'}),
                    html.I(className="fas fa-question-circle fa-lg", id="target2",
                        style={'display': 'inline-block', 'width': '10%', 'color':'#99ccff'},
                        ),
                    dbc.Tooltip("If you want simple variables, then choose None for the second feature.",
                        target="target2", style = {'font-size': 12},
                        ),
                    dcc.Dropdown(
                        id="y1",
                        options=[{"label": i, "value": i} for i in y_features],
                        multi=True,
                        value=None,
                        placeholder='Choose 1 to 10 features',
                        style={
                            #'font-size': "100%",
                            'width': '420px',
                            #'display': 'inline-block',
                            #'border-radius': '15px',
                            #'box-shadow': '1px 1px 1px grey',
                            #'background-color': '#f0f0f0',
                            #'padding': '10px',
                            'margin-bottom': '10px',
                            'margin-right': '10px',
                            }
                    ),
                    dcc.Dropdown(
                        id="operator2",
                        options=[{"label": i, "value": i} for i in operators],
                        multi=False,
                        value='/',
                        style={
                            'font-size': "100%",
                            'width': '30px',
                            'display': 'inline-block',
                            'margin-right': '10px',
                            }
                    ),
                    
                    dcc.Dropdown(
                        id="y2",
                        options=[{"label": i, "value": i} for i in ['None'] + y_features],
                        multi=False,
                        value='None',
                        style={
                            'font-size': "100%",
                            'width': '380px',
                            'display': 'inline-block',
                            }
                    ),
                    ],
                ),
            ],
    )



def control_card3():

    return html.Div(
        id="control-card3",
        children=[
            html.Div(id='control-card3-container',
                children=[
                    html.H5("Dampen extreme outliers", style={'display': 'inline-block', 'width': '260px'}),
                    html.I(className="fas fa-question-circle fa-lg", id="target3",
                        style={'display': 'inline-block', 'color':'#99ccff'},
                        ),
                    dbc.Tooltip("Extreme outlier points are likely the result of time-lags and reporting issues. Choose this option to replace outliers with interpolated data points.",
                        target="target3", style = {'font-size': 12},
                        ),
                    dcc.RadioItems(
                        id="outliers",
                        options=[
                            {'label': ' Dampen outliers', 'value': 'dampen_outliers'},
                            {'label': ' Leave outliers as is', 'value': 'keep_outliers'},
                        ],
                        value='dampen_outliers',
                        labelStyle={'display': 'inline-block', 'width': '160px'}
                    ),
                    ],
                ),
            ],
    )
#########################################################################################
################### DASH APP PLOT FUNCTIONS #############################################
#########################################################################################


#########################################################################################
################################# DASH APP LAYOUT #######################################
#########################################################################################


app.layout = html.Div([
    
    html.Div(
            id='main_df',
            style={'display': 'none'}
        ),
    html.Div(
            id='counties_df',
            style={'display': 'none'}
        ),
    
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                        html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'}),
                      ],
        ),
    
    html.Div(
            id="description_card1",
            className="ten columns",
            children=[description_card1()],
            style={'width': '95%', 'display': 'inline-block',
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
            },
        ),
    
    html.Div(
            id="right-column1",
            className="four columns",
            children=[control_card1(),
                      html.Br(),
                      html.Hr(),
                      control_card2(),
                      html.Br(),
                      html.Hr(),
                      control_card3(),
                      ],
                      style={#'width': '95%',
                             #'display': 'inline-block',
                             'border-radius': '15px',
                             'box-shadow': '1px 1px 1px grey',
                             'background-color': '#f0f0f0',
                             'padding': '10px',
                             'margin-bottom': '10px',
                            },
                        ),

    html.Div(
            id="right-column2",
            className="eight columns",
            children=[
                            
                html.Div(
                        id="Figure1",
                        children=[dcc.Loading(
                            id="loading-2",
                            type="default",
                            fullscreen=False,
                            children=html.Div(id="figure1",
                                children=[dcc.Graph(id="figure_plot1"),
                                        ]))],
                                        style={'width': '95%',
                                                'height': '650px',
                                                'display': 'inline-block',
                                                'border-radius': '15px',
                                                'box-shadow': '1px 1px 1px grey',
                                                'background-color': '#f0f0f0',
                                                'padding': '10px',
                                                'margin-bottom': '10px',
                                                     },
                                    ),
                ]),
])



#########################################################################################
############################    Call backs   #######################################
#########################################################################################



@app.callback(Output('figure_plot1', 'figure'), 
              [Input('x1', 'value'),
              Input('x2', 'value'),
              Input('y1', 'value'),
              Input('y2', 'value'),
              Input('operator1', 'value'),
              Input('operator2', 'value'),
              Input('outliers', 'value')
              ],
              )
def update_results_figure(x1, x2, y1, y2, operator1, operator2, outliers):
    
    if x1 in [None, 'None', ''] or y1 in [None, [None], ['None'], []]:
        figure = go.Figure(data=[go.Table(
                header=dict(values=[],
                        fill_color='#b3d1ff',
                        align='left'),
                        ),
                    ],
                )

        figure.update_layout(title_font=dict(size=14,
                          color="rgb(38, 38, 38)",
                          ),
                          margin=dict(l=10, r=10, b=10, t=0),
                          paper_bgcolor="#f0f0f0",
                          plot_bgcolor="#f0f0f0",
                          height=400,
                          )
            
        return figure
    
    
    
    t_features = [x1] + y1
    if x2 not in [None, 'None', '', []]:
        t_features.append(x2)
    if y2 not in [None, 'None', '', []]:
        t_features.append(y2)
    
    t_features = list(set(t_features))
    tdf = main_df.filter(items=t_features, axis=1)
    
    if x1 == 'date':
        x2 = 'None'
    if 'date' in y1:
        y1 = ['date']
        y2 = 'None'
        
    xlab = str(x1)
    if x2 != 'None':
        
        xlab = x1 + ' ' + operator1 + ' ' + x2
        
        numerator = tdf[x1]
        denominator = tdf[x2]
        
        if operator1 == '/':
            tdf[xlab] = numerator / denominator
        elif operator1 == '*':
            tdf[xlab] = numerator * denominator
        elif operator1 == '+':
            tdf[xlab] = numerator + denominator
        elif operator1 == '-':
            tdf[xlab] = numerator - denominator
    
    ylabs = y1
    if y2 not in ['None', None, np.nan]:
        ylabs = []
        for yvar in y1:
            
            ylab = yvar + ' ' + operator2 + ' ' + y2
            ylabs.append(ylab)
            #print(ylabs)
            
            numerator = tdf[yvar]
            denominator = tdf[y2]
                
            q = 0
            if operator2 == '/':
                q = numerator / denominator
            elif operator2 == '*':
                q = numerator * denominator
            elif operator2 == '+':
                q = numerator + denominator
            elif operator2 == '-':
                q = numerator - denominator
               
            tdf[ylab] = q
            
    #print('xlab:', xlab)
    #print('ylabs:', ylabs, '\n')
    #print(list(tdf), '\n')
    #print(tdf.head(10))
    
    fig_data = []
    clrs = ['#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            '#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            '#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            '#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            '#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            '#ff0000', '#0000ff', '#009900', '#993399', '#009999',
            '#ff9966', '#00ff00', '#3366cc', '#cc6699', '#000066',
            ]
    
    for i, ylab in enumerate(ylabs):
        ttdf = tdf.filter(items=[xlab, ylab], axis=1)
        
        if outliers == 'dampen_outliers':
            ys = ttdf[ylab].tolist()
            xs = ttdf[ylab].tolist()
            
            for ii, y in enumerate(ys):
                if ii > 0:
                    if y > 10 * ys[ii-1]:
                        ys[ii] = ys[ii-1]
            ttdf[ylab] = ys
            
            if xlab != 'date':
                xs = ttdf[ylab].tolist()
                
                for ii, x in enumerate(xs):
                    if ii > 0:
                        if x > 10 * xs[ii-1]:
                            xs[ii] = xs[ii-1]
                ttdf[xlab] = xs
            
        
        ttdf.dropna(how='any', inplace=True)
        fig_data.append(
                    go.Scatter(x = ttdf[xlab], y = ttdf[ylab],
                    mode="markers",
                    marker_color = clrs[i],
                    name = ylab,
                    #text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
                    opacity = 0.75,
                    line=dict(color=clrs[0], width=2),
                ))
        
        X = []
        if xlab == 'date':
            X = list(range(len(ttdf[xlab].tolist())))
        else:
            X = ttdf[xlab].tolist()
        
        Y = ttdf[ylab].tolist()
        
        lowess = sm.nonparametric.lowess
        ty = lowess(Y, X, frac=1/40)
        ty = np.transpose(ty)
        ty = ty[1]
        
        r2 = obs_pred_rsquare(Y, ty)
        r2 = np.round(100*r2, 1)
        
        fig_data.append(
            go.Scatter(x = ttdf[xlab], y = ty,
            mode="lines",
            marker_color = clrs[i],
            name = 'Trend: ' + ylab,
            #text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
            opacity = 0.75,
            line=dict(color=clrs[i], width=2),
        ))
        
    if len(y1) > 1:
        ylab = None
        
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text=xlab,
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            
            yaxis=dict(
                title=dict(
                    text=ylab,
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=600,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )

    ypos = -0.15
    figure.update_layout(
        legend=dict(
            orientation = "h",
            y = ypos,
            yanchor = "top",
            xanchor="left",
            traceorder = "normal",
            font = dict(
                size = 12,
                color = "rgb(38, 38, 38)"
            ),
        )
    )
    
    del tdf
    return figure
    
    
#########################################################################################
############################# Run the server ############################################
#########################################################################################

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True) # modified to run on linux server
