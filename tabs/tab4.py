import dash
import dash_daq as daq
from dash import dcc, html
from dash_canvas import DashCanvas

canvas_width = 200
canvas_height = 200

tab_4_layout = html.Div([
    html.Div([
        html.H3('Recognize name from converted text'),
        html.Div([
            html.H3('Input Name:'),
            html.Br(),
            html.Br(),
            html.Br(),
            dcc.Textarea(
                id='tab4-text',
                value='',
                style={'width': '50%', 'height': 10},
            ),
            html.Div(id='tab4-written', style={'whiteSpace': 'pre-line'}),
        ], style={"padding-left": "20px", "align": "left"}, className="three columns"),

        html.Div([
            html.H3('Image from text'),
            html.Br(),
            html.Div(html.Img(id='tab4-image', width=500), className="five columns"),
            html.Br(),
        ], className='three columns'),

        html.Div([
            html.H3('Recognized Image as..'),
            html.Br(),
            html.H6(id='tab4-name-prediction', children=''),
            html.Br(),
        ], className='three columns'),
    ], className="twelve columns"),
], className="twelve columns")
