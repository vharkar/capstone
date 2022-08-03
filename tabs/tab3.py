import dash
import dash_daq as daq
from dash import dcc, html

tab_3_layout = html.Div([
        html.Div([
            html.H3('Predict name from created images'),
            html.Div([
                html.H4('Pick number between 1 and 9 (there are 9 test files)'),
                daq.NumericInput(
                    id='tab3-file_num',
                    value=1, min=1, max=9, size=50
                ),
                html.Br(),
                html.Br(),
                html.Br(),
            ], style={"padding-left": "20px", "align":"left"}, className="twelve columns"),

            html.Br(),
            html.Br(),

            html.Div([
                html.H3('Handwriting Image'),
                html.Br(),
                html.Div(html.Img(id='tab3-image', width=400), className="five columns"),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Name'),
                html.Br(),
                html.H6(id='tab3-name-written', children='...'),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Recognized as'),
                html.Br(),
                html.H6(id='tab3-name-prediction', children='...'),
                html.Br(),
            ], className='three columns'),
        ], className="twelve columns"),
    ], className="twelve columns")