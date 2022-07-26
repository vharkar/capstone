import dash
import dash_daq as daq
from dash import dcc, html

sourceurl = 'https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/data'
githublink = 'https://github.com/vharkar/capstone'

tab_1_layout = html.Div([

        html.Div([
            html.H3('Predict name from test images'),
            html.Div([
                html.H4('Pick number between 1 and 100 (there are 100 test files)'),
                daq.NumericInput(
                    id='tab1-file_num',
                    value=0, min=0, max=100, size=50
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
                html.Div(html.Img(id='tab1-image', width=400), className="five columns"),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Name'),
                html.Br(),
                html.H6(id='tab1-name-written', children='...'),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Recognized as'),
                html.Br(),
                html.H6(id='tab1-name-prediction', children='...'),
                html.Br(),
            ], className='three columns'),
        ], className="twelve columns"),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
    ], className="twelve columns")