import dash
import dash_daq as daq
from dash import dcc, html
from dash_canvas import DashCanvas

canvas_width = 200
canvas_height = 200

tab_2_layout = html.Div([

    html.Div([
        html.H3('Recognize name from canvas image'),
        html.Div([
            html.H3('Canvas'),
            html.Br(),
            html.Br(),
            html.Br(),
            DashCanvas(
                id='tab2-namedraw',
                lineWidth=2,
                lineColor='rgba(255, 0, 0, 0.5)',
                width=canvas_width,
                height=canvas_height,
                hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "select"],
                goButtonTitle='Submit',
            ),
        ], style={"padding-left": "20px", "align": "left"}, className="three columns"),

        html.Div([
            html.H3('Image from Canvas'),
            html.Br(),
            html.Div(html.Img(id='tab2-image', width=300), className="five columns"),
            html.Br(),
        ], className='three columns'),

        html.Div([
            html.H3('Recognized Image as..'),
            html.Br(),
            html.H6(id='tab2-name-prediction', children=''),
            html.Br(),
        ], className='three columns'),
    ], className="twelve columns"),
], className="twelve columns")
