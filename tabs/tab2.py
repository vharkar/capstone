import dash
import dash_daq as daq
from dash import dcc, html

canvas_width = 200
canvas_height = 200

tab_2_layout = html.Div(children=[
        html.H1('Handwritten Name Recognition'),
        html.Div([

            html.Div([
                html.H3('Pick a test image'),
                html.Br(),
                html.Br(),
                html.Br(),
                DashCanvas(
                    id='namedraw',
                    lineWidth=2,
                    lineColor='rgba(255, 0, 0, 0.5)',
                    width=canvas_width,
                    height=canvas_height,
                    hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "select"],
                    goButtonTitle='Submit',
                ),
            ], style={"padding-left": "20px", "align":"left"}, className="three columns"),

            html.Div(html.Img(id='my-image', width=400), className="five columns"),

            html.Div([
                html.H3('Your name is.....'),
                html.Br(),
                html.H6(id='name-prediction', children='...'),
                html.Br(),
            ], className='three columns'),
        ], className="twelve columns"),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
    ], className="twelve columns")