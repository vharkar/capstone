import numpy as np
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import dash_daq as daq
from PIL import Image
import pandas as pd
import cv2
from keras import backend as K

########### Read CSV of test files ###########
file_to_img = pd.read_csv('./assets/written_name_test_v2.csv')

########### open the model ######
from keras.models import load_model
model = load_model('assets/handwriting-model-1.h5', compile=False)

########### define variables
tabtitle='Handwritten Name Recognition Model'
sourceurl = 'https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/data'
githublink = 'https://github.com/vharkar/capstone'
canvas_width = 200
canvas_height = 200

########### BLANK FIGURE
templates=['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white', 'plotly_dark',
            'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

############ FUNCTIONS#################
## Preprocess Image
def preprocess(img):
    (h, w) = img.shape

    final_img = np.ones([64, 256]) * 255  # blank white image

    # crop
    if w > 256:
        img = img[:, :256]

    if h > 64:
        img = img[:64, :]

    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    #return final_img

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
app.title=tabtitle

app.layout = html.Div(children=[
        html.H2('Handwritten Name Recognition'),
        html.Div([

            html.Div([
                html.H4('Pick number between 1 and 41370 (there are 41370 test files)'),
                daq.NumericInput(
                    id='file_num',
                    value=0, min=0, max=41370, size=120
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
                html.Div(html.Img(id='my-image', width=400), className="five columns"),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Name'),
                html.Br(),
                html.H6(id='name-written', children='...'),
                html.Br(),
            ], className='three columns'),

            html.Div([
                html.H3('Recognized as'),
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


@app.callback(Output('my-image', 'src'),
              Output('name-written', 'children'),
              Output('name-prediction', 'children'),
              Input('file_num', 'value'))
def update_data(rangeval):
    # Get name and image
    #idx = rangeval - 1
    name = file_to_img.loc[rangeval, 'IDENTITY']
    filename = file_to_img.loc[rangeval, 'FILENAME']

    # Get image and process it.
    img_dir = './images/test/' + filename
    imageRead = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(imageRead)
    image = image / 255.

    # make prediction
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])

    return Image.fromarray(imageRead), name, num_to_label(decoded[0])

if __name__ == '__main__':
    app.run_server(debug=True)
