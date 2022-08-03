import numpy as np
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash_canvas.utils import  parse_jsonstring, io_utils
from dash_canvas.utils import array_to_data_url
from PIL import Image
import pandas as pd
import cv2
import json
import pywhatkit as kit
from keras import backend as K
from tabs import tab1, tab2, tab3, tab4

########### Read CSV of test files ###########
file_to_img = pd.read_csv('./assets/written_name_test_v2.csv')
file_to_img_3 = pd.read_csv('./assets/created_names.csv')

########### open the model ######
from keras.models import load_model
model = load_model('assets/handwriting-model-1.h5', compile=False)

########### define variables
tabtitle='Handwritten Name Recognition Model'

########### BLANK FIGURE
templates=['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white', 'plotly_dark',
            'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

canvas_width = 200
canvas_height = 200

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

def img_to_text(img):
    image = preprocess(img)
    image = image / 255.

    # make prediction
    pred = model.predict(image.reshape(1, 256, 64, 1))
    return K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
app.title=tabtitle

## Layout
app.layout = html.Div([
    html.H1('Handwritten Name Recognition'),
    dcc.Tabs(id="tabs-template", value='tab-1-template', children=[
        dcc.Tab(label='Model Prediction', value='tab-1-template'),
        dcc.Tab(label='Model Image Recognition - 1', value='tab-2-template'),
        dcc.Tab(label='Model Image Recognition - 2', value='tab-3-template'),
        dcc.Tab(label='Model Image Recognition - 3', value='tab-4-template'),
    ]),
    html.Div(id='tabs-content-template')
])


############ Callbacks

@app.callback(Output('tabs-content-template', 'children'),
              [Input('tabs-template', 'value')])
def render_content(tab):
    if tab == 'tab-1-template':
        return tab1.tab_1_layout
    elif tab == 'tab-2-template':
        return tab2.tab_2_layout
    elif tab == 'tab-3-template':
        return tab3.tab_3_layout
    elif tab == 'tab-4-template':
        return tab4.tab_4_layout

@app.callback(Output('tab1-image', 'src'),
              Output('tab1-name-written', 'children'),
              Output('tab1-name-prediction', 'children'),
              Input('tab1-file_num', 'value'))
def update_tab1_data(rangeval):
    # Get name and image
    name = file_to_img.loc[rangeval, 'IDENTITY']
    filename = file_to_img.loc[rangeval, 'FILENAME']

    # Get image and process it.
    img_dir = './images/test/' + filename
    imageRead = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    decoded = img_to_text(imageRead)

    return Image.fromarray(imageRead), name, num_to_label(decoded[0])

@app.callback(Output('tab2-image', 'src'),
              Output('tab2-name-prediction', 'children'),
    Input('tab2-namedraw', 'json_data'))
def update_tab2_data(string):
    if string:
        data = json.loads(string)
        #print(data['objects'][0]['path'])  # explore the contents of the shape file

        mask = parse_jsonstring(string, shape=(canvas_width, canvas_height))
        drawnImg = array_to_data_url((255 * mask).astype(np.uint8))

        decoded = img_to_text(mask)

    else:
        raise PreventUpdate
    return drawnImg, num_to_label(decoded[0])

@app.callback(Output('tab3-image', 'src'),
              Output('tab3-name-written', 'children'),
              Output('tab3-name-prediction', 'children'),
              Input('tab3-file_num', 'value'))
def update_tab3_data(rangeval):
    # Get name and image
    name = file_to_img_3.loc[rangeval, 'IDENTITY']
    filename = file_to_img_3.loc[rangeval, 'FILENAME']

    # Get image and process it.
    img_dir = './my_images/' + filename
    imageRead = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)

    decoded = img_to_text(imageRead)

    return Image.fromarray(imageRead), name, num_to_label(decoded[0])

@app.callback(Output('tab4-written', 'children'),
              Output('tab4-image', 'src'),
              Output('tab4-name-prediction', 'children'),
              Input('tab4-text', 'value'))
def update_tab4_data(value):
    # Get name and image

    kit.text_to_handwriting(value, save_to="./tmp_images/hw.jpg")

    # Get image and process it.
    img_dir = './tmp_images/hw.jpg'
    imagecv2 = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    imagepil =Image.open(img_dir)
    size = 500, 500
    imagepil.thumbnail(size, Image.ANTIALIAS)

    decoded = img_to_text(imagecv2)

    return value, imagepil, num_to_label(decoded[0])

if __name__ == '__main__':
    app.run_server(debug=True)
