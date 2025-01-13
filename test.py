import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, no_update, callback
from skimage import data, draw
from scipy import ndimage
from PIL import Image

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)

def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    print(cols,rows)

    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask

img = Image.open('Frame.png')
fig = px.imshow(img, binary_string=True)
fig.update_layout(dragmode="drawclosedpath",width=900)

# fig_hist = px.histogram(img.ravel())

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H3("Draw a path to show the histogram of the ROI"),
        html.Div(
            [dcc.Graph(id="graph-camera", figure=fig),],
            style={"width": "100%", "display": "inline-block", "padding": "0 0"},
        ),
        # html.Div(
        #     [dcc.Graph(id="graph-histogram", figure=fig_hist),],
        #     style={"width": "40%", "display": "inline-block", "padding": "0 0"},
        # ),
    ]
)

@callback(
    Output("graph-camera", "figure"),
    Input("graph-camera", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    # print(relayout_data)
    if "shapes" in relayout_data:
        last_shape = relayout_data["shapes"][-1]
    
        mask = path_to_mask(last_shape["path"], np.array(img).shape)
        # return px.histogram(img[mask]
        return no_update
    else:
        return no_update

if __name__ == "__main__":
    app.run(debug=True)
