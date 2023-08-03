import dash
from dash import html, dcc
from dash.dependencies import Input, Output

#import yolov8

import pandas as pd
import plotly.graph_objs as go
import numpy as np
import cv2


app = dash.Dash(__name__)

# Load the pre-calculated statistics from the corr.csv file
#df_corr = pd.read_csv('corr.csv', index_col=0)

app.layout = html.Div([
    html.H1("Image Statistics Dashboard"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    dcc.Dropdown(
        id='dataset-dropdown',
        options=["mouse", "zebra", "windows", "kangaroos"],
        value='mouse'  # Default dataset selection
    ),
    html.Div(id='output-image-upload'),
    dcc.Graph(id='image-stats-plot'),
    html.Div(id='image-stats-comparison')
])

# Replace this function with our actual image processing logic
def get_image_stats(image_path):
    image = cv2.imread(image_path)

    # Placeholder code for the image statistics (replace this with actual calculations)
    aspect_ratio = image.shape[1] / image.shape[0]
    brightness = 0  # Replace this with actual brightness calculation
    contrast = 0  # Replace this with actual contrast calculation
    sharpness = 0  # Replace this with actual sharpness calculation
    noise = 0  # Replace this with actual noise calculation
    saturation = 0  # Replace this with actual saturation calculation
    entropy = 0  # Replace this with actual entropy calculation
    edges = 0  # Replace this with actual edges calculation
    estimate_noise = 0  # Replace this with actual estimate_noise calculation
    red_channel = 0  # Replace this with actual red channel statistics calculation
    blue_channel = 0  # Replace this with actual blue channel statistics calculation
    green_channel = 0  # Replace this with actual green channel statistics calculation
    salt_pepper_noise = 0  # Replace this with actual salt and pepper noise calculation
    blurriness = 0  # Replace this with actual blurriness calculation

    # Create a dictionary to store the calculated image statistics
    image_statistics = {
        'Aspect Ratio': aspect_ratio,
        'Brightness': brightness,
        'Contrast': contrast,
        'Sharpness': sharpness,
        'Noise': noise,
        'Saturation': saturation,
        'Entropy': entropy,
        'Edges': edges,
        'Estimate Noise': estimate_noise,
        'Red Channel': red_channel,
        'Blue Channel': blue_channel,
        'Green Channel': green_channel,
        'Salt and Pepper Noise': salt_pepper_noise,
        'Blurriness': blurriness
    }
    return image_statistics

@app.callback(Output('output-image-upload', 'children'),
              Output('image-stats-plot', 'figure'),
              Output('image-stats-comparison', 'children'),
              Input('upload-image', 'contents'))
def update_image(content):
    if content is not None:
        _, content_string = content.split(',')
        image_path = 'uploads/image.png'  # Replace with the actual path where the image is saved.

        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(content_string))

        image_stats = calculate_image_statistics(image_path)

        # Create a DataFrame from the calculated image statistics
        df_image_stats = pd.DataFrame([image_stats])

        # Calculate the correlation between the calculated statistics and the pre-calculated dataset
        correlations = df_corr.corrwith(df_image_stats, axis=1)

        # Create a bar plot to visualize the correlations
        correlation_plot = go.Bar(
            x=correlations.index,
            y=correlations.values
        )

        plot_layout = go.Layout(title='Correlations with Pre-Calculated Statistics', xaxis=dict(title='Statistic'),
                                yaxis=dict(title='Correlation'))

        return html.Div([
            html.H3('Uploaded Image:'),
            html.Img(src=content, style={'height': '300px'}),
        ]), {'data': [correlation_plot], 'layout': plot_layout}, html.Div([
            html.H3('Image Statistics:'),
            html.Table([
                html.Tr([html.Th(statistic), html.Td(value)]) for statistic, value in image_stats.items()
            ])
        ])
    else:
        return html.Div(), {}, html.Div()


if __name__ == '__main__':
    app.run_server(debug=True)


