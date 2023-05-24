import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def plot_shapley_scores(scores, features):
    plot_data_y_labs = features
    bar_data_y = [i for i in range(scores.shape[0])]

    yaxis_data = {
        'tickmode': 'array',
        'tickvals': bar_data_y,
        'ticktext': plot_data_y_labs,
        'title': ''
    }

    colors = ['rgba(0,128,128,0.5)' if score >= 0 else 'rgba(199,21,133, 0.5)' for score in scores]
    line_colors = ['rgba(0,128,128,1.0)' if score >= 0 else 'rgba(199,21,133, 1.0)' for score in scores]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=scores, y=bar_data_y, orientation='h',
              marker_color=colors, marker_line_color=line_colors, marker_line_width=2))

    fig.update_layout(yaxis=yaxis_data,
                      font_size=30, showlegend=False)
    return fig


if __name__ == '__main__':

    examples = ['census', 'titanic']

    for example in examples:
        example_path = Path(example)

        results_idx = []
        for res in os.listdir(example_path):
            if res != '.' and res != '..':
                results_idx.append(res.split('_')[-1])
        for res_idx in results_idx:
            res_path = example_path / f'res_{res_idx}'
            image_path = res_path / 'images'
            os.makedirs(image_path, exist_ok=True)

            features_census = ['age', 'fnlwgt', 'education.num', 'marital.status', 'sex', 'hours.per.week']
            features_titanic = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            features = features_census if example == 'census' else features_titanic

            orig_shapley_values = np.loadtxt(res_path / 'orig_shapley_values.txt')
            manipulated_shapley_values = np.loadtxt(res_path / 'manipulated_shapley_values.txt')
            orig_shapley_vals_median = np.loadtxt(res_path / 'orig_shapley_vals_median.txt')
            manip_shapley_vals_median = np.loadtxt(res_path / 'manip_shapley_vals_median.txt')

            fig = plot_shapley_scores(orig_shapley_values, features)
            fig.write_image(image_path / 'orig_shapley_values.png')

            fig = plot_shapley_scores(manipulated_shapley_values, features)
            fig.write_image(image_path / 'manipulated_shapley_values.png')

            fig = plot_shapley_scores(orig_shapley_vals_median, features)
            fig.write_image(image_path / 'orig_shapley_vals_median.png')

            fig = plot_shapley_scores(manip_shapley_vals_median, features)
            fig.write_image(image_path / 'manip_shapley_vals_median.png')
