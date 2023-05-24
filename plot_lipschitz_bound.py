import math
import plotly.graph_objects as go


def phi_2_2(all_n):
    alpha = []
    beta = []
    for n in all_n:
        sum_val_alpha = 0
        for k in range(n):
            sum_val_alpha += 1.0 / math.comb(n-1, k)

        alpha.append(2.0 * sum_val_alpha / (n ** 2))

        sum_val_beta = 0
        for k in range(n-1):
            v1 = math.comb(n-1, k+1)
            v2 = math.comb(n-1, k)
            sum_val_beta += math.comb(n - 2, k) * (1.0 / v1 - 1.0 / v2) ** 2
        beta.append(sum_val_beta / (n ** 2))

    y = [math.sqrt(alpha[n-3] + (n-1) * beta[n-3]) for n in all_n]
    return y


def phi_2_i(all_n):
    y = []
    for n in all_n:
        res = 0
        for k in range(n):
            res += 1.0 / (math.comb(n - 1, k))
        res *= 2.0
        res = math.sqrt(res) / n
        y.append(res)
    return y


if __name__ == '__main__':
    all_n = [i for i in range(3, 35)]

    phi_2_2_y = phi_2_2(all_n)
    phi_2_i_y = phi_2_i(all_n)
    phi_1_1_y = [1 for n in all_n]
    phi_i_i_y = [2 for n in all_n]
    phi_1_2_y = [1.0/math.sqrt(n) for n in all_n]
    phi_1_i_y = [1.0/n for n in all_n]

    bounds = [dict(label='$||\Phi||_{2,2}$', y=phi_2_2_y),
              dict(label='$||\Phi||_{2,\infty}$', y=phi_2_i_y),
              dict(label='$||\Phi||_{1,1}$', y=phi_1_1_y),
              dict(label='$||\Phi||_{\infty,\infty}$', y=phi_i_i_y),
              dict(label='$||\Phi||_{1,2}$', y=phi_1_2_y),
              dict(label='$||\Phi||_{1,\infty}$', y=phi_1_i_y),
              ]

    # line_color = 'teal'
    colorscale = ['navy', 'royalblue', 'grey', 'darkgoldenrod', 'goldenrod', 'darkgrey']
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'x', 'star']
    fig = go.Figure()
    for idx, vals in enumerate(bounds):
        fig.add_trace(
            go.Scatter(x=all_n, y=vals['y'], mode='lines+markers',
                       marker_color=colorscale[idx], marker_line_color=colorscale[idx], marker_symbol=f'{symbols[idx]}-open',
                       marker_size=7,
                       name=vals['label']))

    fig.update_layout(font_size=25, showlegend=True, xaxis_title='Number of features', yaxis_title='Bound',
                      paper_bgcolor='white', plot_bgcolor='white',
                      xaxis=dict(gridcolor='lightgrey', showline=True, linecolor='grey'),
                      yaxis=dict(gridcolor='lightgrey', showline=True, linecolor='grey'))
    fig.write_image('lipschitz_bounds.png')
