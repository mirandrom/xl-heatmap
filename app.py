import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import math

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from flask_caching import Cache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Dash app to visualize embedding similarities as "
                    "a heatmap with dynamic downsampling to handle large "
                    "amounts of embeddings."
    )
    parser.add_argument(
        "--path", "-p", type=str, help="Directory with pytorch embeddings saved as .pt "
                              "files, and vocabulary saved as newline-separated "
                              ".txt files with the same stem. Defaults to "
                              "current working directory.",
        default="."
    )
    parser.add_argument("--debug", "-d", action="store_true", help="enable debug mode")
    parser.add_argument("--cuda", "-c", action="store_true", help="use CUDA")
    args = parser.parse_args()
    return args.path, args.debug, args.cuda

dir, debug, cuda = parse_args()

if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# populate embedding dropdown with torch '.pt' files in the provided directory
fs = list(Path(dir).glob('**/*.pt'))
embeds_dropdown = [{'label': f.stem, 'value': str(f)} for f in fs]
embeds_dropdown += [{'label': 'random', 'value': 'random'}]
embeds = {str(f): torch.load(f, map_location=torch.device('cpu')) for f in fs}
embeds.update({'random': torch.rand(1000,768)})
# get vocabularies from '.txt' files with the same name as '.pt' files
vocabs = {str(f): Path(f.with_suffix('.txt')).read_text().split('\n') for f in fs}
vocabs.update({'random': [str(i) for i in range(1000)]})
vtoi = {k:{v:i for i,v in enumerate(vocab)} for k,vocab in vocabs.items()}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_THRESHOLD': 1,
})


@cache.memoize()
def get_cached_sims(evy, evx, sim):
    """Compute similarities or return cached result.

    :param evy: str representing y-axis embeddings (label in dropdown)
    :param evx: str representing x-axis embeddings (label in dropdown)
    :param sim: str representing similarity type  (label in dropdown)
    :return: torch.FloatTensor of shape (y,x) containing similarities.
    """
    ey, ex = embeds[evy].to(device), embeds[evx].to(device)
    with torch.no_grad():
        if sim == 'ip':
            sims = torch.matmul(ey, ex.t())
        elif sim == 'cos':
            e1 = F.normalize(ey, p=2, dim=-1)
            e2 = F.normalize(ex, p=2, dim=-1)
            sims = torch.matmul(e1, e2.t())
        elif sim == 'l2':
            sims = torch.sqrt(
                -2 * torch.matmul(ey, ex.t())
                + (ey**2).sum(axis=1)[...,None] + (ex**2).sum(axis=1)
            )
    sims = sims.cpu()
    cmin = sims.min().item()
    cmax = sims.max().item()
    return sims, cmin, cmax


app.layout = html.Div(children=[
    html.H1(children='XL Embedding Similarity Heatmap'),
    html.Div(children='''
        A dash application for visualizing embedding similarities 
        as a dynamically downsampled heatmap.
    '''),

    html.Div([
        html.Div([
            html.Label('Embeddings (y)'),
            dcc.Dropdown(id='embeds_y', options=embeds_dropdown,
                         value=embeds_dropdown[0]['value']),
        ], className="six columns"),
        html.Div([
            html.Label('Embeddings (x)'),
            dcc.Dropdown(id='embeds_x', options=embeds_dropdown,
                         value=embeds_dropdown[0]['value']),
        ], className="six columns"),
    ], className="row"),

    html.Div([
        html.Div([
            html.Label('Similarity/Distance'),
            dcc.Dropdown(
                id='sim',
                options=[
                    {'label': 'Cosine', 'value': 'cos'},
                    {'label': 'Inner Product', 'value': 'ip'},
                    {'label': 'Euclidean Distance', 'value': 'l2'},
                ],
                value='ip'
            ),
        ], className="six columns"),
        html.Div([
            html.Label('Resolution'),
            dcc.Input(id="res", type="number", value=100, min=2, max=10_000, debounce=True),
        ], className="six columns"),
    ], className="row"),

    html.Div([
        html.Div([
            html.Label('Search for y-axis embed'),
            dcc.Dropdown(id='ysearch', ),
        ], className="four columns"),
        html.Div([
            html.Label('Search for x-axis embed'),
            dcc.Dropdown(id='xsearch', ),
        ], className="four columns"),
        html.Div([
            html.Label('Search and update heatmap'),
            html.Button(id='search', n_clicks=0, children='Search'),
        ], className="four columns"),
    ], className="row"),

    dcc.Graph(
        id='xl-heatmap',
    ),
    dcc.Store(id='dragmode', data='zoom'),
])


def meanpool(a, yres, xres):
    """ Rebin a heatmap matrix to the specified resolution with mean pooling.

    :param a: a torch.FloatTensor array of the heatmap indexed by a[y][x]
    :param yres: desired resolution on the y axis
    :param xres: desired resolution on the x axis
    :return: The heatmap array a, rebinned to shape (y,x)
    """
    yres = min(yres, a.shape[0])
    xres = min(xres, a.shape[1])
    ky = math.ceil(a.shape[0] / yres)
    kx = math.ceil(a.shape[1] / xres)
    m = torch.nn.AvgPool2d((ky,kx), ceil_mode=True).to(device)
    with torch.no_grad():
        mean_pooled = m(a[None,None,...])[0][0]
    return mean_pooled.cpu()


def downsample(x, y, z, xrange, yrange, res):
    """ Downsample heatmap to specified xrange/yrange and resolution

    :param x: List of full x-axis coordinates (i.e. Embeddings (x) vocab)
    :param y: List of full y-axis coordinates (i.e. Embeddings (y) vocab)
    :param z: Heatmap where z[y][x] is the similarity between embeds y and x.
    :param xrange: List/Tuple of new start/end indices for x
    :param yrange: List/Tuple of new start/end indices for y
    :param res: Specified (y,x) resolution for downsampling
    :return:
    """
    zz = z[yrange[0]:yrange[1],xrange[0]:xrange[1]]
    zz = meanpool(zz, res, res)
    xspan = xrange[1] - xrange[0] + 1
    yspan = yrange[1] - yrange[0] + 1
    xstep = math.ceil(xspan / res)
    ystep = math.ceil(yspan / res)
    xi = list(range(xrange[0], xrange[1], xstep))
    yi = list(range(yrange[0], yrange[1], ystep))
    xx = [f"{x[i]}{' [...]' if xspan > res else ''}" for i in xi]
    yy = [f"{y[i]}{' [...]' if yspan > res else ''}" for i in yi]
    return xx,yy,zz


def get_new_range(relayout, fig, evy, evx, dragmode, res_change,
                  res, search, ysearch, xsearch, reset):
    """ Get new int range from float relayoutData (e.g. zoom/pan/axis-reset)

    :param relayout: plotly dash relayoutData object
    :param fig: plotly graph object figure containing previous xrange/yrange
    :param evy: str representing y-axis embeddings (to retrieve vocab size)
    :param evx: str representing x-axis embeddings (to retrieve vocab size)
    :param dragmode: plotly dash input to handle differences in dragmode
    :param res_change: event trigger to handle resolution change
    :param res: current resolution
    :param search: event trigger to handle embed search
    :param ysearch: state for y-axis embed search
    :param xsearch: state for x-axis embed search
    :param reset: event trigger to handle axis resetting
    :return:
    """
    # handle axis reset
    rescale = relayout and 'xaxis.autorange' in relayout and 'yaxis.autorange' in relayout
    refresh = relayout and 'autosize' in relayout
    reset = (not res_change) and (relayout is None or rescale or refresh or reset)
    if search:
        if xsearch:
            x = vtoi[evx][xsearch]
            x0 = max(0, x - int(res/2))
            x1 = min(len(vocabs[evx]) - 1, x + int(res/2) - 1)
        else:
            x0, x1 = 0, len(vocabs[evx]) - 1
        if ysearch:
            y = vtoi[evy][ysearch]
            y0 = max(0, y - int(res / 2))
            y1 = min(len(vocabs[evy]) - 1, y + int(res / 2) - 1)
        else:
            y0, y1 = 0, len(vocabs[evy]) - 1
    elif reset:
        x0, x1 = 0, len(vocabs[evx]) - 1
        y0, y1 = 0, len(vocabs[evy]) - 1
    elif dragmode in ['zoom', 'pan'] or res_change:
        old_x0 = vtoi[evx][fig['data'][0]['x'][0].replace(' [...]', '')]
        old_x1 = vtoi[evx][fig['data'][0]['x'][-1].replace(' [...]', '')]
        old_y0 = vtoi[evy][fig['data'][0]['y'][0].replace(' [...]', '')]
        old_y1 = vtoi[evy][fig['data'][0]['y'][-1].replace(' [...]', '')]
        old_xrange = old_x1 - old_x0 + 1
        old_yrange = old_y1 - old_y0 + 1
        old_xbinsize = int(old_xrange / (len(fig['data'][0]['x']) - 1))
        old_ybinsize = int(old_yrange / (len(fig['data'][0]['y']) - 1))
        if res_change:
            x0 = old_x0
            x1 = old_x1 + old_xbinsize
            y0 = old_y0
            y1 = old_y1 + old_ybinsize
        else:
            x0 = old_x0 + int(relayout['xaxis.range[0]'] * old_xbinsize) - 1
            x1 = old_x0 + int(relayout['xaxis.range[1]'] * old_xbinsize) + 1
            y0 = old_y0 + int(relayout['yaxis.range[0]'] * old_ybinsize) - 1
            y1 = old_y0 + int(relayout['yaxis.range[1]'] * old_ybinsize) + 1
    # not sure if there are other relayoutData-triggering events
    else:
        raise
    # ensure new xrange/yrange are not out-of-bounds
    x0 = max(x0, 0)
    x1 = min(x1, len(vocabs[evx])-1)
    y0 = max(y0, 0)
    y1 = min(y1, len(vocabs[evy])-1)

    return x0,x1,y0,y1

@app.callback(
    Output('xl-heatmap', 'figure'),
    Output('dragmode', 'data'),
    [Input('xl-heatmap', 'relayoutData'),
    Input('xl-heatmap', 'figure'),
    Input('dragmode', 'data'),
    Input('embeds_y', 'value'),
    Input('embeds_x', 'value'),
    Input('sim', 'value'),
    Input('res', 'value'),
    Input('search', 'n_clicks'),],
    State('xsearch', 'value'),
    State('ysearch', 'value'),
)
def update_figure(relayout, fig, dragmode, evy, evx, sim, res,
                  search_clicks, xsearch, ysearch):
    # store dragmode event to handle zoom/pan rescaling
    if relayout and 'dragmode' in relayout:
        return fig, relayout['dragmode']

    # load full cached data
    _zz, cmin, cmax = get_cached_sims(evy, evx, sim)
    _xt = vocabs[evx]
    _yt = vocabs[evy]

    # downsample data (and reset axes if sims are recomputed)
    triggered = [t['prop_id'] for t in dash.callback_context.triggered]
    res_change = 'res.value' in triggered
    search = 'search.n_clicks' in triggered
    recompute_sims = any([v in triggered for v in
                          ['embeds_y.value', 'embeds_x.value', 'sim.value']])
    x0,x1,y0,y1 = get_new_range(relayout, fig, evy, evx, dragmode, res_change,
                                res, search, ysearch, xsearch,
                                reset=recompute_sims)
    xt, yt, zz = downsample(_xt, _yt, _zz, [x0, x1], [y0, y1], res)

    fig = go.Figure(data=[dict(type='heatmap', x=xt, y=yt, z=zz, zmin=cmin, zmax=cmax)])
    if search:
        lines = []
        if xsearch:
            x = xt.index(xsearch)
            lines += [dict(type='line', x0=x, x1=x, y0=0, y1=len(yt)-1, )]
        if ysearch:
            y = yt.index(ysearch)
            lines += [dict(type='line', y0=y, y1=y, x0=0, x1=len(xt)-1)]
        fig.update_layout(shapes=lines)

    return fig, dragmode

@app.callback(
    Output('xsearch', 'options'),
    Input('embeds_x', 'value')
)
def update_xsearch(evx):
    return [{'label': v, 'value': v} for v in vocabs[evx]]

@app.callback(
    Output('ysearch', 'options'),
    Input('embeds_y', 'value')
)
def update_ysearch(evy):
    return [{'label': v, 'value': v} for v in vocabs[evy]]

if __name__ == '__main__':
    app.run_server(debug=debug)