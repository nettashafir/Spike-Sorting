import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def dummy(data, x=None, title=None, label=None, ylabel=None, xlabel=None):
    if x is not None:
        plt.plot(x, data, label=label)
    else:
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if label is not None:
        plt.legend()
    plt.show()


def smart(data, title=None):
    fig = make_subplots(rows=1, cols=1)
    for samples in data:
        fig.add_traces([go.Scatter(x=list(range(len(samples))), y=samples)])
    fig.update_layout(title_text=title)
    fig.show()
