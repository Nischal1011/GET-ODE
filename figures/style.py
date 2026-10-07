'''Shared plotting style for every paper figure. GIL-ODE is always tab:blue (matches Fig. 1).'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {
    'GIL-ODE': 'tab:blue', 'LG-ODE': 'tab:orange', 'ODE-RNN': 'tab:green', 'Latent-ODE': 'tab:red',
    'Edge-GNN': 'tab:purple', 'RNN-NRI': 'tab:brown', 'CSG-ODE': 'tab:pink',
    'GIL-ODE (no lifting)': 'tab:cyan', 'GIL-ODE (causal)': 'tab:gray',
}
DATASETS = {'spring': 'Springs', 'charged': 'Charged', 'ieee39': 'IEEE 39-bus'}


def setup():
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 200, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def color(name):
    return COLORS.get(name, 'black')


def save(fig, path):
    fig.savefig(path)
    if path.endswith('.pdf'):
        fig.savefig(path[:-4] + '.png')
    print('wrote', path)
