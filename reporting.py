"""
This file produces result for reporting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def grid_search(
    report_data='./checkpoints/grid_search/grid_search_summary.json',
    graph_filename='./checkpoints/grid_search/grid_search_results.pdf'
):
    df = pd.read_json(report_data)
    print('Best', df.iloc[df['val_loss'].argmin()])
    print('Worst', df.iloc[df['val_loss'].argmax()])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title("Val/Test MSE losses w.r.t $(nh_1, nh_2)$", fontsize=16, pad=12)

    nh_pairs = df[['nh1', 'nh2']].to_numpy()
    ticks, labels = [], []
    for idx, nh in enumerate(nh_pairs):
        ticks.append(idx)
        labels.append(f"({nh[0]}, {nh[1]})")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation='vertical')
    sns.lineplot(data=df[['val_loss', 'test_loss']], ax=ax, dashes=False, linewidth=1.2, markers=True)
    ax.set_ylabel("MSE Losses", fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)
    plt.setp(ax.spines.values(), color='#374151')
    plt.show()
    fig.savefig(graph_filename, bbox_inches='tight')


def reg_noise(
    report_data='./checkpoints/reg_noise/reg_noise_summary.json',
    graph_filename='./checkpoints/reg_noise/reg_noise_results.pdf'
):
    df = pd.read_json(report_data)
    s1 = df[df['sigma'] == 0.05]
    print('Best s=0.05', s1.iloc[s1['val_loss'].argmin()])
    print('Worst s=0.05', s1.iloc[s1['val_loss'].argmax()])

    s2 = df[df['sigma'] == 0.15]
    print('Best s=0.15', s2.iloc[s2['val_loss'].argmin()])
    print('Worst s=0.15', s2.iloc[s2['val_loss'].argmax()])

    s1 = df[df['sigma'] == 0.05]
    s2 = df[df['sigma'] == 0.15]

    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel('Hidden nodes in 2nd layer', fontsize=12)
    ax.set_ylabel('Losses', fontsize=12)
    ax.set_title('Val/Test losses w.r.t $\sigma$')

    plt.plot(x, s1[['val_loss']], label="val_loss - $\sigma = 0.05$", marker='o', linewidth=2., color='b')
    plt.plot(x, s1[['test_loss']], label="test_loss - $\sigma = 0.05$", marker='o', linewidth=2., color='r')
    plt.plot(x, s2[['val_loss']], label="val_loss - $\sigma = 0.15$", marker='o', linewidth=2., linestyle='--', color='b')
    plt.plot(x, s2[['test_loss']], label="test_loss - $\sigma = 0.15$", marker='o', linewidth=2., linestyle='--', color='r')
    plt.xticks(x, ["3", "5", "6", "9"])
    plt.legend()
    plt.setp(ax.spines.values(), color='#374151')
    plt.show()
    fig.savefig(graph_filename, bbox_inches='tight')


if __name__ == '__main__':
    # grid_search()
    reg_noise()
