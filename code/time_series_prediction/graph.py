import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os
from scipy.interpolate import interp1d
from celluloid import Camera


def plot_stats(h, stats_ds=None, stats_filename=None, savefig_path=None, \
    flip_plot=False, color='blue', fig=None, axs=None, legend_title='', pointplot=False):
    '''Plot statistics from .csv file or from statictics DataFrame

    Parameters
    ----------
    stats_ds: pd.DataFrame
        columns: points_left, n_iterations, non_predictable, rmse, mape
    stats_filename: str
        Path to stats.csv file: pd.DataFrame with columns 
        points_left, n_iterations, non_predictable, rmse, mape
    savefig_path: str
        Path to save plot, None if saving is not needed
    '''
    if stats_ds is None and stats_filename is None:
        return 
    if stats_ds is None:
        stats_ds = pd.read_csv(stats_filename, index_col=0)
    if fig is None:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    xlabel = 'points left' if not flip_plot else 'points thrown'
    x = stats_ds['points_left'].values
    if flip_plot:
        x = h - x

    xticks = get_xticks(h)
    if pointplot:
        p1 = sns.pointplot(x=x, y=stats_ds['n_iterations'], ax=axs[0][0], color=color, label=legend_title, scale=0.5)
        axs[0][0].set(xlabel=xlabel, ylabel='iterations', \
            title='Number of iterations to converge', xticks=xticks)
        p1.axes.grid(True, axis='both')

        p2 = sns.pointplot(x=x, y=stats_ds['non_predictable'], ax=axs[0][1], color=color, label=legend_title, scale=0.5)
        axs[0][1].set(xlabel=xlabel, ylabel='non-predictable points', \
            title='Number of non-predicable points', xticks=xticks)
        p2.axes.grid(True, axis='both')

        p3 = sns.pointplot(x=x, y=stats_ds['rmse'], ax=axs[1][0], color=color, label=legend_title, scale=0.5)
        axs[1][0].set(xlabel=xlabel, ylabel='RMSE', title='RMSE', xticks=xticks)
        axs[1][0].legend()
        p3.axes.grid(True, axis='both')

        p4 = sns.pointplot(x=x, y=stats_ds['mape'], ax=axs[1][1], color=color, label=legend_title, scale=0.5)
        axs[1][1].set(xlabel=xlabel, ylabel='MAPE', title='MAPE', xticks=xticks)
        p4.axes.grid(True, axis='both')
    else:
        sns.lineplot(x=x, y=stats_ds['n_iterations'], ax=axs[0][0], color=color, label=legend_title)
        axs[0][0].set(xlabel=xlabel, ylabel='iterations', \
            title='Number of iterations to converge', xticks=xticks)

        sns.lineplot(x=x, y=stats_ds['non_predictable'], ax=axs[0][1], color=color, label=legend_title)
        axs[0][1].set(xlabel=xlabel, ylabel='non-predictable points', \
            title='Number of non-predicable points', xticks=xticks)

        sns.lineplot(x=x, y=stats_ds['rmse'], ax=axs[1][0], color=color, label=legend_title)
        axs[1][0].set(xlabel=xlabel, ylabel='RMSE', title='RMSE', xticks=xticks)

        sns.lineplot(x=x, y=stats_ds['mape'], ax=axs[1][1], color=color, label=legend_title)
        axs[1][1].set(xlabel=xlabel, ylabel='MAPE', title='MAPE', xticks=xticks)

    if savefig_path is not None:
        fig.savefig(savefig_path)

    # plt.show()

    return fig, axs


def plot_unified_and_possible_preds(up, pp, Y2, ax=None, \
    title='Unified and possible predictions', savefig_path=None):
    '''Plot unified predictions and corresponding possible predictions with true values
    for self-healing algorithm

    Parameters
    ----------
    up: list or 1D np.ndarray
        List of unified predictions
    pp: list of lists or list of np.ndarrays
        List of lists of possible predictions
    Y2: list or 1D np.ndarray
        True values of predicted points
    ax: plt.Axis
        Axis to plot on, for animation to work
    title: str
        Plot title
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 14))
    h = len(up)
    x = list(range(1, h + 1))
    predictable = np.argwhere(up != 'N').reshape(1, -1)[0]
    x = np.take(x, predictable)
    y = np.take(up, predictable)
    y = y.astype(float)
    non_pred = np.argwhere(up == 'N').reshape(1, -1)[0] + 1
    print(non_pred)

    # pp = np.delete(pp, axis=0)
    x2 = []
    y2 = []
    for i in range(len(pp)):
        x2.extend([i + 1] * len(pp[i]))
        y2.extend(pp[i])

    sns.scatterplot(x=x2, y=y2, color='green', ax=ax, linewidth=0, alpha=0.5, s=20)
    sns.scatterplot(x=x, y=y, color='red', ax=ax, linewidth=0, s=100)
    sns.lineplot(x=x, y=y, color='red', ax=ax, alpha=0.5)
    sns.lineplot(x=list(range(1, h+1)), y=Y2[:h], ax=ax, color='blue')
    sns.scatterplot(x=non_pred, y=[0] * len(non_pred), ax=ax, color='purple')

    ax.set(title=title, ylabel='value', xlabel='h')

    if savefig_path is not None:
        fig.savefig(savefig_path)

    plt.show()


def plot_healing_animation(working_path, Y2, title='Self-healing animation', \
    saveanim_path=None, savefig_path=None, up_logs=None, pp_logs=None, true_points=[]):
    '''Function for animated plot for self-healing
    each new frame is self-healing iteration
    In working folder must be presented files:
        unified_pred_logs.dat
        possible_pred_logs.dat
    '''

    if up_logs == None and pp_logs == None:
        with open(working_path + 'unified_pred_logs.dat', 'rb') as f:
            up_logs = pickle.load(f)

        with open(working_path + 'possible_pred_logs.dat', 'rb') as f:
            pp_logs = pickle.load(f)

    fig, ax = plt.subplots(figsize=(16, 10))
    n_frames = len(up_logs)
    h = len(up_logs[0])
    if len(true_points) > 0:
        true_x = np.argwhere(true_points != 'N').reshape(1, -1)[0]
    else:
        true_x = []

    camera = Camera(fig)
    ticks = list(range(0, h, 5))
    for i in range(n_frames):
        if len(true_points) > 0:
            sns.scatterplot(x=true_x + 1, y=np.take(true_points, true_x).astype(float), color='blue', s=50, ax=ax)
        plot_unified_and_possible_preds(up_logs[i], pp_logs[i], Y2, ax=ax, title=title)
        ax.set(xlabel='h', ylabel='value', xticks=ticks, title=title)
        camera.snap()

    anim = camera.animate(blit=True, interval=2000, repeat=True, repeat_delay=2000)
    if saveanim_path is not None:
        anim.save(saveanim_path)
    if savefig_path is not None:
        fig.savefig(savefig_path)


def get_xticks(h):
    if h <= 20:
        xticks = list(range(0, h + 1, 2))
    elif h <= 50:
        xticks = list(range(0, h + 1, 5))
    else:
        xticks = list(range(0, h + 1, 10))
    return xticks


def plot_errors(rmses, mapes, non_preds, exp_names=[], fig_title=''):
    '''Plot function for errors

    Parameters
    ----------
    rmses: list of lists
        List of RMSE lists for each experiment in comparison
    mapes: list of lists
        List of MAPE -//-
    non_preds: list of lists
        List of percentage of non-predictable points -//-
    '''
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 18))
    axd = plt.figure(constrained_layout=True, figsize=(18, 12)).subplot_mosaic(
        """
        AABB
        .CC.
        """
    )
    ax0 = axd['A']
    ax1 = axd['B']
    ax2 = axd['C']
    n_experiments = len(rmses)
    h = len(rmses[0])
    x = list(range(1, h + 1))
    colors = sns.color_palette('deep', 10)
    if len(exp_names) == 0:
        exp_names = [''] * n_experiments
    for i in range(n_experiments):
        sns.lineplot(x=x, y=rmses[i], color=colors[i], ax=ax0, label=exp_names[i])
        sns.lineplot(x=x, y=mapes[i], color=colors[i], ax=ax1, label=exp_names[i])
        sns.lineplot(x=x, y=non_preds[i], color=colors[i], ax=ax2, label=exp_names[i])
    
    ax0.set(xlabel='prediction horizon (h)', ylabel='RMSE', title='RMSE')
    ax1.set(xlabel='prediction horizon (h)', ylabel='MAPE', title='MAPE')
    ax2.set(xlabel='prediction horizon (h)', ylabel='Percentage of non-predictable', title='Percentage of non-predictable points')
    
    plt.show()

    return axd


def get_prediction_matrix_from_file(filepath, h_max, n_iterations):
    # format
    # h - test_i - unified_prediction - possible_predictions
    prediction_matrix = np.empty((h_max, n_iterations), dtype=object)
    with open(filepath, 'rb') as f:
        while True:
            try:
                (h, test_i) = pickle.load(f)
                unified_prediction = pickle.load(f)
                # possible_prediction = pickle.load(f)

                prediction_matrix[h - 1][test_i] = unified_prediction
            except EOFError:
                break
    return prediction_matrix


def get_errors_from_prediction_matrix(prediction_matrix, h_max, n_iterations, Y2):
    # h_max = prediction_matrix.shape[0]
    # n_iterations = prediction_matrix.shape[1]
    n_test_passed = 31 + h_max

    Y_true = Y2[n_test_passed:n_test_passed+n_iterations]
    # print(len(Y_true), n_test_passed+n_iterations)

    rmse = []
    mape = []
    npr = []
    for h in range(h_max):
        unified_preds = prediction_matrix[h][:n_iterations]
        predictable = np.argwhere(unified_preds != 'N')
        unified_preds_predicted = np.take(unified_preds, predictable).astype(float)
        if len(np.take(Y_true, predictable)) > 0:
            rmse.append(np.sqrt(mean_squared_error(np.take(Y_true, predictable), \
                unified_preds_predicted)))
            mape.append(mean_absolute_percentage_error(np.take(Y_true, predictable), \
                unified_preds_predicted))
        else:
            rmse.append(np.nan)
            mape.append(np.nan)
        npr.append(np.count_nonzero(unified_preds == 'N') / n_iterations)
    return rmse, mape, npr


def plot_experiment_results(working_directory, exp_short_names, \
    h_max, n_iterations, exp_names=None, fig_title=''):

    rmses = []
    mapes = []
    non_preds = []
    if exp_names is None:
        exp_names = exp_short_names
    for exp_short_name in exp_short_names:
        # prediction_matrix = get_prediction_matrix_from_file(\
        #     os.path.join(working_directory, 'predictions_' + exp_short_name + '.dat'),\
        #     h_max, n_iterations)
        with open(os.path.join(working_directory, 'pm_' + exp_short_name + '.dat'), 'rb') as f:
            prediction_matrix = pickle.load(f)
            prediction_matrix = np.delete(prediction_matrix, (0), axis=0)
        rmse, mape, npr = get_errors_from_prediction_matrix(prediction_matrix, h_max, n_iterations)
        rmses.append(rmse)
        mapes.append(mape)
        non_preds.append(npr)
    
    fig = plot_errors(rmses, mapes, non_preds, exp_names, fig_title)
    return fig


def plot_thrown_points_exp_results(working_directory, exp_short_names, \
    h_max, exp_names, fig_title, Y2, pointplot=False):
    colors = sns.color_palette('tab10', 10)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))
    fig.suptitle(fig_title)
    for i in range(len(exp_short_names)):
        stats_ds = pd.DataFrame(columns=['points_left', 'n_iterations', \
            'non_predictable', 'rmse', 'mape'])
        with open(os.path.join(working_directory, \
            'predictions_' + exp_short_names[i] + '.dat'), 'rb') as f:
            while True:
                try:
                    thrown = pickle.load(f)
                    unified_preds = pickle.load(f)
                    n_iterations = pickle.load(f)
                    predictable = np.argwhere(unified_preds != 'N').reshape(1, -1)[0]
                    # print(len(predictable), unified_preds)
                    # print(thrown, unified_preds, n_iterations)
                    if len(predictable) > 0:
                        stats_ds = stats_ds.append({
                            'points_left' : h_max - thrown, 
                            'n_iterations' : n_iterations, 
                            'non_predictable' : h_max - len(predictable), 
                            'rmse' : np.sqrt(mean_squared_error(np.take(Y2[:h_max], predictable), np.take(unified_preds, predictable))), 
                            'mape' : mean_absolute_percentage_error(np.take(Y2[:h_max], predictable), np.take(unified_preds, predictable))
                        }, ignore_index=True)
                    else:
                        stats_ds = stats_ds.append({
                            'points_left' : h_max - thrown, 
                            'n_iterations' : n_iterations, 
                            'non_predictable' : h_max - len(predictable), 
                            'rmse' : np.nan, 
                            'mape' : np.nan
                        }, ignore_index=True)
                except EOFError:
                    break

        fig, axs = plot_stats(h_max, stats_ds=stats_ds, \
            flip_plot=True, color=colors[i], fig=fig, axs=axs, \
                legend_title=exp_names[i], pointplot=pointplot) 
    return fig


def plot_healing_errors_progression(Y2, up_logs):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(27, 6))
    n_iterations = len(up_logs)
    h = len(up_logs[0])
    rmses = []
    mapes = []
    non_preds = []
    for i in range(n_iterations):
        predictable = np.argwhere(up_logs[i] != 'N').reshape(1, -1)[0]
        rmses.append(np.sqrt(mean_squared_error(np.take(up_logs[i], predictable), \
            np.take(Y2[:h], predictable))))
        mapes.append(mean_absolute_percentage_error(np.take(up_logs[i], predictable), \
            np.take(Y2[:h], predictable)))
        non_preds.append(h - len(predictable))

    sns.lineplot(x=list(range(1, n_iterations + 1)), y=rmses, ax=axs[0])
    sns.lineplot(x=list(range(1, n_iterations + 1)), y=mapes, ax=axs[1])
    sns.lineplot(x=list(range(1, n_iterations + 1)), y=non_preds, ax=axs[2])
    axs[0].set(xlabel='prediction horizon (h)', ylabel='RMSE', title='RMSE')
    axs[1].set(xlabel='prediction horizon (h)', ylabel='MAPE', title='MAPE')
    axs[2].set(xlabel='prediction horizon (h)', ylabel='Percentage of non-predictable', title='Percentage of non-predictable points')

    return fig
    

def plot_pp_logs_spreads(pp_logs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 8))
    color_palette = sns.color_palette('hls', 10)
    sns.scatterplot(x=[0] * len(pp_logs[1][25]), y=pp_logs[1][25], color=color_palette[0], ax=ax)
    for i in range(1, len(pp_logs[1:]) - 1):
        y = pp_logs[i + 1][25]
        y = y[len(pp_logs[i][25]):]
        # print(len(pp_logs[i+1][25]), len(pp_logs[i][25]))
        sns.scatterplot(x=[i] * len(y), y=y, color=color_palette[i], ax=ax)

    return fig


def plot_patterns(iteration, logs_23, label_mapping, up_logs, Y2):
    h = 100
    n_clusters = len(np.unique(list(label_mapping[23 + 100 * iteration].values())))
    fig, ax = plt.subplots(nrows=n_clusters, ncols=1, figsize=(16, 12 * n_clusters))

    for i in range(len(logs_23[iteration])):
        cluster = label_mapping[23 + 100 * iteration].get(logs_23[iteration][i][0], -1)
        pattern = logs_23[iteration][i][2]
        j = logs_23[iteration][i][1]
        x = [24] * 4
        for k in range(j):
            x[k] -= np.sum(pattern[k:j])
        for k in range(j+1,4):
            x[k] += np.sum(pattern[j:k])

        y = [up_logs[iteration + 1][x[k] - 1] for k in range(4)]
        y[j] = logs_23[iteration][i][0]
        # y = [logs_23[iteration][i][0]] * 4
        sns.scatterplot(x=x, y=y, ax=ax[cluster + 1], color='purple', s=30)
        sns.lineplot(x=x, y=y, ax=ax[cluster + 1], color='purple', alpha=0.2)

    
    for j in range(n_clusters):
        sns.lineplot(x=list(range(1, h+1)), y=Y2[:h], ax=ax[j], color='blue')
        pp = list(label_mapping[23 + 100 * iteration].keys())
        y2 = list(filter(lambda x: label_mapping[23 + 100 * iteration].get(x, -1) == j - 1, pp))
        x2 = [24] * len(y2)
        sns.scatterplot(x=x2, y=y2, color='green', ax=ax[j], linewidth=0, alpha=0.5, s=20)
        ax[j].set(title=f'Cluster {j - 1}', xlim=[0, 55]) #, ylim=[min(y2) - 0.1, max(y2) + 0.1])
        
    return fig