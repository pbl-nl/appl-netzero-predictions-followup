import os
import textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from modules.config_loader import load_config


config = load_config()

# Distribution of indicators
# Function to plot a 4x4 grid of indicators
def plot_indicators_grid(df, start_index, end_index, fig_no):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))  # Size can be adjusted
    fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing to accommodate multiline titles
    fig.suptitle(f'Indicators Distribution - Figure {fig_no}', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        idx = start_index + i
        if idx < end_index:
            column_data = df.iloc[:, idx].dropna()  # Drop NA values for accurate plot and test
            sns.histplot(column_data, kde=True, ax=ax)
            wrapped_title = textwrap.fill(df.columns[idx], 35)  # Wrap titles longer than 25 characters
            ax.set_title(wrapped_title, fontsize=12)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlabel('')
            # Check if there are at least 3 non-null data points to perform Shapiro-Wilk Test (t-test)
            if len(column_data) >= 3:
                _, p_value = shapiro(column_data)
                normality_text = f"p = {p_value:.4f}" + (" (Normal)" if p_value > 0.05 else " (Not normal)")
                color = 'green' if p_value > 0.05 else 'red'
            else:
                normality_text = "Insufficient data"
                color = 'orange'

            ax.text(0.95, 0.95, normality_text, transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    color=color, fontsize=9)
        else:
            ax.axis('off')

    filename = os.path.join(config["output_path"], f"Indicators Distribution - Figure {fig_no}")
    plt.savefig(filename)
    # Close the figure to prevent it from showing when plt.show() is called
    plt.close(fig)


# Regression plot for every indicator in relation to a target variable
def plot_regression_grid(df, target_variable, start_index, end_index, fig_no):
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))  # Size can be adjusted
    fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing
    fig.suptitle(f'Regression Plots - Figure {fig_no}', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        idx = start_index + i
        if idx < end_index:
            # Assuming df.columns[idx] is an indicator and target_variable is the target
            sns.regplot(x=df.columns[idx], y=target_variable, data=df, ax=ax, scatter_kws={'alpha': 0.5})
            # Wrap titles longer than 25 characters
            wrapped_title = textwrap.fill(df.columns[idx], 35)
            ax.set_title(wrapped_title, fontsize=12)
            ax.set_ylabel(target_variable, fontsize=10)
            # Removing x-labels to prevent clutter
            ax.set_xlabel('')
        else:
            ax.axis('off')

    filename = os.path.join(config["output_path"], f"Regression Plots - Figure {fig_no}")
    plt.savefig(filename)
    plt.close(fig)
