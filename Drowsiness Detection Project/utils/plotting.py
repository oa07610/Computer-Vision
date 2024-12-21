import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def annotate_points(ax, df, x_col, y_col):
    for idx, row in df.iterrows():
        ax.text(row[x_col]+0.2, row[y_col]+0.002, row['Model'], fontsize=10,
                color='black', weight='semibold')

def plot_evaluation_metrics(df):
    sns.set(style="whitegrid", context='talk')
    palette = sns.color_palette("Set2", n_colors=len(df))
    model_colors = dict(zip(df['Model'], palette))

    def plot_scatter(x, y, title, ylabel, filename):
        plt.figure(figsize=(8,6))
        ax = sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue='Model',
            palette=model_colors,
            s=150,
            edgecolor='black',
            legend=False
        )
        annotate_points(ax, df, x, y)
        plt.title(title, fontsize=16)
        plt.xlabel('Average Inference Time per Frame (ms)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    # Plot 1: Inference Time vs. mAP@0.5
    plot_scatter('Inference Time (ms)', 'mAP@0.5',
                 'Inference Time vs. mAP@0.5 for YOLO Models',
                 'mAP@0.5',
                 'Inference_Time_vs_mAP@0.5.png')

    # Plot 2: Inference Time vs. mAP@0.5:0.95
    plot_scatter('Inference Time (ms)', 'mAP@0.5:0.95',
                 'Inference Time vs. mAP@0.5:0.95 for YOLO Models',
                 'mAP@0.5:0.95',
                 'Inference_Time_vs_mAP@0.5_0.95.png')

    # Plot 3: Inference Time vs. Average Precision
    plot_scatter('Inference Time (ms)', 'Average Precision',
                 'Inference Time vs. Average Precision for YOLO Models',
                 'Average Precision',
                 'Inference_Time_vs_Average_Precision.png')

    # Plot 4: Inference Time vs. Average Recall
    plot_scatter('Inference Time (ms)', 'Average Recall',
                 'Inference Time vs. Average Recall for YOLO Models',
                 'Average Recall',
                 'Inference_Time_vs_Average_Recall.png')

    # Combined Plot
    df_melted = df.melt(
        id_vars=['Model', 'Inference Time (ms)'],
        value_vars=['mAP@0.5', 'mAP@0.5:0.95', 'Average Precision', 'Average Recall'],
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(8,6))
    g = sns.scatterplot(
        data=df_melted,
        x='Inference Time (ms)',
        y='Value',
        hue='Model',
        style='Metric',
        palette=model_colors,
        s=200,
        edgecolor='black'
    )

    for idx, row in df_melted.iterrows():
        plt.text(row['Inference Time (ms)']+0.2, row['Value']+0.002, row['Model'], fontsize=9)

    plt.title('Inference Time vs. Accuracy Metrics for YOLO Models', fontsize=18)
    plt.xlabel('Average Inference Time per Frame (ms)', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig('Inference_Time_vs_Accuracy_Metrics.png', dpi=300)
    plt.show()