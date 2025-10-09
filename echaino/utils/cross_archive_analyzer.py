# FILE: utils/cross_archive_analyzer.py
# PATH: D:\urchinScanner\utils\cross_archive_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO
from datetime import datetime
import logging
import gc
from typing import List, Dict, Tuple
import warnings
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, shapiro, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

from utils.visualization_utils import plot_lock, remove_extreme_outliers  # Import the outlier function

logger = logging.getLogger(__name__)


class CrossArchiveAnalyzer:
    """Analyzer for comparative metrics across archives with enhanced visualizations and statistics."""

    def __init__(self, merged_df: pd.DataFrame, metrics: List[str], selected_class: str = ''):
        self.merged_df = merged_df
        self.metrics = metrics
        self.selected_class = selected_class
        self.um_metrics_map = {
            'area': 'area_um2',
            'perimeter': 'perimeter_um',
            'diameter': 'diameter_um'
        }

        # Filter to relevant columns
        relevant_um_cols = [self.um_metrics_map[m] for m in metrics if m in self.um_metrics_map]
        if not relevant_um_cols:
            logger.warning("No valid metrics selected; analysis will be empty")
        relevant_cols = ['archive'] + relevant_um_cols
        available_cols = [col for col in relevant_cols if col in self.merged_df.columns]
        self.merged_df = self.merged_df[available_cols]

        # NEW: Assign default class columns if missing for legacy data
        if 'class_id' not in self.merged_df.columns:
            self.merged_df['class_id'] = 0  # Default for old data
            logger.info("Assigned default class_id=0 for legacy data in cross-analysis")
        if 'class_name' not in self.merged_df.columns:
            self.merged_df['class_name'] = "urchin"  # Default for old data

        # Apply outlier removal before any analysis
        self.outliers_removed = 0
        if not self.merged_df.empty:
            original_count = len(self.merged_df)

            # Remove extreme outliers using the same function from visualization_utils
            columns_to_check = [col for col in ['diameter_um'] if col in self.merged_df.columns]
            if columns_to_check:
                logger.info(f"Applying outlier removal to {columns_to_check}")
                self.merged_df = remove_extreme_outliers(self.merged_df, columns_to_check, multiplier=3.0)
                self.outliers_removed = original_count - len(self.merged_df)
                logger.info(f"Removed {self.outliers_removed} extreme outliers from {original_count} total records")

        # Set enhanced visual style
        sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.family'] = 'sans-serif'

        logger.debug(f"Initialized with merged_df shape: {self.merged_df.shape} (after outlier removal)")

    def aggregate_data(self) -> pd.DataFrame:
        """Compute summary stats per archive for selected metrics."""
        if self.merged_df.empty or self.merged_df.drop(columns='archive', errors='ignore').empty:
            logger.warning("Empty or no metric columns in merged DF; returning empty summary")
            return pd.DataFrame(columns=['archive'])

        group_cols = ['archive']
        agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']

        numeric_cols = self.merged_df.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            logger.warning("No numeric columns for aggregation")
            return pd.DataFrame(self.merged_df['archive'].unique(), columns=['archive'])

        summary = self.merged_df.groupby(group_cols)[numeric_cols].agg(agg_funcs)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        # Add coefficient of variation (CV) for variability assessment
        for metric in self.um_metrics_map.values():
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in summary.columns and std_col in summary.columns:
                summary[f'{metric}_cv'] = (summary[std_col] / summary[mean_col] * 100).fillna(0)

                # Percent difference from overall mean
                overall_mean = self.merged_df[metric].mean(skipna=True)
                if not pd.isna(overall_mean) and overall_mean != 0:
                    summary[f'{metric}_pct_diff'] = (
                            (summary[mean_col] - overall_mean) / overall_mean * 100
                    ).fillna(0)

        return summary

    def perform_statistical_tests(self) -> Dict:
        """Perform comprehensive statistical tests between archives."""
        results = {
            'normality_tests': {},
            'between_group_tests': {},
            'pairwise_comparisons': {},
            'effect_sizes': {}
        }

        unique_archives = self.merged_df['archive'].unique()
        if len(unique_archives) < 2:
            logger.warning("Need at least 2 archives for statistical comparisons")
            return results

        for metric_name, metric_col in self.um_metrics_map.items():
            if metric_col not in self.merged_df.columns:
                continue

            # Prepare data by archive
            archive_data = {}
            for archive in unique_archives:
                data = self.merged_df[self.merged_df['archive'] == archive][metric_col].dropna()
                if len(data) >= 3:  # Need at least 3 samples for tests
                    archive_data[archive] = data.values

            if len(archive_data) < 2:
                continue

            # 1. Normality tests for each archive
            normality_results = {}
            all_normal = True
            for archive, data in archive_data.items():
                if len(data) >= 3:
                    stat, p_value = shapiro(data)
                    normality_results[archive] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                    if p_value <= 0.05:
                        all_normal = False
            results['normality_tests'][metric_name] = normality_results

            # 2. Between-group test
            data_arrays = list(archive_data.values())
            if all_normal and len(data_arrays) >= 2:
                # One-way ANOVA for normal data
                stat, p_value = f_oneway(*data_arrays)
                test_name = 'One-way ANOVA'
            else:
                # Kruskal-Wallis for non-normal data
                stat, p_value = kruskal(*data_arrays)
                test_name = 'Kruskal-Wallis'

            results['between_group_tests'][metric_name] = {
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # 3. Pairwise comparisons if overall test is significant
            if p_value < 0.05 and len(archive_data) > 2:
                pairwise_results = {}
                archive_names = list(archive_data.keys())

                for i, j in itertools.combinations(range(len(archive_names)), 2):
                    archive1, archive2 = archive_names[i], archive_names[j]
                    data1, data2 = archive_data[archive1], archive_data[archive2]

                    # Mann-Whitney U test (works for both normal and non-normal)
                    stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')

                    # Bonferroni correction
                    n_comparisons = len(list(itertools.combinations(archive_names, 2)))
                    corrected_p = min(p_val * n_comparisons, 1.0)

                    # Cohen's d effect size
                    pooled_std = np.sqrt((np.std(data1, ddof=1) ** 2 + np.std(data2, ddof=1) ** 2) / 2)
                    cohen_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                    pairwise_results[f"{archive1} vs {archive2}"] = {
                        'statistic': stat,
                        'p_value': p_val,
                        'corrected_p': corrected_p,
                        'cohen_d': cohen_d,
                        'significant': corrected_p < 0.05
                    }

                results['pairwise_comparisons'][metric_name] = pairwise_results

        return results

    def generate_visualizations(self) -> Dict[str, str]:
        """Generate enhanced comparative plots with modern aesthetics."""
        plots = {}
        unique_archives = self.merged_df['archive'].unique()
        if len(unique_archives) < 2:
            logger.warning("Need at least 2 archives for comparisons; skipping plots")
            return plots

        # Sort archives for consistent ordering
        sorted_archives = sorted(unique_archives)

        # Define custom color palette
        n_archives = len(unique_archives)
        colors = sns.color_palette("husl", n_archives)
        archive_colors = dict(zip(sorted_archives, colors))  # Zip with sorted for consistency

        with plot_lock:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # 1. Enhanced violin plots with strip plots for each metric
                for metric_name, metric_col in self.um_metrics_map.items():
                    if metric_name in self.metrics and metric_col in self.merged_df.columns:
                        try:
                            metric_data = self.merged_df[['archive', metric_col]].dropna()
                            if metric_data.empty:
                                continue

                            fig, ax = plt.subplots(figsize=(12, 8))

                            # Violin plot with inner quartiles
                            sns.violinplot(data=metric_data, x='archive', y=metric_col,
                                           palette=archive_colors, inner='quartile',
                                           linewidth=1.5, ax=ax, order=sorted_archives)

                            # Overlay strip plot for individual points
                            sns.stripplot(data=metric_data, x='archive', y=metric_col,
                                          size=3, alpha=0.5, color='black', ax=ax, order=sorted_archives)

                            # Customize plot
                            unit = 'µm²' if metric_name == 'area' else 'µm'
                            ax.set_title(f'{metric_name.capitalize()} Distribution Across Archives',
                                         fontsize=16, fontweight='bold', pad=20)
                            ax.set_xlabel('Archive', fontsize=12)
                            ax.set_ylabel(f'{metric_name.capitalize()} ({unit})', fontsize=12)

                            # Rotate x-labels
                            plt.xticks(rotation=45, ha='right')

                            # Add mean markers
                            means = metric_data.groupby('archive')[metric_col].mean()
                            for i, (archive, mean_val) in enumerate(means.items()):
                                ax.hlines(mean_val, i - 0.3, i + 0.3, colors='red',
                                          linestyles='--', linewidth=2)

                            # Add overall mean line
                            overall_mean = metric_data[metric_col].mean()
                            ax.axhline(overall_mean, color='black', linestyle=':',
                                       linewidth=1.5, alpha=0.7,
                                       label=f'Overall Mean: {overall_mean:.2f} {unit}')

                            # Add sample sizes
                            y_min = ax.get_ylim()[0]
                            for i, archive in enumerate(sorted_archives):
                                count = len(metric_data[metric_data['archive'] == archive])
                                ax.text(i, y_min * 0.95, f'n={count}',
                                        ha='center', va='top', fontsize=10,
                                        bbox=dict(boxstyle="round,pad=0.3",
                                                  facecolor='white', alpha=0.8))

                            ax.legend(loc='upper right')
                            plt.tight_layout()
                            plots[f'violin_{metric_name}'] = self._fig_to_base64(fig)
                            plt.close(fig)

                        except Exception as e:
                            logger.error(f"Error generating violin plot for {metric_name}: {e}")

                # 2. Ridge plot for all metrics combined
                try:
                    melt_cols = [col for col in self.um_metrics_map.values()
                                 if col in self.merged_df.columns]
                    if melt_cols:
                        # Normalize each metric to 0-1 scale for comparison
                        normalized_df = self.merged_df.copy()
                        for col in melt_cols:
                            col_min = normalized_df[col].min()
                            col_max = normalized_df[col].max()
                            if col_max > col_min:
                                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)

                        long_df = pd.melt(normalized_df, id_vars=['archive'],
                                          value_vars=melt_cols, var_name='metric', value_name='value')
                        long_df = long_df.dropna()

                        if not long_df.empty:
                            # Create ridge plot
                            g = sns.FacetGrid(long_df, row='archive', hue='archive',
                                              aspect=15, height=1.2, palette=archive_colors,
                                              row_order=sorted_archives)

                            # Draw densities
                            g.map(sns.kdeplot, 'value', bw_adjust=0.5,
                                  clip_on=False, fill=True, alpha=0.6, linewidth=1.5)
                            g.map(sns.kdeplot, 'value', bw_adjust=0.5,
                                  clip_on=False, color="black", linewidth=1)

                            # Iterate over axes to customize
                            def label(x, color, label):
                                ax = plt.gca()
                                ax.text(0, .2, label, fontweight="bold", color=color,
                                        ha="left", va="center", transform=ax.transAxes)

                            g.map(label, 'value')

                            # Remove axes details
                            g.set_titles("")
                            g.set(yticks=[])
                            g.despine(bottom=True, left=True)

                            # Add title and labels
                            g.fig.suptitle('Normalized Distribution Ridge Plot',
                                           fontsize=16, fontweight='bold', y=0.98)
                            g.set_xlabels('Normalized Value (0-1)', fontsize=12)

                            plots['ridge_plot'] = self._fig_to_base64(g.fig)
                            plt.close(g.fig)

                except Exception as e:
                    logger.error(f"Error generating ridge plot: {e}")

                # 3. Enhanced scatter plot with marginal distributions
                if 'area' in self.metrics and 'diameter' in self.metrics:
                    scatter_cols = ['area_um2', 'diameter_um']
                    if all(col in self.merged_df.columns for col in scatter_cols):
                        scatter_df = self.merged_df[['archive'] + scatter_cols].dropna()
                        if not scatter_df.empty:
                            try:
                                # Create joint plot
                                g = sns.JointGrid(data=scatter_df, x='area_um2', y='diameter_um',
                                                  height=10, ratio=5, space=0.2)

                                # Main scatter plot
                                g.plot_joint(sns.scatterplot, hue=scatter_df['archive'],
                                             palette=archive_colors, s=50, alpha=0.7,
                                             hue_order=sorted_archives)

                                # Marginal plots
                                for archive in sorted_archives:
                                    archive_data = scatter_df[scatter_df['archive'] == archive]
                                    g.ax_marg_x.hist(archive_data['area_um2'], bins=30,
                                                     alpha=0.5, color=archive_colors[archive])
                                    g.ax_marg_y.hist(archive_data['diameter_um'], bins=30,
                                                     alpha=0.5, color=archive_colors[archive],
                                                     orientation='horizontal')

                                # Add regression line
                                g.plot_joint(sns.regplot, scatter=False,
                                             color='black', line_kws={'linewidth': 2})

                                # Calculate correlation
                                corr = scatter_df[scatter_cols].corr().iloc[0, 1]
                                g.ax_joint.text(0.05, 0.95, f'r = {corr:.3f}',
                                                transform=g.ax_joint.transAxes,
                                                fontsize=12, verticalalignment='top',
                                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                                g.set_axis_labels('Area (µm²)', 'Diameter (µm)', fontsize=12)
                                g.fig.suptitle('Area vs Diameter with Marginal Distributions',
                                               fontsize=16, fontweight='bold', y=1.02)

                                # Move legend
                                g.ax_joint.legend(bbox_to_anchor=(1.1, 1), loc='upper left')

                                plt.tight_layout()
                                plots['joint_scatter'] = self._fig_to_base64(g.fig)
                                plt.close(g.fig)

                            except Exception as e:
                                logger.error(f"Error generating joint plot: {e}")

                # 4. Correlation heatmap if multiple metrics
                if len(self.metrics) > 1:
                    try:
                        metric_cols = [self.um_metrics_map[m] for m in self.metrics
                                       if m in self.um_metrics_map and self.um_metrics_map[m] in self.merged_df.columns]
                        if len(metric_cols) >= 2:
                            fig, ax = plt.subplots(figsize=(8, 6))

                            # Calculate correlation matrix
                            corr_matrix = self.merged_df[metric_cols].corr()

                            # Create heatmap
                            sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                                        cmap='coolwarm', center=0,
                                        square=True, linewidths=1,
                                        cbar_kws={"shrink": 0.8},
                                        ax=ax)

                            # Customize
                            ax.set_title('Correlation Matrix of Selected Metrics',
                                         fontsize=16, fontweight='bold', pad=20)

                            # Clean up labels
                            clean_labels = [m.capitalize().replace('_', ' ') for m in self.metrics]
                            ax.set_xticklabels(clean_labels, rotation=45, ha='right')
                            ax.set_yticklabels(clean_labels, rotation=0)

                            plt.tight_layout()
                            plots['correlation_heatmap'] = self._fig_to_base64(fig)
                            plt.close(fig)

                    except Exception as e:
                        logger.error(f"Error generating correlation heatmap: {e}")

        gc.collect()
        return {k: v for k, v in plots.items() if v}

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 PNG."""
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error in fig_to_base64: {e}")
            return ''

    def _format_statistical_results(self, stats_results: Dict) -> str:
        """Format statistical test results as HTML."""
        html = "<div class='stats-section'>"
        html += "<h2>Statistical Analysis</h2>"

        # Normality tests
        html += "<div class='stats-subsection'>"
        html += "<h3>Normality Tests (Shapiro-Wilk)</h3>"
        html += "<table class='stats-table'>"
        html += "<tr><th>Metric</th><th>Archive</th><th>W-statistic</th><th>p-value</th><th>Distribution</th></tr>"

        for metric, results in stats_results['normality_tests'].items():
            for archive, test in results.items():
                dist_type = "Normal" if test['is_normal'] else "Non-normal"
                p_formatted = f"{test['p_value']:.4f}" if test['p_value'] > 0.0001 else "<0.0001"
                html += f"<tr><td>{metric}</td><td>{archive}</td>"
                html += f"<td>{test['statistic']:.4f}</td><td>{p_formatted}</td>"
                html += f"<td class='{'normal' if test['is_normal'] else 'non-normal'}'>{dist_type}</td></tr>"

        html += "</table>"
        html += "<p class='interpretation'>Note: p-value > 0.05 indicates normal distribution</p>"
        html += "</div>"

        # Between-group tests
        html += "<div class='stats-subsection'>"
        html += "<h3>Between-Group Comparisons</h3>"
        html += "<table class='stats-table'>"
        html += "<tr><th>Metric</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Result</th></tr>"

        for metric, test in stats_results['between_group_tests'].items():
            result = "Significant difference" if test['significant'] else "No significant difference"
            p_formatted = f"{test['p_value']:.4f}" if test['p_value'] > 0.0001 else "<0.0001"
            result_class = 'significant' if test['significant'] else 'not-significant'

            html += f"<tr><td>{metric}</td><td>{test['test']}</td>"
            html += f"<td>{test['statistic']:.4f}</td><td>{p_formatted}</td>"
            html += f"<td class='{result_class}'>{result}</td></tr>"

        html += "</table>"
        html += "<p class='interpretation'>Note: p-value < 0.05 indicates significant difference between archives</p>"
        html += "</div>"

        # Pairwise comparisons
        if stats_results['pairwise_comparisons']:
            html += "<div class='stats-subsection'>"
            html += "<h3>Post-hoc Pairwise Comparisons (with Bonferroni correction)</h3>"

            for metric, comparisons in stats_results['pairwise_comparisons'].items():
                html += f"<h4>{metric.capitalize()}</h4>"
                html += "<table class='stats-table'>"
                html += "<tr><th>Comparison</th><th>U-statistic</th><th>p-value</th><th>Corrected p</th><th>Cohen's d</th><th>Effect Size</th><th>Result</th></tr>"

                for comp_name, comp_results in comparisons.items():
                    # Interpret Cohen's d
                    d = abs(comp_results['cohen_d'])
                    if d < 0.2:
                        effect = "Negligible"
                    elif d < 0.5:
                        effect = "Small"
                    elif d < 0.8:
                        effect = "Medium"
                    else:
                        effect = "Large"

                    result = "Significant" if comp_results['significant'] else "Not significant"
                    result_class = 'significant' if comp_results['significant'] else 'not-significant'

                    p_formatted = f"{comp_results['p_value']:.4f}" if comp_results['p_value'] > 0.0001 else "<0.0001"
                    corrected_p_formatted = f"{comp_results['corrected_p']:.4f}" if comp_results[
                                                                                        'corrected_p'] > 0.0001 else "<0.0001"

                    html += f"<tr><td>{comp_name}</td>"
                    html += f"<td>{comp_results['statistic']:.2f}</td>"
                    html += f"<td>{p_formatted}</td>"
                    html += f"<td>{corrected_p_formatted}</td>"
                    html += f"<td>{comp_results['cohen_d']:.3f}</td>"
                    html += f"<td>{effect}</td>"
                    html += f"<td class='{result_class}'>{result}</td></tr>"

                html += "</table>"

            html += "<p class='interpretation'>Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large effect</p>"
            html += "</div>"

        html += "</div>"
        return html

    def generate_report(self, root_dir: str) -> str:
        """Generate enhanced HTML report with modern styling and statistics."""
        project_root = os.path.dirname(root_dir) if 'archive' in os.path.basename(root_dir).lower() else root_dir
        reports_dir = os.path.join(project_root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_subdir = os.path.join(reports_dir, f'cross_archive_report_{timestamp}')
        os.makedirs(report_subdir, exist_ok=True)

        # Generate components
        summary_df = self.aggregate_data()
        summary_path = os.path.join(report_subdir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)

        plots = self.generate_visualizations()
        stats_results = self.perform_statistical_tests()

        # Build HTML with embedded CSS
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Cross-Archive Analysis Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }

        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 30px;
        }

        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }

        .header-info {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .header-info p {
            margin: 5px 0;
        }

        .outlier-info {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }

        .outlier-info strong {
            color: #856404;
        }

        .section {
            background-color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .visualization {
            margin: 30px 0;
            text-align: center;
        }

        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .stats-section {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-top: 40px;
        }

        .stats-subsection {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .stats-table {
            font-size: 14px;
        }

        .significant {
            color: #e74c3c;
            font-weight: bold;
        }

        .not-significant {
            color: #27ae60;
        }

        .normal {
            color: #27ae60;
        }

        .non-normal {
            color: #e67e22;
        }

        .interpretation {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 14px;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }

        @media (max-width: 768px) {
            .grid-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
"""

        # Header
        html += "<h1>Cross-Archive Analysis Report</h1>"
        html += "<div class='header-info'>"
        html += f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        if self.selected_class:
            html += f"<p><strong>Selected Class:</strong> {self.selected_class}</p>"
        archives_list = ', '.join(sorted(self.merged_df['archive'].unique())) if not self.merged_df.empty else 'None'
        html += f"<p><strong>Archives Analyzed:</strong> {len(self.merged_df['archive'].unique())}</p>"
        html += f"<p><strong>Archives:</strong> {archives_list}</p>"
        html += f"<p><strong>Metrics:</strong> {', '.join(self.metrics)}</p>"
        html += f"<p><strong>Total Samples (after outlier removal):</strong> {len(self.merged_df)}</p>"
        html += "</div>"

        # Outlier removal information
        if self.outliers_removed > 0:
            html += "<div class='outlier-info'>"
            html += f"<strong>Data Quality Note:</strong> {self.outliers_removed} extreme outliers were removed "
            html += "from the analysis using the IQR method (3.0x multiplier). These are likely computer vision errors."
            html += "</div>"

        # Summary Statistics
        html += "<div class='section'>"
        html += "<h2>Summary Statistics</h2>"
        if summary_df.empty:
            html += "<p>No summary data available.</p>"
        else:
            # Format the summary table for better readability
            formatted_summary = summary_df.copy()
            # Round numeric columns
            numeric_cols = formatted_summary.select_dtypes(include=[np.number]).columns
            formatted_summary[numeric_cols] = formatted_summary[numeric_cols].round(2)
            html += formatted_summary.to_html(index=False, classes='summary-table')
        html += "</div>"

        # Visualizations
        html += "<div class='section'>"
        html += "<h2>Visualizations</h2>"

        if not plots:
            html += "<p>No visualizations generated (insufficient data).</p>"
        else:
            # Individual metric plots in grid
            html += "<div class='grid-container'>"
            for metric in self.metrics:
                key = f'violin_{metric}'
                if key in plots:
                    html += "<div class='visualization'>"
                    html += f"<h3>{metric.capitalize()} Distribution</h3>"
                    html += f"<img src='data:image/png;base64,{plots[key]}'/>"
                    html += "</div>"
            html += "</div>"

            # Other plots
            if 'ridge_plot' in plots:
                html += "<div class='visualization'>"
                html += "<h3>Normalized Distribution Comparison</h3>"
                html += f"<img src='data:image/png;base64,{plots['ridge_plot']}'/>"
                html += "</div>"

            if 'joint_scatter' in plots:
                html += "<div class='visualization'>"
                html += "<h3>Bivariate Analysis</h3>"
                html += f"<img src='data:image/png;base64,{plots['joint_scatter']}'/>"
                html += "</div>"

            if 'correlation_heatmap' in plots:
                html += "<div class='visualization'>"
                html += "<h3>Metric Correlations</h3>"
                html += f"<img src='data:image/png;base64,{plots['correlation_heatmap']}'/>"
                html += "</div>"

        html += "</div>"

        # Statistical Analysis
        html += self._format_statistical_results(stats_results)

        # Footer
        html += """
    <div style='margin-top: 50px; padding: 20px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd;'>
        <p>Report generated by EchAIno Analyzer</p>
        <p style='font-size: 12px;'>Note: Extreme outliers were automatically removed to improve analysis quality.</p>
    </div>
</body>
</html>
"""

        # Save report
        html_path = os.path.join(report_subdir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Generated enhanced report at {html_path}")
        return html_path