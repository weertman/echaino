# FILE: utils\visualization_utils.py
# PATH: D:\urchinScanner\utils\visualization_utils.py

import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib

matplotlib.use(
    'Agg')  # Set non-interactive backend for thread safety (avoids GUI conflicts in multi-threaded environments)
import matplotlib.pyplot as plt
import seaborn as sns  # For enhanced plots (violin, pairplot) and colorblind-safe palettes
from typing import Dict, Optional, List
import numpy as np  # For calculations like skewness
from threading import Lock  # For explicit thread safety on shared resources (e.g., plotting)

logger = logging.getLogger(__name__)

# Global lock for matplotlib operations (ensures thread safety, as matplotlib isn't inherently thread-safe)
plot_lock = Lock()


def remove_extreme_outliers(df: pd.DataFrame, columns: List[str], multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove rows that are extreme outliers in any of the specified columns using IQR.

    Args:
        df: Input DataFrame.
        columns: List of column names to check for outliers (e.g., ['diameter_um', 'area_um2']).
        multiplier: IQR multiplier (3.0 for extreme outliers; use 1.5 for mild).

    Returns:
        Filtered DataFrame with outliers removed.
    """
    if df.empty:
        return df

    # NEW: Group by 'class_id' if present for per-class outlier removal
    if 'class_id' in df.columns:
        filtered_groups = []
        for class_id, group in df.groupby('class_id'):
            outlier_mask = pd.Series(False, index=group.index)  # Initialize mask per group
            for col in columns:
                if col in group.columns:
                    Q1 = group[col].quantile(0.25)
                    Q3 = group[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0:  # Avoid division by zero or degenerate cases
                        continue
                    lower = Q1 - multiplier * IQR
                    upper = Q3 + multiplier * IQR
                    col_outliers = (group[col] < lower) | (group[col] > upper)
                    outlier_mask = outlier_mask | col_outliers  # Union across columns per group
            filtered_group = group[~outlier_mask].copy()
            num_dropped = len(group) - len(filtered_group)
            logger.info(f"Dropped {num_dropped} extreme outliers from class {class_id} ({len(group)} rows)")
            filtered_groups.append(filtered_group)
        return pd.concat(filtered_groups) if filtered_groups else pd.DataFrame(columns=df.columns)
    else:
        # Original logic if no class_id
        outlier_mask = pd.Series(False, index=df.index)  # Initialize mask
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:  # Avoid division by zero or degenerate cases
                    continue
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                col_outliers = (df[col] < lower) | (df[col] > upper)
                outlier_mask = outlier_mask | col_outliers  # Union across columns

        filtered_df = df[~outlier_mask].copy()  # Filter and return copy to avoid warnings
        num_dropped = len(df) - len(filtered_df)
        logger.info(f"Dropped {num_dropped} extreme outliers from {len(df)} rows.")
        return filtered_df


def generate_report(df: pd.DataFrame, save_dir: str, options: Optional[Dict] = None):
    """
    Generate honest visualizations and text summary from measurements DataFrame, focused on micrometer (µm) scale.

    Adheres to principles of data visualization (Tufte/Few-inspired):
    - Clarity/Simplicity: Minimal non-data ink, standard plot types, light grids.
    - Integrity/Honesty: Axes from 0, no distortions, flag anomalies, colorblind-safe palettes. Excludes pixel-based metrics per user preference.
    - Data Density/Exploration: Small multiples/facets where possible, show full distributions (e.g., violins with outliers).
    - Context/Labeling: Integrated annotations (e.g., means, correlations), subtitles explaining insights.
    - Accessibility: Viridis colormap, high contrast.
    - Thread Safety: Uses 'Agg' backend and global lock for plotting to prevent concurrent access issues.

    Conversions (performed within this function):
    - If cm-based columns present (area_cm2, perimeter_cm), convert to µm: area_um2 = area_cm2 * 1e8, perimeter_um = perimeter_cm * 1e4.
    - Ignores pixel columns entirely (no reports/plots generated from them).
    - If no convertible columns, logs warning and skips relevant parts.

    Args:
        df: pandas DataFrame from measurements.csv (expected columns: id, box_id, area_cm2, perimeter_cm, confidence, source [optional]).
          Converts cm to µm internally.
        save_dir: Directory to save report files
        options: Optional dict for customization (e.g., {'plots': ['violin_area_um2', 'facet_histogram_perimeter_um', 'scatter_area_vs_perimeter_um', 'pairplot_key_metrics'], 'sample_size': 1000})

    Saves:
        - measurements_summary_{timestamp}.txt: Full statistical summary with insights (e.g., skewness flags, outlier detection) – µm only.
        - Individual PNG plots with timestamps and annotations – µm only.

    Skips plots if required columns missing; samples large DFs for readability; logs actions.
    """
    if df.empty:
        logger.warning("Empty DataFrame provided; skipping report generation")
        return

    # Default options if none provided – focused on µm plots only, including diameter
    if options is None:
        options = {
            'plots': [
                'violin_area_um2',
                'violin_diameter_um',  # Explicitly include diameter visualization
                'facet_histogram_perimeter_um',
                'scatter_area_vs_perimeter_um',
                'pairplot_key_metrics'
            ],
            'sample_size': 1000  # Sample for large DFs to avoid clutter
        }

    # Set seaborn style for simplicity and accessibility (minimal chartjunk, colorblind palette)
    sns.set(style="whitegrid", palette="viridis", context="notebook")

    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert cm to µm if columns present (integrity: explicit conversion with logging)
    if 'area_cm2' in df.columns:
        df['area_um2'] = df['area_cm2'] * 1e8  # 1 cm² = 10^8 µm²
        logger.info("Converted area_cm2 to area_um2")
    if 'perimeter_cm' in df.columns:
        df['perimeter_um'] = df['perimeter_cm'] * 1e4  # 1 cm = 10^4 µm
        logger.info("Converted perimeter_cm to perimeter_um")

    # NEW: Handle NaN/empty in converted columns (set to 0 for plotting; log count)
    um_cols = [col for col in df.columns if 'um' in col.lower()]
    for col in um_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count} NaN values in {col}; filling with 0 for visualization")
            df[col] = df[col].fillna(0)

    # Drop pixel columns if present (per user preference: no pixel reports)
    pixel_cols = [col for col in df.columns if 'pixels' in col]
    if pixel_cols:
        df = df.drop(columns=pixel_cols)
        logger.info(f"Dropped pixel-based columns: {pixel_cols}")

    # Remove extreme outliers
    #columns_to_check = [col for col in ['area_um2', 'perimeter_um', 'diameter_um'] if col in df.columns]
    columns_to_check = [col for col in ['diameter_um'] if col in df.columns]
    multiplier = 3.0
    if columns_to_check:
        logger.info(f"Performing IQR outlier filtering with multiplier={multiplier} on columns: {columns_to_check}")
        filtered_df = remove_extreme_outliers(df, columns_to_check, multiplier=multiplier)
        num_dropped = len(df) - len(filtered_df)
    else:
        filtered_df = df.copy()
        num_dropped = 0
        logger.warning("No columns to check for outliers; using original DataFrame")

    # Enhanced Text Summary: Honest stats + insights, µm-only (e.g., skewness, outliers, suggestions)
    summary_lines = []
    summary_lines.append("Measurements Summary Report (Micrometer Scale Only)")
    summary_lines.append("=================================================")
    summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total entries (original): {len(df)}")
    summary_lines.append(f"Total entries (after outlier removal): {len(filtered_df)}")
    summary_lines.append(f"Dropped extreme outliers: {num_dropped}")
    summary_lines.append("\nColumn Non-Null Counts (µm metrics only):")
    um_cols = [col for col in filtered_df.columns if 'um' in col.lower() or col in ['confidence', 'source', 'class_id']]  # NEW: Include class_id
    if not um_cols:
        summary_lines.append("No µm-convertible columns found; summary limited.")
    else:
        summary_lines.append(filtered_df[um_cols].notnull().sum().to_string())
        summary_lines.append("\nDescriptive Statistics (µm metrics only):")
        summary_lines.append(filtered_df[um_cols].describe().to_string())

    # NEW: If class_id present, add per-class stats
    if 'class_id' in filtered_df.columns:
        summary_lines.append("\nCounts by Class:")
        summary_lines.append(filtered_df['class_id'].value_counts().to_string())
        # Per-class descriptives
        per_class_desc = filtered_df.groupby('class_id')[um_cols].describe().to_string()
        summary_lines.append("\nPer-Class Descriptive Statistics:")
        summary_lines.append(per_class_desc)

    # Add correlations if relevant µm columns exist
    metric_cols = [col for col in ['area_um2', 'perimeter_um', 'diameter_um', 'confidence'] if
                   col in filtered_df.columns]  # Include diameter_um
    if len(metric_cols) >= 2:
        correlation = filtered_df[metric_cols].corr().to_string()
        summary_lines.append("\nCorrelations Between Key µm Metrics:")
        summary_lines.append(correlation)

    # Insights: Flag anomalies (e.g., skewness > 1 suggests long-tail; outliers via IQR)
    summary_lines.append("\nData Insights and Flags (µm scale):")
    for col in metric_cols:
        if col in filtered_df.columns:
            skew = filtered_df[col].skew(skipna=True)
            summary_lines.append(f"- {col}: Skewness = {skew:.2f}")
            if abs(skew) > 1:
                summary_lines.append(
                    f"  (High skewness; distribution may be long-tailed – check for measurement errors)")

            Q1, Q3 = filtered_df[col].quantile(0.25), filtered_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((filtered_df[col] < (Q1 - 1.5 * IQR)) | (filtered_df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                summary_lines.append(f"  ({outliers} potential outliers detected via IQR – investigate these objects)")

    # NEW: Per-class insights if class_id present
    if 'class_id' in filtered_df.columns:
        summary_lines.append("\nPer-Class Insights:")
        for class_id, group in filtered_df.groupby('class_id'):
            for col in metric_cols:
                if col in group.columns:
                    skew = group[col].skew(skipna=True)
                    summary_lines.append(f"- Class {class_id} - {col}: Skewness = {skew:.2f}")
                    if abs(skew) > 1:
                        summary_lines.append(f"  (High skewness in class {class_id}; possible class-specific issues)")

    if 'source' in filtered_df.columns:
        summary_lines.append("\nCounts by Source:")
        summary_lines.append(filtered_df['source'].value_counts().to_string())

    summary_lines.append(
        "\nSuggestions: Use plots below to explore µm distributions and relationships. If skewness is high, consider log transformations for analysis. No pixel metrics included per preference.")

    txt_path = os.path.join(save_dir, f'measurements_summary_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"Saved µm-focused text summary to {txt_path}")

    # Sample if large for plot readability (honest subsampling with warning)
    plot_df = filtered_df.copy()
    if len(plot_df) > options['sample_size']:
        plot_df = plot_df.sample(options['sample_size'], random_state=42)
        logger.info(f"Sampled {options['sample_size']} rows for plots to enhance readability (full data in summary)")

    # Generate plots with enhanced principles – µm only, within lock for thread safety
    with plot_lock:
        plt.close('all')

        for plot_type in options['plots']:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))  # Standard size

                # NEW: Common hue for class faceting if 'class_id' present
                hue = 'class_id' if 'class_id' in plot_df.columns else None

                if plot_type == 'violin_area_um2':
                    if 'area_um2' in plot_df.columns and not plot_df['area_um2'].dropna().empty:
                        sns.violinplot(data=plot_df, y='area_um2', ax=ax, hue=hue,  # NEW: hue by class
                                       inner='quartile')  # Shows full dist + quartiles
                        ax.set_title('Violin Plot of Area (µm²)')
                        ax.set_ylabel('Area (µm²)')
                        ax.set_ylim(0, plot_df['area_um2'].max() * 1.05)
                        # Annotation: Mean line
                        mean_val = plot_df['area_um2'].mean()
                        ax.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} µm²')
                        ax.legend()
                        ax.text(0.05, 0.95, 'Full distribution with quartiles; outliers visible',
                                transform=ax.transAxes, va='top', fontsize=10)
                    else:
                        logger.warning(
                            "Skipping violin_area_um2: 'area_um2' column missing or empty (conversion may have failed)")
                        plt.close(fig)
                        continue

                elif plot_type == 'violin_diameter_um':
                    if 'diameter_um' in plot_df.columns and not plot_df['diameter_um'].dropna().empty:
                        sns.violinplot(data=plot_df, y='diameter_um', ax=ax, hue=hue,  # NEW: hue by class
                                       inner='quartile')
                        ax.set_title('Violin Plot of Diameter (µm)')
                        ax.set_ylabel('Diameter (µm)')
                        ax.set_ylim(0, plot_df['diameter_um'].max() * 1.05)
                        # Annotation: Mean line
                        mean_val = plot_df['diameter_um'].mean()
                        ax.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} µm')
                        ax.legend()
                        ax.text(0.05, 0.95, 'Full distribution with quartiles; outliers visible',
                                transform=ax.transAxes, va='top', fontsize=10)
                    else:
                        logger.warning("Skipping violin_diameter_um: 'diameter_um' column missing or empty")
                        plt.close(fig)
                        continue

                elif plot_type == 'facet_histogram_perimeter_um':
                    if 'perimeter_um' in plot_df.columns and not plot_df['perimeter_um'].dropna().empty:
                        if 'source' in plot_df.columns and plot_df['source'].nunique() > 1:
                            # Small multiples for comparison (Tufte-inspired)
                            g = sns.FacetGrid(plot_df, col='source', col_wrap=3, height=4, aspect=1.5)
                            g.map(sns.histplot, 'perimeter_um', bins='auto', kde=True)  # Add KDE for smooth trend
                            g.set_titles(col_template="{col_name}")
                            g.set_xlabels('Perimeter (µm)')
                            g.set_ylabels('Frequency')
                            g.set(xlim=(0, plot_df['perimeter_um'].max() * 1.05))
                            g.fig.suptitle('Faceted Histogram of Perimeter (µm) by Source', y=1.02)
                            g.fig.text(0.5, -0.05, 'Compare distributions across sources; KDE shows density',
                                       ha='center', fontsize=10)
                            # Save faceted grid
                            plot_path = os.path.join(save_dir, f'{plot_type}_{timestamp}.png')
                            g.savefig(plot_path, dpi=300, bbox_inches='tight')
                            plt.close(g.fig)
                            logger.info(f"Saved {plot_type} to {plot_path}")
                            continue  # Skip standard save
                        else:
                            sns.histplot(data=plot_df, x='perimeter_um', bins='auto', kde=True, ax=ax)
                            ax.set_title('Histogram of Perimeter (µm) with KDE')
                            ax.set_xlabel('Perimeter (µm)')
                            ax.set_ylabel('Frequency')
                            ax.set_xlim(0, plot_df['perimeter_um'].max() * 1.05)
                            ax.text(0.05, 0.95, 'KDE overlays smooth distribution; no source for faceting',
                                    transform=ax.transAxes, va='top', fontsize=10)
                    else:
                        logger.warning(
                            "Skipping facet_histogram_perimeter_um: 'perimeter_um' column missing or empty (conversion may have failed)")
                        plt.close(fig)
                        continue

                elif plot_type == 'scatter_area_vs_perimeter_um':
                    if all(col in plot_df.columns for col in ['area_um2', 'perimeter_um']) and not plot_df[
                        'area_um2'].dropna().empty and not plot_df['perimeter_um'].dropna().empty:
                        sns.scatterplot(data=plot_df, x='area_um2', y='perimeter_um', ax=ax, alpha=0.7,
                                        edgecolor='white', hue=hue)  # NEW: hue by class
                        # Add regression line for trend (honest relationship view)
                        sns.regplot(data=plot_df, x='area_um2', y='perimeter_um', scatter=False, color='red', ax=ax)
                        ax.set_title('Scatter Plot: Area vs Perimeter (µm scale) with Regression')
                        ax.set_xlabel('Area (µm²)')
                        ax.set_ylabel('Perimeter (µm)')
                        ax.set_xlim(0, plot_df['area_um2'].max() * 1.05)
                        ax.set_ylim(0, plot_df['perimeter_um'].max() * 1.05)
                        # Annotation: Correlation
                        if len(plot_df) > 1:
                            corr = plot_df[['area_um2', 'perimeter_um']].corr().iloc[0, 1]
                            ax.text(0.05, 0.95, f'Correlation: {corr:.2f}\n(Positive suggests expected scaling)',
                                    transform=ax.transAxes, va='top', fontsize=10)
                    else:
                        logger.warning(
                            "Skipping scatter_area_vs_perimeter_um: Missing or empty 'area_um2' or 'perimeter_um' (conversion may have failed)")
                        plt.close(fig)
                        continue

                elif plot_type == 'pairplot_key_metrics':
                    metric_cols = [col for col in ['area_um2', 'perimeter_um', 'diameter_um', 'confidence'] if
                                   col in plot_df.columns and not plot_df[col].dropna().empty]
                    if len(metric_cols) >= 2:
                        g = sns.pairplot(plot_df[metric_cols], diag_kind='kde', plot_kws={'alpha': 0.6}, hue=hue)  # NEW: hue by class
                        g.fig.suptitle('Pairplot of Key µm Metrics (Scatter + KDE Diagonals)', y=1.02)
                        g.fig.text(0.5, -0.05, 'Explore relationships and distributions; correlations on off-diagonals',
                                   ha='center', fontsize=10)
                        # Save pairplot
                        plot_path = os.path.join(save_dir, f'{plot_type}_{timestamp}.png')
                        g.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(g.fig)
                        logger.info(f"Saved {plot_type} to {plot_path}")
                        continue
                    else:
                        logger.warning(
                            "Skipping pairplot_key_metrics: Need at least 2 non-empty µm metric columns (conversion may have failed)")
                        continue

                else:
                    logger.warning(f"Unknown plot type: {plot_type}; skipping")
                    plt.close(fig)
                    continue

                # Save standard plot (if not already saved)
                plot_path = os.path.join(save_dir, f'{plot_type}_{timestamp}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {plot_type} to {plot_path}")

            except Exception as e:
                logger.error(f"Error generating {plot_type}: {e}")
                plt.close()  # Clean up on error