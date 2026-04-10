"""Results Logger for Hyperparameter Optimization.

Persists Optuna study results to SQLite, CSV, and generates visualization plots.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
import yaml
import pandas as pd
import optuna


class OptimizationResultsLogger:
    """
    Saves optimization results to multiple formats and generates visualizations.

    Usage:
        logger = OptimizationResultsLogger('results/opt_S6_car')
        logger.save_study(study)
        logger.generate_plots(study)
    """

    def __init__(self, output_path: str):
        """
        Initialize results logger.

        Args:
            output_path: Directory to save results (will be created if not exists)
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_study(self, study: optuna.Study, metadata: Optional[dict] = None) -> dict:
        """
        Save complete study to CSV, YAML, and SQLite.

        Args:
            study: Optuna study object
            metadata: Additional metadata to save (strategy, dataset, etc.)

        Returns:
            Dict with paths to saved files
        """
        saved_files = {}

        # 1. Extract trials to DataFrame
        df_trials = study.trials_dataframe()

        # 2. Save to CSV
        csv_path = self.output_path / 'trials.csv'
        df_trials.to_csv(csv_path, index=False)
        saved_files['csv'] = str(csv_path)
        print(f"📄 Trials guardados a: {csv_path}")

        # 3. Save best config to YAML
        best_config = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'direction': study.direction.name,
        }
        if metadata:
            best_config['metadata'] = metadata

        yaml_path = self.output_path / 'best_config.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
        saved_files['yaml'] = str(yaml_path)
        print(f"📋 Best config guardado a: {yaml_path}")

        # 4. Save to SQLite (for complex queries)
        db_path = self.output_path / 'optimization.db'
        conn = sqlite3.connect(str(db_path))

        # Save trials
        df_trials.to_sql('trials', conn, if_exists='replace', index=False)

        # Save metadata
        if metadata:
            df_meta = pd.DataFrame([metadata])
            df_meta.to_sql('metadata', conn, if_exists='replace', index=False)

        # Save best params
        df_best = pd.DataFrame([study.best_params])
        df_best.to_sql('best_params', conn, if_exists='replace', index=False)

        conn.close()
        saved_files['sqlite'] = str(db_path)
        print(f"🗄️  Base de datos guardada a: {db_path}")

        # 5. Save study object itself (for later analysis)
        try:
            study_path = self.output_path / 'study.pkl'
            joblib = __import__('joblib')
            joblib.dump(study, str(study_path))
            saved_files['study_pickle'] = str(study_path)
            print(f"📦 Study object guardado a: {study_path}")
        except Exception as e:
            import logging
            logging.exception("Error saving study object")
            raise

        return saved_files

    def generate_plots(self, study: optuna.Study) -> dict:
        """
        Generate interactive visualization plots using Optuna + Plotly.

        Args:
            study: Optuna study object

        Returns:
            Dict with paths to generated HTML plots
        """
        plot_files = {}

        try:
            # 1. Optimization history (convergence curve)
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title="Optimization History - Convergence")
            path1 = self.output_path / 'optimization_history.html'
            fig1.write_html(str(path1), include_plotlyjs='cdn', full_html=True)
            plot_files['optimization_history'] = str(path1)
            print(f"📈 Optimization history: {path1}")

            # 2. Parameter importance
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title="Parameter Importance")
            path2 = self.output_path / 'param_importance.html'
            fig2.write_html(str(path2), include_plotlyjs='cdn', full_html=True)
            plot_files['param_importance'] = str(path2)
            print(f"📊 Parameter importance: {path2}")

            # 3. Parallel coordinate plot
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.update_layout(title="Parallel Coordinate Plot")
            path3 = self.output_path / 'parallel_coordinate.html'
            fig3.write_html(str(path3), include_plotlyjs='cdn', full_html=True)
            plot_files['parallel_coordinate'] = str(path3)
            print(f"🔀 Parallel coordinate: {path3}")

            # 4. Contour plot (interaction between params)
            if len(study.best_params) >= 2:
                fig4 = optuna.visualization.plot_contour(study)
                fig4.update_layout(title="Contour Plot - Parameter Interactions")
                path4 = self.output_path / 'contour.html'
                fig4.write_html(str(path4), include_plotlyjs='cdn', full_html=True)
                plot_files['contour'] = str(path4)
                print(f"🗺️  Contour plot: {path4}")

            # 5. Slice plot (individual param effects)
            fig5 = optuna.visualization.plot_slice(study)
            fig5.update_layout(title="Slice Plot - Individual Parameter Effects")
            path5 = self.output_path / 'slice.html'
            fig5.write_html(str(path5), include_plotlyjs='cdn', full_html=True)
            plot_files['slice'] = str(path5)
            print(f"📉 Slice plot: {path5}")

            # 6. Timeline plot (trial execution time)
            fig6 = optuna.visualization.plot_timeline(study)
            fig6.update_layout(title="Timeline - Trial Execution Time")
            path6 = self.output_path / 'timeline.html'
            fig6.write_html(str(path6), include_plotlyjs='cdn', full_html=True)
            plot_files['timeline'] = str(path6)
            print(f"⏱️  Timeline: {path6}")

        except Exception as e:
            print(f"⚠️  Error generando plots: {e}")
            print(f"   Esto puede deberse a dependencias faltantes o datos insuficientes")

        return plot_files

    def generate_summary_report(self, study: optuna.Study, metadata: Optional[dict] = None) -> str:
        """
        Generate a human-readable summary report.

        Args:
            study: Optuna study object
            metadata: Additional metadata

        Returns:
            Path to summary report file
        """
        # Calculate statistics
        df_trials = study.trials_dataframe()
        completed_trials = df_trials[df_trials['state'] == 'COMPLETE']

        if len(completed_trials) == 0:
            raise ValueError("No completed trials to summarize")

        # Build report
        report_lines = [
            "=" * 70,
            "OPTIMIZATION SUMMARY REPORT",
            "=" * 70,
            "",
            f"Strategy: {metadata.get('strategy', 'N/A') if metadata else 'N/A'}",
            f"Dataset: {metadata.get('dataset', 'N/A') if metadata else 'N/A'}",
            f"Metric: {metadata.get('metric', 'N/A') if metadata else 'N/A'}",
            "",
            "-" * 70,
            "RESULTS SUMMARY",
            "-" * 70,
            f"Total trials: {len(study.trials)}",
            f"Completed trials: {len(completed_trials)}",
            f"Pruned trials: {len(df_trials[df_trials['state'] == 'PRUNED'])}",
            f"Failed trials: {len(df_trials[df_trials['state'] == 'FAIL'])}",
            "",
            "-" * 70,
            "BEST RESULT",
            "-" * 70,
            f"Best trial: #{study.best_trial.number}",
            f"Best {metadata.get('metric', 'metric')}: {study.best_value:.6f}",
            "",
            "Best hyperparameters:",
        ]

        for param, value in study.best_params.items():
            report_lines.append(f"  - {param}: {value}")

        report_lines.extend([
            "",
            "-" * 70,
            "STATISTICS (completed trials)",
            "-" * 70,
            f"Mean {metadata.get('metric', 'metric')}: {completed_trials['value'].mean():.6f}",
            f"Std {metadata.get('metric', 'metric')}: {completed_trials['value'].std():.6f}",
            f"Min {metadata.get('metric', 'metric')}: {completed_trials['value'].min():.6f}",
            f"Max {metadata.get('metric', 'metric')}: {completed_trials['value'].max():.6f}",
            f"Median {metadata.get('metric', 'metric')}: {completed_trials['value'].median():.6f}",
            "",
            "-" * 70,
            "TOP 5 TRIALS",
            "-" * 70,
        ])

        top5 = completed_trials.nlargest(5, 'value')
        for _, row in top5.iterrows():
            report_lines.append(f"\nTrial #{row['number']}: {row['value']:.6f}")
            for col in df_trials.columns:
                if col.startswith('params_'):
                    param_name = col.replace('params_', '')
                    report_lines.append(f"  {param_name}: {row[col]}")

        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.output_path / 'summary_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n📝 Summary report: {report_path}")
        print("\n" + report_text)

        return str(report_path)
