"""
Infrastructure: Persistence

Módulo reservado para implementaciones de persistencia y checkpointing.

Posibles usos futuros:
──────────────────────
1. ModelCheckpoint
   - Guardar/cargar bosques globales y locales entre rondas.
   - Formatos: JSON (árboles serializados), pickle, joblib.
   - Metadata asociada: ronda, configuración, métricas, seed.
   - Ejemplo ruta: checkpoints/run_001/round_05/global_forest.pkl

2. ExperimentRegistry
   - Registro de experimentos con sus configuraciones y resultados.
   - Tabla: experiment_id, config_hash, dataset, strategy, metrics, timestamp.
   - Permite reproducibilidad y comparación entre runs.

3. MetricsHistory
   - Almacenamiento persistente de métricas por ronda/cliente.
   - SQLite/CSV/Parquet para análisis con pandas.
   - Series temporales: accuracy_t, f1_t, convergence_delta_t.

4. FederationStateManager
   - Estado completo de la federación para resumir entrenamientos:
     • Bosques locales de cada cliente.
     • Bosque global acumulado.
     • Historial de árboles seleccionados por ronda.
     • Semilla y estado del RNG.
   - Clave para experimentos largos que pueden interrumpirse.

5. ArtifactStore
   - Almacenamiento de artefactos: modelos finales, reportes, visualizaciones.
   - Versionado de modelos (MLflow-like).
   - Metadata: dataset_version, hyperparams, performance, author.

6. ResultsCache
   - Cache de resultados intermedios para evitar recomputación.
   - Hash de configuración → resultado almacenado.
   - Útil para grid search de hiperparámetros.

Notas de implementación:
- Para modelos: `joblib` es eficiente para Random Forests.
- Para métricas: SQLite + SQLAlchemy o directamente pandas.to_parquet().
- Considerar `mlflow` o `wandb` para tracking de experimentos.
- Los checkpoints deben incluir siempre la configuración completa del run.
- Importante: no persistir datos sensibles del dataset.
"""
