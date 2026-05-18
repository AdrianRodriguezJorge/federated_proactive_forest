# Guía de Estilo y Rigurosidad Científica

* **Estilo**: Todo el código generado debe seguir estrictamente PEP 8 y estar completamente documentado con docstrings explicativos en cada función.
* **Fuga de datos (Data Leakage)**: Al generar o modificar pipelines de evaluación de Machine Learning o particionamiento de conjuntos de datos federados, asegurar que ningún dato de prueba/validación influya en el entrenamiento de los bosques locales o en el agregador global.
