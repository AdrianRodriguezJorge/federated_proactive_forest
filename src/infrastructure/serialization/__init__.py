"""
Infrastructure: Serialization

Módulo reservado para implementaciones de serialización de modelos y datos.

Posibles usos futuros:
──────────────────────
1. ForestSerializer
   - Serialización eficiente de bosques de árboles (ProactiveForest, ProgressiveForest).
   - Formatos soportados:
     • JSON: legible, para debugging e intercambio.
     • msgpack: binario compacto, rápido.
     • pickle/joblib: nativo de Python, mejor para sklearn.
   - Estructura JSON: {trees: [{split_feature, threshold, left, right, ...}],
                       metadata: {n_estimators, alpha, classes, ...}}

2. ModelVersioning
   - Versionado de modelos serializados.
   - Metadata incluida: versión del código, fecha, configuración, métricas.
   - Compatibilidad hacia atrás: poder cargar modelos de versiones anteriores.

3. CrossPlatformSerializer
   - Garantizar que modelos guardados en una máquina se puedan cargar en otra.
   - Manejo de diferencias de arquitectura (endianness, precisión float).
   - Útil para entornos heterogéneos en FL real.

4. CompressionPipeline
   - Compresión de modelos para reducir coste de comunicación cliente-servidor.
   - Técnicas:
     • Quantización de thresholds (float64 → float32).
     • Pruning de árboles poco relevantes.
     • gzip/zstd sobre el serializado.
   - Métrica: ratio de compresión vs pérdida de accuracy.

5. TreeProtocolBuffer
   - Definición de protobuf para árboles (interoperabilidad con otros sistemas).
   - Útil si se quiere exponer el FL como servicio gRPC.
   - Schema: message Tree { int32 feature = 1; double threshold = 2; ... }

6. ConfigSerializer
   - Serialización/deserialización de configuraciones de experimentos.
   - YAML ↔ dict ↔ objeto Config.
   - Validación de esquemas al cargar.

Notas de implementación:
- `joblib` es la opción más práctica para sklearn RandomForest.
- `msgpack` + `numpy` para eficiencia en comunicación FL.
- Para producción: considerar `onnx` si se necesita interoperabilidad.
- Los modelos serializados deben incluir siempre:
  • Versión del esquema de serialización.
  • Hiperparámetros del modelo.
  • Nombres de features y clases.
  • Métricas de evaluación.
"""
