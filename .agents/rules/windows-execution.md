# Reglas de Ejecución en Windows

* **Siempre usar cmd.exe /c**: Para cualquier ejecución de comandos, scripts o procesos, ejecutar usando: `cmd.exe /c "<comando>"`
* **Rutas con slash (`/`)**: Nunca usar backslashes (`\`) en rutas de archivos, siempre usar `/` para evitar problemas con la sandbox.
* **Sin buffering en Python**: Forzar salida inmediata en Python usando `python -u`.
* **Evitar shells interactivas y pipes sin encapsular**: Todo comando con redirecciones debe estar encapsulado en comillas dentro de `cmd.exe /c`.
