## **Fase 1 — Construcción local guiada por ruleta**

Cada cliente inicia el entrenamiento con un **Proactive Forest vacío**, cuyo mecanismo central es una **ruleta de atributos**. En esta ruleta, todos los atributos comienzan con la misma probabilidad de selección. El entrenamiento se organiza en ventanas de tamaño W, siendo el valor por defecto cinco árboles por ronda.

Durante la construcción de cada árbol, los atributos se eligen siguiendo las probabilidades de la ruleta, que se ajusta dinámicamente según el comportamiento de los árboles previamente construidos. Los atributos que aportan mayor diversidad y contribuyen a mejorar la eficacia incrementan su probabilidad de selección, mientras que aquellos redundantes o poco útiles la reducen. Por tanto, se logra un equilibrio entre eficacia y diversidad, evitando la generación innecesaria de árboles y asegurando que el modelo local evolucione de forma controlada.

---

## **Fase 2 — Comunicación y agregación federada**

Al finalizar cada ventana, el cliente transmite al servidor un **vector de probabilidades de atributos**, en lugar de los árboles completos. Este vector refleja la importancia relativa de cada atributo en el dataset local y constituye la información mínima necesaria para coordinar la federación.

El servidor recibe los vectores de todos los clientes y aplica un proceso de agregación federada. Se contemplan cuatro alternativas metodológicas:

1. **Promedio ponderado por tamaño de dataset:** favorece a clientes con más datos, reduciendo el sesgo de aquellos con muestras pequeñas.
2. **Media simple:** garantiza equidad básica, asignando el mismo peso a todos los clientes.
3. **Mediana (robusta):** evita que clientes con datos sesgados distorsionen la ruleta global.
4. **Consenso por eficacia (Macro-F1):** ajusta la influencia de cada cliente según su desempeño predictivo local.
5. **Consenso proactivo (PCD):** prioriza la contribución de clientes cuyos atributos fomentan una mayor diversidad estructural.

El resultado es una **ruleta global de atributos**, que sintetiza el conocimiento estadístico de toda la federación y se redistribuye a los clientes constituyendo un ejemplo de **orquestación ciega**, ya que el servidor nunca recibe modelos completos ni estructuras internas, sino únicamente vectores estadísticos. La comunicación se realiza mediante la **Infraestructura FLEX**, optimizando el ancho de banda y reforzando la privacidad al evitar el intercambio de árboles en esta fase. Esto asegura coordinación global con un coste de comunicación mínimo.

---
## **Fase 3 — Actualización local guiada por conocimiento global**

Una vez que el servidor redistribuye la ruleta global de atributos, cada cliente debe integrarla con su propia ruleta local. Este proceso se realiza mediante un esquema de ponderación controlado por el parámetro 𝛽. La fórmula general es:
$$
Ruleta^{(k)} = \beta \cdot Ruleta^{(local)} + (1 - \beta) \cdot Ruleta^{(global)} 
$$
donde:

- ***Ruleta^k:*** es la nueva ruleta que utilizará el cliente (k) para continuar su entrenamiento.
- ***Ruleta^local:*** representa la distribución de probabilidades de atributos que el cliente ha construido con sus propios datos hasta el momento.
- ***Ruleta^global:*** es la distribución agregada que el servidor ha generado a partir de todos los clientes.
- ***𝛽:*** es un hiperparámetro que regula el balance entre conocimiento local y global.

Por defecto, se establece 𝛽 = 0, lo que significa que la ruleta local se sustituye completamente por la global, asegurando una fuerte coordinación entre clientes. Sin embargo, este parámetro puede ajustarse para experimentar con diferentes grados de influencia: valores cercanos a 1 privilegian la especialización local, mientras que valores intermedios permiten un equilibrio entre la experiencia propia y el conocimiento compartido.

De esta manera, los clientes continúan entrenando nuevos árboles guiados por una ruleta que refleja tanto su aprendizaje individual como la relevancia global de los atributos en la federación.

---

## **Fase 4 — Criterio de parada**

El entrenamiento federado se organiza en rondas sucesivas de construcción de (W) árboles. Al inicio de cada una de estas rondas, el sistema evalúa los siguientes criterios para decidir la continuidad del proceso:

- **Convergencia local (Evaluación previa):** Antes de comenzar la construcción de un nuevo bloque de árboles, *Progressive Forest* evalúa la mejora en la eficacia del bosque local. Si la mejora es inferior al umbral de convergencia durante **dos episodios consecutivos**, el cliente deja de participar en el entrenamiento y pasa directamente a estar disponible para la fase de inferencia.
- **Límite global (Parada técnica):** Si cualquier cliente alcanza **T_max** (por defecto 100 árboles), el entrenamiento se detiene por completo para toda la federación, garantizando que el modelo no crezca indefinidamente y optimizando los recursos.

---

## **Fase 5 — Inferencia**

La inferencia se realiza de manera **local**. Cada cliente utiliza su propio bosque, entrenado bajo la guía de la ruleta global, para realizar predicciones. El resultado final se obtiene mediante la votación mayoritaria de los árboles que componen el bosque local. Este esquema de inferencia refuerza la independencia de los clientes, elimina latencias de consulta y asegura máxima privacidad, ya que los datos nunca salen del dispositivo.