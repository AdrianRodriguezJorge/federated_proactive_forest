# **Fase 1 — Entrenamiento local en los clientes**

Cada cliente entrena un **Proactive Forest (PF) local**, empleando el mecanismo de actualización de probabilidades regulado por el hiperparámetro de diversidad $\alpha$. La elección de PF frente a Random Forest (RF) se fundamenta en su capacidad demostrada para mejorar simultáneamente la eficacia predictiva y la diversidad del conjunto de árboles. La incorporación del mecanismo complementario **Progressive Forest** permite reducir el número total de árboles necesarios, lo que disminuye la carga de cómputo y memoria en dispositivos clientes con recursos limitados. Esta decisión metodológica asegura modelos locales compactos y eficientes sin sacrificar precisión.

---

# **Fase 2 — Comunicación de modelos al servidor**

Una vez completado el entrenamiento local, cada cliente transmite al servidor los **árboles completos** de su bosque junto con metadatos de diversidad y métricas de desempeño. Este procedimiento se alinea con las prácticas predominantes en arquitecturas cliente‑servidor de Aprendizaje Federado Horizontal (HFL), donde la comunicación de modelos completos se realiza de manera asincrónica debido a su mayor escalabilidad y a que resulta más adecuado en contextos de baja conectividad, donde los clientes no pueden depender de una sincronización estricta. La transmisión de árboles completos evita la necesidad de aplicar técnicas adicionales de compresión o transformación de parámetros en los clientes, reduciendo así su carga computacional.

Para la serialización y transmisión de bosques completos se emplea la **Infraestructura FLEX**, que garantiza la interoperabilidad y eficiencia en la comunicación de objetos del modelo. Esta elección asegura un equilibrio adecuado entre seguridad y rendimiento, y permite la integración con mecanismos de agregación segura en entornos que requieran mayores garantías de privacidad.

---

# **Fase 3 — Agregación global (estrategias S1–S7)**

El servidor construye el bosque federado aplicando siete **estrategias de agregación** que combinan enfoques de concatenación, ranking y selección progresiva para equilibrar eficacia, diversidad y equidad entre clientes.
==«El servidor dispone de un conjunto de validación global (independiente de los datos de los clientes) que se emplea para evaluar la convergencia en las estrategias progresivas.»==
1. **Concatenación simple (S1):** unión directa de todos los árboles locales sin criterios de selección.
2. **Listas globales ordenadas con Progressive Forest (S2–S4):** 
	- Incorporación progresiva de árboles ordenados según criterios de desempeño: exactitud, Macro‑F1 y promedio ponderado entre Macro‑F1 y diversidad PCD (*Pairwise Correlation Diversity*) respectivamente. 
	- En estas estrategias, la agregación se realiza en bloques sucesivos de _**k**_ árboles, donde el tamaño del bloque $k$ se define de forma estática (típicamente 5 árboles) para que coincida con la dinámica de crecimiento de los modelos locales. 
	- El servidor procesa la lista global y ejecuta el mecanismo de comprobación de convergencia de Progressive Forest: si la mejora en la eficacia es inferior al umbral durante **dos episodios consecutivos**, el proceso se detiene. 
	- **Re-ordenamiento proactivo (S4):** En el caso de la estrategia S4, el ranking no es estático; el servidor re-calcula la diversidad PCD de los candidatos restantes en cada episodio, priorizando dinámicamente aquellos árboles que mejor complementan al bosque global en construcción.
	- Este mecanismo garantiza consistencia metodológica con el entrenamiento local y evita la incorporación excesiva de árboles cuando el modelo ya ha estabilizado su capacidad predictiva.
3. **Listas por cliente con round‑robin y Progressive Forest (S5–S7):**
	- Cada cliente genera su propia lista ordenada y los árboles se incorporan de manera equitativa mediante round‑robin, según criterios de desempeño (exactitud, Macro‑F1 y promedio ponderado entre Macro‑F1 y diversidad PCD respectivamente), mitigando la sobre‑representación de clientes dominantes. 
	- En estas estrategias, la agregación se realiza incorporando un árbol por cliente activo en cada iteración (round-robin), siguiendo el orden de cada lista local.  
	- Tras cada ronda completa (un episodio de tamaño igual al número de clientes), el servidor evalúa la convergencia: si la mejora es inferior al umbral durante **dos episodios consecutivos**, se detiene la agregación. 
	- **Re-ordenamiento proactivo (S7):** Al igual que en S4, la estrategia S7 emplea un ranking dinámico que se actualiza tras cada episodio para asegurar que la selección round-robin considere la diversidad acumulada en el bosque federado.
	- En caso de que un cliente agote sus árboles disponibles, la ronda continúa con los clientes restantes hasta alcanzar la convergencia o agotar la totalidad de los árboles enviados. Este procedimiento asegura equidad entre clientes y evita que el modelo global crezca innecesariamente una vez alcanzada la estabilidad.

El uso de **Progressive Forest** incrementa el tiempo de construcción del bosque, pero produce modelos más compactos y eficientes durante la fase de clasificación, lo cual resulta beneficioso para clientes con recursos limitados. Las estrategias basadas en listas por cliente reducen el riesgo de sesgo en la representación global. La métrica **Macro‑F1** es especialmente relevante en escenarios con desbalance de clases, mientras que el criterio **PCD**, respaldado por literatura especializada, permite evaluar la diversidad del conjunto. El promedio ponderado entre Macro‑F1 y PCD proporciona un equilibrio adecuado entre eficacia y diversidad en la selección de árboles.

---

# **Fase 4 — Actualización del modelo en los clientes**

Tras la agregación global, cada cliente actualiza su bosque incorporando todos los árboles del modelo federado, excluyendo únicamente aquellos de origen local que ya fueron seleccionados durante la agregación. Este mecanismo evita duplicaciones y preserva la especialización local, al tiempo que incorpora la capacidad de generalización del modelo global. No se reemplaza el bosque local completo para evitar la pérdida de árboles que reflejen relaciones particulares de los datos de cada cliente. La exclusión de duplicados también previene sesgos derivados de otorgar un peso excesivo a ciertos árboles, aunque esta política puede ajustarse si se desea modificar la ponderación relativa entre bosque local y global.

---

# **Fase 5 — Inferencia**

La inferencia se realiza mediante una **votación híbrida ponderada**, combinando las predicciones del bosque local y del bosque global. El peso de la votación se distribuye mediante una ponderación por defecto de $w_{local} = 0.4$ y $w_{global} = 0.6$, asegurando un balance entre el conocimiento especializado del cliente y la generalización federada, aunque los valores son configurables para adaptarse a distintos escenarios. Este esquema aprovecha simultáneamente las relaciones específicas capturadas por los árboles locales y la capacidad de generalización del modelo federado. La combinación híbrida ofrece mayor robustez frente a la heterogeneidad de los datos distribuidos. La inferencia se ejecuta de manera **totalmente local**, lo que asegura independencia del servidor y elimina latencias de consulta, aspecto crítico en aplicaciones de tiempo real como sistemas de detección de intrusiones en red o teclados móviles. 

---

# **Rondas federadas**

En esta arquitectura, cada ronda federada se reinicia desde cero, sin heredar modelos previos. En bosques aleatorios y proactivos, los árboles se construyen íntegramente en cada iteración y no se actualizan mediante gradientes, por lo que múltiples rondas no aportan mejoras sustanciales de precisión. Por el contrario, incrementan los costos de comunicación y el riesgo de sesgo acumulativo. Reiniciar cada ronda asegura reproducibilidad, equidad entre clientes y eficiencia en entornos con recursos limitados. Esta decisión constituye un diseño explícito de la propuesta y responde a la necesidad de optimizar recursos y garantizar transparencia metodológica.

