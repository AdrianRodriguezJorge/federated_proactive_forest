## **Fase 1 — Entrenamiento local en los clientes**

Cada cliente entrena su modelo de manera independiente utilizando el algoritmo **Proactive Forest (PF)**, que introduce un mecanismo de actualización de probabilidades para la selección de atributos y un componente de meta‑aprendizaje que regula la diversidad y el tamaño del bosque. A diferencia de un Random Forest clásico, PF no se limita a la aleatoriedad, sino que ajusta dinámicamente las probabilidades de los atributos en función de su desempeño previo, lo que garantiza que los árboles generados sean diversos y compactos.

El entrenamiento se organiza en **ventanas de tamaño fijo (W)**, típicamente de cinco árboles, aunque este valor puede ajustarse como hiperparámetro. Cada ventana constituye una unidad de construcción: el cliente genera los (W) árboles, calcula métricas de desempeño (Accuracy, Macro‑F1) y diversidad interna (PCD u otra medida), y prepara un paquete de información para el servidor. Este esquema por ventanas permite un equilibrio entre eficiencia de comunicación y calidad del modelo, ya que reduce la frecuencia de transmisión respecto a métodos nodo‑a‑nodo, pero mantiene suficiente granularidad para evaluar la diversidad y eficacia de los árboles.

---

## **Fase 2 — Comunicación de ventanas al servidor**

Una vez completada cada ventana, el cliente transmite al servidor:

- La **estructura completa de los (W) árboles** (nodos, umbrales, etiquetas).
- Las métricas de desempeño y diversidad asociadas.

La transmisión se realiza en formato eficiente (ej. **Protobuf**), lo que asegura compacidad, interoperabilidad y seguridad en el canal de comunicación. Este diseño evita la necesidad de compresión adicional en los clientes y reduce su carga computacional, aspecto crítico en dispositivos con recursos limitados.

---

## **Fase 3 — Agregación global con Round Robin**

El servidor recibe las ventanas de todos los clientes y construye el bosque federado mediante un proceso **secuencial e incremental**. La selección de cada árbol depende del **estado actualizado** del bosque global; por tanto, los scores y rankings se calculan **dinámicamente** para cada cliente en el momento en que le corresponde su turno. El procedimiento consta de tres pasos operativos, seguidos por la iteración entre clientes:

#### 1. Evaluación de score por árbol

Cuando es el turno del cliente ***k***, el servidor toma los ***W*** árboles de su ventana y calcula para cada árbol ***T*** el score combinado: 
$$\text{Score}(T)=\alpha\cdot F1(T)+(1-\alpha)\cdot \text{Diversidad}(T\mid G)$$ 
- ***F1(T)***: Macro‑F1 del árbol ***T*** estimada sobre la validación local reportada por el cliente.
- ***Diversidad(T | G)***: medida de disimilitud de ***T*** respecto al bosque global actual ***G***, se emplea **PCD** (Partition‑Coverage Distance) por defecto.
- **G** es el conjunto de árboles ya incorporados al bosque global en la ronda en curso.
- **Primera ronda**: ***G*** está vacío, la diversidad no es informativa; por tanto se fija 𝛼 = 1 para que el ordenamiento inicial del primer cliente dependa exclusivamente de la eficacia.
#### 2. Selección del mejor árbol del cliente en turno

Para el cliente en turno, los árboles se ordenan según el score calculado en el paso anterior y se **elige el árbol con mayor score** dentro de esa ventana. Ese árbol constituye la contribución candidata del cliente en la ronda actual.
#### 3. Incorporación inmediata y actualización del bosque global

El árbol seleccionado se incorpora **inmediatamente** al bosque global. Tras la incorporación, el bosque global actualizado se utiliza como referencia para evaluar al siguiente cliente. De este modo, cada incorporación afecta la evaluación de los clientes posteriores en la misma ronda.
#### Iteración por los ***k*** clientes mediante Round Robin

- En cada ronda, el servidor genera una **permutación aleatoria** de los k clientes y procesa a los clientes en ese orden para mitigar sesgos por posición.
- Para cada cliente de la permutación se ejecutan los pasos 1–3 en secuencia; **no** se precalculan ni se reutilizan scores o rankings calculados antes de su turno.
- Al finalizar el recorrido por los ***k*** clientes, la ronda ha añadido exactamente ***k*** árboles al bosque global, uno por cliente.
---
## **Fase 4 — Criterio de parada (Progressive global)**

La construcción del bosque federado no tiene un tamaño fijo predeterminado. Se aplica el algoritmo **Progressive Forest** a nivel de servidor para decidir cuándo detener el entrenamiento:

- Tras cada ronda de incorporación, se evalúa la eficacia del bosque global acumulado.
- Si la mejora entre episodios consecutivos es menor o igual a un umbral de convergencia (ej. 0.002) durante dos periodos seguidos, se detiene el entrenamiento.
- Además, se define un **hiperparámetro de máximo de rondas** (ej. 20), que actúa como límite superior para evitar crecimiento indefinido.
    - Con 5 clientes y ventanas de 5 árboles, este límite representaría un máximo de 100 árboles en el bosque global.
    - Este valor ofrece un equilibrio razonable entre compacidad y capacidad predictiva, aunque puede ajustarse según el escenario.

---

## **Fase 5 — Actualización del modelo en los clientes**

Tras la agregación global, el servidor distribuye las ventanas seleccionadas a los clientes. Cada cliente:

- Incorpora los árboles globales a su bosque local.
- Excluye únicamente los árboles propios que ya fueron seleccionados, para evitar duplicaciones y sesgos en la votación.
- El resultado es un bosque local que refleja tanto la **generalización global** como la **especialización local**, manteniendo un balance entre conocimiento compartido y particularidades de los datos propios.

---

## **Fase 6 — Inferencia**

La inferencia se realiza de manera **local**, sin necesidad de comunicación con el servidor, lo que asegura baja latencia y robustez en aplicaciones de tiempo real. Se emplea un esquema de **votación híbrida ponderada**:  
$$ \hat{y} = \arg\max_c \left( \lambda \cdot p_{local}(c|x) + (1-\lambda) \cdot p_{global}(c|x) \right) $$

- Por defecto, $$(\lambda = 0.5)$$
- Este parámetro puede ajustarse para dar más peso al conocimiento local o al global según el grado de heterogeneidad de los datos.
- La combinación híbrida aprovecha simultáneamente las relaciones específicas capturadas por los árboles locales y la capacidad de generalización del modelo federado.

---

## **Rondas federadas**

El entrenamiento se organiza en **rondas sucesivas de ventanas**. Cada ronda incorpora nuevos árboles al bosque global mediante Round Robin. El criterio de parada se controla con Progressive Forest y el límite máximo de rondas. Este diseño asegura reproducibilidad, equidad entre clientes y eficiencia en entornos con recursos limitados.

