---
title: "Primeros pasos con Google Antigravity"
source: "https://codelabs.developers.google.com/getting-started-google-antigravity?hl=es-419#8"
author:
published:
created: 2026-05-17
description: "En este codelab, se te guiará por el proceso de instalación y experiencia con las funciones de Google Antigravity, una plataforma de desarrollo centrada en los agentes."
tags:
  - "clippings"
---
*access\_time* Quedan 25 minutos

## Acerca de este codelab

*subject* Última actualización: abr 30, 2026

*account\_circle* Escrito por Romin Irani & Mete Atamel

## 1\. Introducción

En este codelab, aprenderás sobre [Google Antigravity](https://antigravity.google/?hl=es-419), una plataforma de desarrollo con agentes que evoluciona el IDE hacia la era centrada en los agentes.

A diferencia de los asistentes de programación estándar que solo completan automáticamente líneas, Antigravity proporciona un "Centro de control" para administrar agentes autónomos que pueden planificar, programar e incluso navegar por la Web para ayudarte a crear.

Antigravity se diseñó como una plataforma centrada en los agentes. Se presupone que la IA no es solo una herramienta para escribir código, sino un actor autónomo capaz de planificar, ejecutar, validar y realizar iteraciones en tareas de ingeniería complejas con una intervención humana mínima.

### Qué aprenderás

- Instalar y configurar Antigravity
- Explorar los conceptos clave de Antigravity, como el Administrador de agentes, el Editor, el navegador y mucho más
- Personalizar Antigravity con tus propias reglas y flujo de trabajo, junto con consideraciones de seguridad

### Requisitos

Actualmente, Antigravity está disponible como vista previa para las cuentas personales de Gmail. Incluye una cuota gratuita para usar modelos premium.

Antigravity debe estar instalado de forma local en tu sistema. El producto está disponible en Mac, Windows y distribuciones específicas de Linux. Además de tu propia máquina, necesitarás lo siguiente:

- Navegador web Chrome
- Una cuenta de Gmail (cuenta personal de Gmail)

Este codelab está diseñado para usuarios y desarrolladores de todos los niveles (incluidos los principiantes).

### Problemas con los informes

A medida que trabajes en el codelab y con Antigravity, es posible que encuentres problemas.

Si tienes problemas relacionados con el codelab (errores de escritura, instrucciones incorrectas), abre un error con el botón `Report a mistake` que se encuentra en la esquina inferior izquierda de este codelab:

![b06b582bcd847f6d.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/b06b582bcd847f6d_2880.png?hl=es-419)

Si tienes errores o solicitudes de funciones relacionadas con Antigravity, informa el problema dentro de la app. Puedes hacerlo en Agent Manager con el vínculo `Provide Feedback` que se encuentra en la esquina inferior izquierda:

![281ac826fb44d427.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/281ac826fb44d427_2880.png?hl=es-419)

También puedes ir al editor con el vínculo `Report Issue` que se encuentra debajo del ícono de tu perfil:

![e8afd782a8f92129.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e8afd782a8f92129_2880.png?hl=es-419)

## 2\. Instalación

Si aún no tienes instalado Antigravity, comencemos por instalarlo. Actualmente, el producto está disponible para la versión preliminar, y puedes usar tu cuenta personal de Gmail para comenzar a usarlo.

Ve a la página de [descargas](https://antigravity.google/download?hl=es-419) y haz clic en la versión del sistema operativo que corresponda a tu caso. Inicia el instalador de la aplicación y, luego, instálala en tu máquina. Una vez que completes la instalación, inicia la aplicación de Antigravity. Deberías ver una pantalla similar a la siguiente:

![29fada39721093c.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/29fada39721093c_2880.png?hl=es-419)

Haz clic en `Next` cada vez. A continuación, se detallan los pasos clave:

- **Choose setup flow:** Esta opción te permite importar la configuración existente de VS Code o Cursor. Comenzaremos de nuevo.
- **Elige un tipo de tema del editor**: Elegiremos el tema oscuro, pero la decisión es completamente tuya.
- **¿Cómo quieres usar el agente de Antigravity?**

![7ca55560ec377130.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/7ca55560ec377130_2880.png?hl=es-419)

Analicemos esto con un poco más de detalle. Recuerda que la configuración se puede cambiar en cualquier momento a través de Antigravity User Settings (Linux/Windows: `Ctrl + ,` Mac: `Cmd + ,`).

Antes de analizar las opciones, veamos algunas propiedades específicas (que se encuentran a la derecha del diálogo).

### Política de ejecución de la terminal

Se trata de darle al agente la capacidad de ejecutar comandos (aplicaciones o herramientas) en tu terminal:

- **Siempre continuar:** Siempre ejecuta automáticamente los comandos de la terminal (excepto los que se encuentran en una lista de bloqueo configurable).
- **Solicitar revisión:** Solicita la revisión y aprobación del usuario antes de ejecutar comandos de terminal.

### Política de opiniones

A medida que el agente realiza su tarea, crea varios artefactos (plan de tareas, plan de implementación, etc.). La política de revisión está configurada de manera que puedas determinar quién decide si se debe revisar. Si siempre quieres revisarlo o dejar que el agente decida sobre esto Por lo tanto, aquí también hay tres opciones.

- **Siempre continuar:** El agente nunca solicita una revisión.
- **Agent Decides:** El agente decidirá cuándo solicitar la revisión.
- **Solicitar revisión:** El agente siempre solicita una revisión.

### Política de ejecución de JavaScript

Cuando está habilitado, el agente puede usar herramientas del navegador para abrir URLs, leer páginas web y interactuar con el contenido del navegador. Esta política controla cómo se ejecuta JavaScript en el navegador.

- **Siempre continuar:** El agente no se detendrá para pedir permiso para ejecutar JavaScript en el navegador. Esto le brinda al agente la máxima autonomía para realizar acciones y validaciones complejas en el navegador, pero también tiene la mayor exposición a vulnerabilidades de seguridad.
- **Solicitar revisión:** El agente siempre se detendrá para pedir permiso para ejecutar código de JavaScript en el navegador.
- **Inhabilitado:** El agente nunca ejecutará código JavaScript en el navegador.

Ahora que comprendemos las diferentes políticas, las 4 opciones de la izquierda no son más que parámetros de configuración específicos para las políticas de ejecución de terminal, revisión y ejecución de JavaScript para 3 de ellas, y una 4ª opción disponible en la que podemos controlarlo de forma completamente personalizada. Estas 4 opciones están disponibles para que podamos elegir cuánta autonomía quieres darle al agente para ejecutar comandos en la terminal y obtener artefactos revisados antes de continuar con la tarea.

Estas 4 opciones son las siguientes:

- **Modo seguro**: El modo seguro proporciona controles de seguridad mejorados para el agente, lo que te permite restringir su acceso a recursos externos y operaciones sensibles. Cuando se habilita el modo seguro, se aplican varias medidas de seguridad para proteger tu entorno.
- **Desarrollo basado en revisiones (recomendado)**: El agente solicitará revisiones con frecuencia.
- **Desarrollo impulsado por el agente**: El agente nunca pedirá una revisión.
- **Configuración personalizada**

La opción **Desarrollo basado en la revisión** es un buen equilibrio y la más recomendada, ya que permite que el agente tome una decisión y vuelva a comunicarse con el usuario para obtener su aprobación.

A continuación, se muestra la página de configuración **Configura tu editor**, en la que puedes elegir tus preferencias para lo siguiente:

- Vinculaciones de teclas: Puedes configurar las vinculaciones de teclas.
- Extensiones: Puedes instalar extensiones populares de idiomas y otras extensiones recomendadas.
- Línea de comandos: Puedes instalar la herramienta de línea de comandos para abrir Antigravity con `agy`.

Ahora, puedes **acceder a Google.** Como se mencionó anteriormente, Antigravity está disponible en modo de vista previa y es gratis si tienes una cuenta personal de Gmail. **Accede ahora con tu cuenta.** Se abrirá el navegador para que puedas acceder. Si la autenticación se realiza correctamente, verás un mensaje similar al siguiente y se te redireccionará a la aplicación de Antigravity. Fluye con la corriente.

Por último, **Condiciones de Uso**. Puedes decidir si quieres habilitar la opción o no y, luego, hacer clic en `Next`.

Esto te llevará al momento de la verdad, en el que Antigravity estará esperando para colaborar contigo.

## 3\. Administrador de agentes

¡Ya estamos listos para comenzar!

Antigravity bifurca la base de código abierto de Visual Studio Code (VS Code), pero altera radicalmente la experiencia del usuario para priorizar la administración de agentes por sobre la edición de texto. La interfaz se bifurca en dos ventanas principales distintas: `Editor` y `Agent Manager`. Esta separación de responsabilidades refleja la distinción entre la contribución individual y la administración de ingeniería.

## Administrador de agentes: Centro de control

Cuando se inicia Antigravity, el usuario suele ver los mensajes **Open Folder, Open Agent Manager** y **Clone Repository.**

![2afba3ba858adf03.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/2afba3ba858adf03_2880.png?hl=es-419)

Este **Administrador de agentes** actúa como un panel de control de la misión. Está diseñado para la organización de alto nivel, lo que permite a los desarrolladores generar, supervisar e interactuar con varios agentes que operan de forma asíncrona en diferentes espacios de trabajo o tareas.

En esta vista, el desarrollador actúa como arquitecto. Definen objetivos de alto nivel, como los siguientes:

- Refactoriza el módulo de autenticación
- Actualiza el árbol de dependencias
- Genera un paquete de pruebas para la API de facturación

Cada una de estas solicitudes genera una instancia de agente dedicada. La IU proporciona una visualización de estos flujos de trabajo paralelos, en la que se muestra el estado de cada agente, los **artefactos** que produjo (planes, resultados, diferencias) y las solicitudes pendientes de aprobación humana.

Esta arquitectura aborda una limitación clave de los IDE anteriores, que ofrecían una experiencia más parecida a la de un chatbot, que eran lineales y síncronos. En una interfaz de chat tradicional, el desarrollador debe esperar a que la IA termine de generar código antes de hacer la siguiente pregunta. En la vista de administrador de Antigravity, un desarrollador puede enviar cinco agentes diferentes para que trabajen en cinco errores distintos de forma simultánea, lo que multiplica su rendimiento de manera efectiva.

Si haces clic en `Open Folder`, tendrás la opción de abrir un espacio de trabajo.

Piensa en el espacio de trabajo como lo conocías en VS Code y listo. Para abrir una carpeta local, haz clic en el botón y, luego, selecciona una carpeta para comenzar. En mi caso, tenía una carpeta en mi carpeta principal llamada `my-agy-projects`, y la seleccioné. Puedes usar una carpeta completamente diferente. Ten en cuenta que puedes omitir este paso por completo si lo deseas y abrir un lugar de trabajo en cualquier momento más adelante.

Una vez que completes este paso, estarás en la ventana Agent Manager, que se muestra a continuación:

![156224e223eeda36.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/156224e223eeda36_2880.png?hl=es-419)

Notarás que la aplicación está configurada de inmediato para iniciar una conversación nueva en la carpeta del espacio de trabajo (`my-agy-projects`) que se seleccionó. Puedes usar tus conocimientos existentes sobre el trabajo con otras aplicaciones de IA (Cursor, Gemini CLI) y usar `@` y otras formas de incluir contexto adicional durante la instrucción.

Echa un vistazo a los menús desplegables `Model Selection`. El menú desplegable Selección de modelo te permite elegir uno de los modelos disponibles en el momento para que lo use tu agente. La lista se muestra a continuación:

![ca0b386cb97d1661.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/ca0b386cb97d1661_2880.png?hl=es-419)

Del mismo modo, vemos que el agente estará en un modo `Fast` predeterminado. Pero también podemos usar el modo `Plan`.

![b2b81488960677c2.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/b2b81488960677c2_2880.png?hl=es-419)

Veamos qué dice la [documentación](https://antigravity.google/docs/agent-modes-settings?hl=es-419) sobre este tema:

- `Planning`: Un agente puede planificar antes de ejecutar tareas. Úsalo para investigaciones profundas, tareas complejas o trabajo colaborativo. En este modo, el agente organiza su trabajo en grupos de tareas, produce artefactos y toma otras medidas para investigar, analizar y planificar su trabajo de manera exhaustiva y lograr una calidad óptima. Aquí verás muchos más resultados.
- `Fast`: Un agente ejecutará tareas directamente. Úsalo para tareas simples que se pueden completar más rápido, como cambiar el nombre de variables, iniciar algunos comandos de Bash o realizar otras tareas más pequeñas y localizadas. Esto es útil cuando la velocidad es un factor importante y la tarea es lo suficientemente simple como para no preocuparse por una calidad inferior.

Si conoces el concepto de presupuesto de pensamiento y términos similares en los agentes, considera que esta es la capacidad de controlar el pensamiento del agente, lo que tiene un impacto directo en el presupuesto de pensamiento. Por el momento, usaremos los valores predeterminados, pero recuerda que, en el momento del lanzamiento, la disponibilidad del modelo Gemini 3 Pro se basa en cuotas limitadas para todos, por lo que debes esperar mensajes adecuados que indiquen si agotaste esas cuotas gratuitas para el uso de Gemini 3.

Ahora, dediquemos un poco de tiempo al Administrador de agentes (ventana) y comprendamos algunas cosas, de modo que queden claros los componentes básicos, cómo navegar en Antigravity y mucho más. A continuación, se muestra la ventana del Administrador de agentes:

![eaba0c6ee17369e2.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/eaba0c6ee17369e2_2880.png?hl=es-419)

Consulta el diagrama anterior con los números:

1. `Start Conversation`: Haz clic aquí para iniciar una conversación nueva. Esto te llevará directamente a la entrada en la que se muestra `Ask anything`.
2. `Workspaces`: Mencionamos los espacios de trabajo y que puedes trabajar en cualquier espacio de trabajo que desees. Puedes agregar más espacios de trabajo en cualquier momento y seleccionar cualquiera de ellos cuando inicies la conversación.
3. `Editor View`: Puedes cambiar a la vista del editor en cualquier momento. Se mostrará la carpeta del espacio de trabajo y los archivos generados. Puedes editar los archivos directamente allí o incluso proporcionar orientación intercalada, comandos en el editor, para que el agente pueda hacer algo o cambiar según tus recomendaciones o instrucciones modificadas. Veremos la vista del editor en detalle más adelante.

## 4\. Navegador Antigravity

Según la [documentación](https://antigravity.google/docs/browser-subagent?hl=es-419), cuando el agente quiere interactuar con el navegador, invoca un subagente del navegador para que se encargue de la tarea. El subagente del navegador ejecuta un modelo especializado para operar en las páginas que están abiertas en el navegador administrado por Antigravity, que es diferente del modelo que seleccionaste para el agente principal.

Este subagente tiene acceso a una variedad de herramientas necesarias para controlar tu navegador, como hacer clic, desplazarse, escribir, leer registros de la consola y mucho más. También puede leer tus páginas abiertas a través de la captura del DOM, capturas de pantalla o análisis de Markdown, así como grabar videos.

Esto significa que debemos iniciar e instalar la extensión para el navegador Antigravity. Para ello, iniciaremos una conversación y seguiremos los pasos.

Inicia una conversación nueva en un espacio de trabajo y asigna la siguiente tarea: `go to antigravity.google`

**Envía la tarea**. Verás al agente analizar la tarea y podrás inspeccionar el proceso de pensamiento. En algún momento, procederá correctamente y mencionará que debe configurar el agente del navegador, como se muestra a continuación. Haz clic en `Setup`.

![e7119f40e093afd2.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e7119f40e093afd2_2880.png?hl=es-419)

Se abrirá el navegador y se mostrará un mensaje para instalar la extensión, como se muestra a continuación:

![82fb87d7d75b4a6c.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/82fb87d7d75b4a6c_2880.png?hl=es-419)

Continúa y se te dirigirá a la extensión de Chrome que podrás instalar.

![f3468f0e5f3bb075.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/f3468f0e5f3bb075_2880.png?hl=es-419)

Una vez que instales la extensión correctamente, Antigravity Agent comenzará a trabajar y te indicará que espera que le permitas realizar su tarea. Deberías ver algo de actividad en la ventana del navegador que se abrió:

![7f0367e00ac36d5a.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/7f0367e00ac36d5a_2880.png?hl=es-419)

Vuelve a cambiar la vista del Administrador de agentes y deberías ver lo siguiente:

![b9d89e1ebefcfd76.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/b9d89e1ebefcfd76_2880.png?hl=es-419)

Esto es exactamente lo que esperábamos que sucediera, ya que le pedimos al agente que visitara el sitio web de `antigravity.google`. Otorga el permiso y verás que se navegó al sitio web de forma segura, como se muestra a continuación:

![77fcc38b5fb4ca7c.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/77fcc38b5fb4ca7c_2880.png?hl=es-419)

## 5\. Artefactos

Antigravity crea **artefactos** a medida que planifica y realiza su trabajo como una forma de comunicar su trabajo y obtener comentarios del usuario humano. Estos son archivos Markdown enriquecidos, diagramas de arquitectura, imágenes, grabaciones del navegador, diferencias de código, etcétera.

Los artefactos resuelven la **"brecha de confianza".** Cuando un agente afirma que **"corrigió el error"**, el desarrollador debía leer el código para verificarlo. En Antigravity, el agente produce un artefacto para demostrarlo.

Estos son los principales artefactos que produce Antigravity:

- `Task Lists`: Antes de escribir código, el agente genera un plan estructurado. Por lo general, no es necesario editar este plan, pero puedes revisarlo y, en algunos casos, agregar un comentario para cambiarlo, si es necesario.
- `Implementation Plan`: Se usa para diseñar cambios dentro de tu base de código para completar una tarea. Estos planes contienen detalles técnicos sobre qué revisiones son necesarias y están diseñados para que el usuario los revise, a menos que la política de revisión de artefactos esté configurada como "Siempre continuar".
- `Walkthrough`: Se crea una vez que el agente completó la implementación de la tarea, como un resumen de los cambios y cómo probarlos.
- `Code diffs`: Si bien técnicamente no es un artefacto, Antigravity también produce diferencias de código que puedes revisar y comentar.
- `Screenshots`: El agente captura el estado de la IU antes y después de un cambio.
- `Browser Recordings`: Para las interacciones dinámicas (p.ej., "Haz clic en el botón de acceso, espera el spinner y verifica que se cargue el panel"), el agente graba un video de su sesión. El desarrollador puede mirar este video para verificar que se cumpla el requisito funcional sin ejecutar la app por su cuenta.

Se generan artefactos que aparecen en las vistas del Administrador de agentes y del Editor.

En la vista del editor, en la esquina inferior derecha, puedes hacer clic en `Artifacts`:

![5deff47fe0a93aa1.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/5deff47fe0a93aa1_2880.png?hl=es-419)

En la vista del Administrador de agentes, en la parte superior derecha, junto a `Review changes`, deberías ver un botón para activar o desactivar los artefactos. Si está activado, puedes ver la lista de artefactos generados:

![5320f447471c43eb.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/5320f447471c43eb_2880.png?hl=es-419)

Deberías ver la vista Artifacts como se muestra a continuación. En nuestro caso, le indicamos al agente que visite la página `antigravity.google`, por lo que capturó la captura de pantalla y creó un video de la misma:

![19d9738bb3c7c0c9.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/19d9738bb3c7c0c9_2880.png?hl=es-419)

Puedes ver las diferencias de código en `Review Changes` en la vista del editor:

![e1d8fd6e7df4daf3.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e1d8fd6e7df4daf3_2880.png?hl=es-419)

Los desarrolladores pueden interactuar con estos artefactos y las diferencias de código con "comentarios al estilo de Documentos de Google". Puedes seleccionar una acción o tarea específica, proporcionar un comando de la forma en que te gustaría que se ejecutara y, luego, enviarlo al agente. Luego, el agente incorporará estos comentarios y realizará las iteraciones correspondientes. Considera usar Documentos de Google interactivos, en los que le brindes comentarios al autor y este los tenga en cuenta.

## 6\. Editor

El editor conserva la familiaridad de VS Code, lo que garantiza que se respete la memoria muscular de los desarrolladores experimentados. Incluye el explorador de archivos estándar, el resaltado de sintaxis y el ecosistema de extensiones.

Puedes hacer clic en el botón `Open Editor` en la esquina superior derecha del Administrador de agentes para ir al Editor.

## Configuración y extensiones

En una configuración típica, verías el editor, la terminal y el agente:

![7996408528de93e1.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/7996408528de93e1_2880.png?hl=es-419)

Si este no es el caso, puedes activar o desactivar los paneles de la terminal y del agente de la siguiente manera:

- Para activar o desactivar el panel de la terminal, usa el acceso directo `` Ctrl + ` ``.
- Para activar o desactivar el panel del agente, usa la combinación de teclas `Cmd + L`.

Además, Antigravity puede instalar algunas extensiones durante la configuración, pero, según el lenguaje de programación que uses, es probable que debas instalar más extensiones. Por ejemplo, para el desarrollo en Python, estas son las extensiones que podrías elegir instalar:

![bd33a79837b5a12a.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/bd33a79837b5a12a_2880.png?hl=es-419)

## Editor

### Autocompletar

Mientras escribes código en el editor, se activa un autocompletado inteligente que puedes aceptar presionando la tecla **Tab**:

![e90825ed7a009350.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e90825ed7a009350_2880.png?hl=es-419)

### Presiona Tab para importar

Recibirás la sugerencia **tab to import** para agregar las dependencias faltantes:

![bcab60794caa0aec.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/bcab60794caa0aec_2880.png?hl=es-419)

### Presiona Tab para saltar

Recibirás sugerencias de **tab para saltar** y llevar el cursor al siguiente lugar lógico del código:

![8610ae5217be7fe5.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/8610ae5217be7fe5_2880.png?hl=es-419)

### Comandos

Puedes activar comandos con `Cmd + I` en el editor o la terminal para obtener sugerencias intercaladas con lenguaje natural.

En el editor, puedes solicitar un método para calcular los números de Fibonacci y, luego, aceptarlo o rechazarlo:

![13a615e515cea100.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/13a615e515cea100_2880.png?hl=es-419)

En la terminal, puedes obtener sugerencias de comandos:

![5a75e560f998cedc.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/5a75e560f998cedc_2880.png?hl=es-419)

## Panel lateral del agente

En el editor, puedes activar o desactivar el panel lateral del agente de varias maneras.

### Apertura manual

Puedes activar o desactivar manualmente el panel del agente a la derecha con el atajo `Cmd + L`.

Puedes comenzar a hacer preguntas, usar `@` para incluir más contexto, como archivos, directorios o servidores de MCP, o usar `/` para hacer referencia a un flujo de trabajo (una instrucción guardada):

![95c5a6d31d771748.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/95c5a6d31d771748_2880.png?hl=es-419)

También puedes elegir entre dos modos de conversación: `Fast` o `Planning`:

![d3d1449f12510e3e.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/d3d1449f12510e3e_2880.png?hl=es-419)

`Fast` se recomienda para tareas rápidas, mientras que `Planning` se recomienda para tareas más complejas en las que el agente crea un plan que puedes aprobar.

También puedes elegir diferentes modelos para la conversación:

![af709bcc03c1e21e.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/af709bcc03c1e21e_2880.png?hl=es-419)

### Explicar y corregir

Otra forma de activar el agente es colocar el cursor sobre un problema y seleccionar `Explain and fix`:

![e45cbe02ed76b9c1.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e45cbe02ed76b9c1_2880.png?hl=es-419)

### Enviar problemas al agente

También puedes ir a la sección `Problems` y seleccionar `Send all to Agent` para que el agente intente solucionar esos problemas:

![e4992d14708005d0.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/e4992d14708005d0_2880.png?hl=es-419)

### Enviar el resultado de la terminal al agente

Incluso puedes seleccionar una parte del resultado de la terminal y enviarla al agente con `Cmd + L`:

![c40293bab474c9b1.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/c40293bab474c9b1_2880.png?hl=es-419)

## Cómo alternar entre el Editor y el Administrador de agentes

En cualquier momento, puedes alternar entre el modo de edición y el modo de administrador de agentes completo con el botón `Open Agent Manager` en la parte superior derecha cuando estés en el modo de edición, y volver a hacer clic en el botón `Open Editor` en la parte superior derecha cuando estés en el modo de administrador de agentes.

También puedes usar la combinación de teclas `Cmd + E` para alternar entre los dos modos.

## 7\. Enviar comentarios

La capacidad de Antigravity para recopilar sin esfuerzo tus comentarios en cada etapa de la experiencia es su principal característica. A medida que el agente trabaja en una tarea, crea diferentes artefactos en el proceso:

- Un plan de implementación y una lista de tareas (antes de la codificación)
- Diferencias de código (a medida que se genera el código)
- Una guía para verificar los resultados (después de la codificación)

Estos elementos son una forma en que Antigravity comunica sus planes y su progreso. Lo que es más importante, también son una forma de proporcionar comentarios al agente en comentarios al estilo de Documentos de Google. Esto es muy útil para guiar al agente de manera eficaz en la dirección que deseas.

Intentemos crear una aplicación de lista de tareas simple y veamos cómo podemos proporcionar comentarios a Antigravity en el proceso.

## Modo de planificación

Primero, debes asegurarte de que Antigravity esté en modo `Planning` (en lugar de modo `Fast`).Puedes seleccionar esta opción en el chat del panel lateral del agente. Esto garantiza que Antigravity cree un plan de implementación y una lista de tareas antes de comenzar a escribir el código. Luego, prueba con una instrucción como esta: `Create a todo list web app using Python`. Esto iniciará el agente para que comience a planificar y producir un plan de implementación.

## Plan de implementación

Un plan de implementación es un resumen de lo que Antigravity pretende hacer, la pila tecnológica que usará y una descripción general de los cambios propuestos.

```
Implementation Plan - Python Todo App
Goal
Create a simple, functional, and aesthetically pleasing Todo List web application using Python (Flask).

Tech Stack
Backend: Python with Flask
Frontend: HTML5, CSS3 (Vanilla), Jinja2 templates
...
```

Este es también el primer lugar en el que puedes proporcionar comentarios. En nuestro caso, el agente quiere usar Flask como el framework web de Python. Podemos agregar un comentario al plan de implementación para usar FastAPI en su lugar. Una vez que agregues el comentario, envíalo o pídele a Antigravity que `Proceed` con el plan de implementación actualizado.

## Lista de tareas

Después de que se actualiza el plan de implementación, Antigravity crea una lista de tareas. Esta es una lista concreta de los pasos que seguirá Antigravity para crear y verificar la app.

```
Task Plan
 Create requirements.txt
 Create directory structure (static/css, templates)
 Create static/css/style.css
 Create templates/index.html
 Create main.py with FastAPI setup and Database logic
 Verify application
```

Este es el segundo lugar donde puedes proporcionar comentarios.

Por ejemplo, en nuestro caso de uso, puedes agregar instrucciones de verificación más detalladas con el siguiente comentario: `Verify application by adding, editing, and deleting a todo item and taking a screenshot.`

## Cambios en el código

En este punto, Antigravity generará código en archivos nuevos. Puedes `Accept all` o `Reject all` estos cambios en el panel lateral del chat del agente sin entrar en detalles.

También puedes hacer clic en `Review changes` para ver los detalles de los cambios y agregar comentarios detallados sobre el código. Por ejemplo, podemos agregar el siguiente comentario en [`main.py`](http://main.py/): `Add basic comments to all methods`

Esta es una excelente manera de iterar el código con Antigravity.

## Explicación

Una vez que Antigravity termina de escribir el código, inicia el servidor y abre un navegador para verificar la app. Realizará algunas pruebas manuales, como agregar tareas, actualizar tareas, etcétera. Todo esto gracias a la extensión para el navegador de Antigravity. Al final, crea un archivo de recorrido para resumir lo que hizo para verificar la app, lo que incluye una captura de pantalla o un flujo de verificación con una grabación del navegador.

También puedes comentar la captura de pantalla o la grabación del navegador en el recorrido. Por ejemplo, podemos agregar un comentario `Change the blue theme to orange theme` y enviarlo. Después de enviar el comentario, Antigravity realiza los cambios, verifica los resultados y actualiza el recorrido.

## Deshacer los cambios

Por último, después de cada paso, si no te gusta el cambio, puedes deshacerlo desde el chat. Solo tienes que elegir el ↩️ `Undo changes up to this point` en el chat.

## 8\. Reglas y flujos de trabajo

Antigravity incluye algunas opciones de personalización: **Reglas** y **Flujos de trabajo**.

En el **modo de edición**, haz clic en `...` en la esquina superior derecha y elige `Customizations`. Verás `Rules` y `Workflows`:

![ff8babd8d8bcfa83.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/ff8babd8d8bcfa83_2880.png?hl=es-419)

Las **reglas** ayudan a guiar el comportamiento del agente. Son instrucciones que puedes proporcionar para asegurarte de que el agente las siga mientras genera código y pruebas. Por ejemplo, es posible que quieras que el agente siga un determinado estilo de código o que siempre documente los métodos. Puedes agregarlas como reglas, y el agente las tendrá en cuenta.

Los **flujos de trabajo** son instrucciones guardadas que puedes activar a pedido con `/` mientras interactúas con el agente. También guían el comportamiento del agente, pero el usuario los activa a pedido.

Una buena analogía es que las **reglas** son más como instrucciones del sistema, mientras que los flujos de trabajo son más como instrucciones guardadas que puedes elegir a pedido.

Tanto las **Reglas** como los **Flujos de trabajo** se pueden aplicar de forma global o por espacio de trabajo, y se pueden guardar en las siguientes ubicaciones:

- Regla global: `~/.gemini/GEMINI.md`
- Flujo de trabajo global: `~/.gemini/antigravity/global_workflows/<YOUR_WORKFLOW_NAME>.md`
- Reglas del espacio de trabajo: `your-workspace/.agents/rules/`
- Flujos de trabajo de Workspace: `your-workspace/.agents/workflows/`

Agreguemos algunas reglas y flujos de trabajo en el espacio de trabajo.

## Agregar una regla

Primero, agreguemos una regla de estilo de código. Ve a `Rules` y selecciona el botón `+Workspace`. Asigna un nombre, como `code-style-guide`, con las siguientes reglas de estilo de código:

```
* Make sure all the code is styled with PEP 8 style guide
* Make sure all the code is properly commented
```

En segundo lugar, agreguemos otra regla para asegurarnos de que el código se genere de forma modular con ejemplos en una regla `code-generation-guide`:

```
* The main method in main.py is the entry point to showcase functionality.
* Do not generate code in the main method. Instead generate distinct functionality in a new file (eg. feature_x.py)
* Then, generate example code to show the new functionality in a new method in main.py (eg. example_feature_x) and simply call that method from the main method.
```

Las dos reglas se guardan y están listas:

![bfd179dfef6b2355.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/bfd179dfef6b2355_2880.png?hl=es-419)

## Cómo agregar un flujo de trabajo

También definamos un flujo de trabajo para generar pruebas de unidades. Esto nos permitirá activar pruebas de unidades cuando estemos satisfechos con el código (en lugar de que el agente genere pruebas de unidades todo el tiempo).

Ve a `Workflows` y selecciona el botón `+Workspace`. Asigna un nombre, como `generate-unit-tests`, con lo siguiente:

```
* Generate unit tests for each file and each method
* Make sure the unit tests are named similar to files but with test_ prefix
```

El flujo de trabajo también está listo para usarse:

![d22059258592f0e1.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/d22059258592f0e1_2880.png?hl=es-419)

## Probar

Ahora veamos las reglas y los flujos de trabajo en acción. Crea un archivo `main.py` de estructura en tu espacio de trabajo:

```
def main():
    pass

if __name__ == "__main__":
    main()
```

Ahora, ve a la ventana de chat del agente y pregúntale lo siguiente: `Implement binary search and bubble sort.`

Después de uno o dos minutos, deberías obtener tres archivos en el espacio de trabajo: `main.py`, `bubble_sort.py` y `binary_search.py`. También notarás que se implementaron todas las reglas: el archivo principal no está desordenado y tiene el código de ejemplo, cada función se implementa en su propio archivo, todo el código está documentado y tiene un buen estilo:

```
from binary_search import binary_search, binary_search_recursive
from bubble_sort import bubble_sort, bubble_sort_descending

def example_binary_search():
    """
    Demonstrate binary search algorithm with various test cases.
    """
    ...

def example_bubble_sort():
    """
    Demonstrate bubble sort algorithm with various test cases.
    """
    ...

def main():
    """
    Main entry point to showcase functionality.
    """
    example_binary_search()
    example_bubble_sort()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

Ahora que estamos satisfechos con el código, veamos si podemos activar el flujo de trabajo de generación de pruebas de unidades.

Ve al chat y comienza a escribir `/generate`. Antigravity ya conoce nuestro flujo de trabajo:

![8a3efd9e3be7eb6f.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/8a3efd9e3be7eb6f_2880.png?hl=es-419)

Selecciona `generate-unit-tests` y, luego, ingresa. Después de unos segundos, aparecerán archivos nuevos en tu espacio de trabajo: `test_binary_search.py` y `test_bubble_sort.py` con varias pruebas ya implementadas.

![11febd7940ef8199.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/11febd7940ef8199_2880.png?hl=es-419)

¡Genial!

## 9\. Habilidades

Si bien los modelos subyacentes de Antigravity (como Gemini) son generalistas potentes, no conocen el contexto específico de tu proyecto ni los estándares de tu equipo. Cargar cada regla o herramienta en la ventana de contexto del agente genera una "sobrecarga de herramientas", mayores costos, latencia y confusión.

Las habilidades de Antigravity resuelven este problema a través de la divulgación progresiva. Una **habilidad** es un paquete especializado de conocimiento que permanece inactivo hasta que se necesita. Solo se carga en el contexto del agente cuando tu solicitud específica coincide con la descripción de la habilidad.

## Estructura y alcance

Las habilidades son paquetes basados en directorios. Puedes definirlos en dos permisos según tus necesidades:

- Alcance global (`~/.gemini/antigravity/skills/`): Disponible en todos tus proyectos (p.ej., "Formatea JSON", "Revisión general del código").
- Alcance del lugar de trabajo (`<workspace-root>/.agents/skills/`): Solo está disponible dentro de un proyecto específico (p.ej., "Implementar en la etapa de pruebas de esta app", "Generar código estándar para este framework específico").

## Anatomía de una habilidad

Un directorio de habilidades típico se ve de la siguiente manera:

```
my-skill/
├── SKILL.md    #(Required) metadata & instructions.
├── scripts/    # (Optional) Python or Bash scripts for execution.
├── references/ # (Optional) text, documentation, or templates.
└── assets/     # (Optional) Images or logos.
```

Ahora agreguemos algunas habilidades.

## Habilidad de revisión de código

Esta es una habilidad solo de instrucciones, es decir, solo necesitamos crear el archivo `SKILL.md`, que contendrá los metadatos y las instrucciones de las habilidades. Creemos una **habilidad global** que proporcione detalles al agente para revisar los cambios de código en busca de errores, problemas de estilo y prácticas recomendadas.

Primero, crea el directorio que contendrá esta habilidad global.

```
mkdir -p ~/.gemini/antigravity/skills/code-review
```

Crea un archivo `SKILL.md` en el directorio anterior con el contenido que se muestra a continuación:

```
---
name: code-review
description: Reviews code changes for bugs, style issues, and best practices. Use when reviewing PRs or checking code quality.
---

# Code Review Skill

When reviewing code, follow these steps:

## Review checklist

1. **Correctness**: Does the code do what it's supposed to?
2. **Edge cases**: Are error conditions handled?
3. **Style**: Does it follow project conventions?
4. **Performance**: Are there obvious inefficiencies?

## How to provide feedback

- Be specific about what needs to change
- Explain why, not just what
- Suggest alternatives when possible
```

Ten en cuenta que el archivo `SKILL.md` anterior contiene los metadatos (nombre y descripción) en la parte superior y, luego, las instrucciones. Cuando se carga, el agente solo lee los metadatos de las habilidades que configuraste y carga las instrucciones de la habilidad solo si es necesario.

### Probar

Crea un archivo llamado `demo_bad_code.py` con el siguiente contenido:

```
import time

def get_user_data(users, id):
   # Find user by ID
   for u in users:
       if u['id'] == id:
            return u
   return None

def process_payments(items):
   total = 0
   for i in items:
       # Calculate tax
       tax = i['price'] * 0.1
       total = total + i['price'] + tax
       time.sleep(0.1) # Simulate slow network call
  
   return total

def run_batch():
   users = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
   items = [{'price': 10}, {'price': 20}, {'price': 100}]
  
   u = get_user_data(users, 3)
   print("User found: " + u['name']) # Will crash if None
  
   print("Total: " + str(process_payments(items)))

if __name__ == "__main__":
   run_batch()
```

Pregúntale al agente: `review the @demo_bad_code.py file`. El agente debe identificar la habilidad `code-review`, cargar los detalles y, luego, realizar la acción según las instrucciones proporcionadas en el archivo `code-review/SKILL.md`.

A continuación, se muestra un ejemplo del resultado:

![d90a989f4555e2fc.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/d90a989f4555e2fc_2880.png?hl=es-419)

## La habilidad de plantilla de encabezado de código

A veces, una skill necesita usar un bloque grande de texto estático (como un encabezado de licencia). Es un desperdicio poner este texto directamente en la instrucción. En cambio, lo colocamos en una carpeta `resources/` y le indicamos al agente que lo lea solo cuando sea necesario.

Primero, crea el directorio que contendrá esta **habilidad de Workspace**.

```
mkdir -p .agents/skills/license-header-adder/resources
```

Crea `.agents/skills/license-header-adder/resources/HEADER.txt` con el texto de tu licencia:

```
/*
 * Copyright (c) 2026 YOUR_COMPANY_NAME LLC.
 * All rights reserved.
 * This code is proprietary and confidential.
 */
```

Crea un archivo `.agents/skills/license-header-adder/SKILL.md` con el siguiente contenido:

```
---
name: license-header-adder
description: Adds the standard corporate license header to new source files.
---

# License Header Adder

This skill ensures that all new source files have the correct copyright header.

## Instructions
1. **Read the Template**: Read the content of \`resources/HEADER.txt\`.
2. **Apply to File**: When creating a new file, prepend this exact content.
3. **Adapt Syntax**: 
   - For C-style languages (Java, TS), keep the \`/* */\` block.
   - For Python/Shell, convert to \`#\` comments.
```

### Probar

Pregúntale al agente lo siguiente: `Create a new Python script named data_processor.py that prints 'Hello World'.`

El agente leerá la plantilla, convertirá los comentarios de estilo C al estilo Python y los antepondrá automáticamente a tu nuevo archivo.

Al crear estas habilidades, convertiste de manera eficaz el modelo generalista de Gemini en un especialista para tu proyecto. Codificaste tus prácticas recomendadas, ya sea siguiendo los lineamientos de revisión de código o los encabezados de licencia. En lugar de pedirle repetidamente a la IA que "recuerde agregar la licencia" o que "corrija el formato de la confirmación", el agente ahora sabe instintivamente cómo trabajar en tu equipo.

## 10\. Seguridad y control

Darle acceso a un agente de IA a tu sistema de archivos, terminal y navegador es un arma de doble filo. Permite la generación, la depuración y la implementación de código autónomas, pero también abre vectores para la inyección de instrucciones y el robo de datos. Antigravity aborda este problema a través de varios parámetros de configuración de control de seguridad.

Ve a la configuración de Antigravity y consulta la configuración en **Agent**.

## Política de artefactos

La primera es la Política de Revisión de Artefactos. Especifica el comportamiento del agente cuando solicita una revisión de los artefactos.

<table><tbody><tr><td colspan="1" rowspan="1"><p><strong>Modo de política</strong></p></td><td colspan="1" rowspan="1"><p><strong>Descripción</strong></p></td></tr><tr><td colspan="1" rowspan="1"><p>Always Proceeds</p></td><td colspan="1" rowspan="1"><p>El agente nunca solicita una revisión. Esto maximiza la autonomía del agente, pero también tiene el mayor riesgo de que el agente opere sobre contenido no seguro o artefactos inyectados.</p></td></tr><tr><td colspan="1" rowspan="1"><p>El agente decide</p></td><td colspan="1" rowspan="1"><p>El agente decidirá cuándo solicitar la revisión según la complejidad de la tarea y la preferencia del usuario.</p></td></tr><tr><td colspan="1" rowspan="1"><p>Solicita una revisión</p></td><td colspan="1" rowspan="1"><p>El agente siempre solicita una revisión</p></td></tr></tbody></table>

Es recomendable establecer esta política en `Asks for Review`.

## Política y zona de pruebas de la terminal

La siguiente es la política sobre los comandos de la terminal.

Existe una política de `Terminal Command Auto Execution`. Este parámetro de configuración determina la autonomía del agente en relación con los comandos de shell:

<table><tbody><tr><td colspan="1" rowspan="1"><p><strong>Modo de política</strong></p></td><td colspan="1" rowspan="1"><p><strong>Descripción</strong></p></td></tr><tr><td colspan="1" rowspan="1"><p>Solicitar revisión</p></td><td colspan="1" rowspan="1"><p>El agente siempre solicita confirmación antes de ejecutar comandos de terminal (excepto los que se encuentran en la lista de entidades permitidas).</p></td></tr><tr><td colspan="1" rowspan="1"><p>Siempre continuar</p></td><td colspan="1" rowspan="1"><p>El agente nunca solicita confirmación antes de ejecutar comandos de la terminal (excepto los que se encuentran en la lista de bloqueo). Esto le brinda al agente la máxima capacidad para operar durante períodos prolongados sin intervención, pero también tiene el mayor riesgo de que ejecute un comando de terminal no seguro.</p></td></tr></tbody></table>

Al principio, debes elegir `Request Review` y, luego, permitir gradualmente que el agente ejecute más comandos de terminal por espacio de trabajo.

También hay una sección `Enable Terminal Sandbox` que debes activar. Cuando está habilitada, los comandos de la terminal se ejecutan con restricciones de zona de pruebas.

## Política de Acceso a Archivos

De forma predeterminada, el agente solo puede acceder a los archivos de tu espacio de trabajo. Esta es una buena idea y limita los archivos a los que puede acceder el agente. Si alguna vez necesitas más, hay un botón de activación `Agent Non-Workspace File Access`. Cuando se habilita, esto permite que el agente vea y edite archivos fuera del espacio de trabajo actual de forma automática.

## Permisos de agente

Antigravity usa un sistema de permisos unificado para controlar qué acciones puede realizar el agente en tu nombre. Cada acción se representa como un recurso de permiso que se puede colocar en una de las tres listas siguientes:

- **Permitir**: La acción se aprueba automáticamente sin solicitar confirmación.
- **Rechazar**: La acción se bloquea de inmediato.
- **Preguntar**: El agente se detiene y te pide aprobación antes de continuar.

Cada entrada de las listas Permitir, Denegar o Preguntar sigue el formato `action(target)`.

Aquí, `action` es uno de los tipos de acción admitidos y `target` es un patrón que describe lo que abarca el permiso.

<table><tbody><tr><td colspan="1" rowspan="1"><p><strong>Acción</strong></p></td><td colspan="1" rowspan="1"><p><strong>Formato de destino</strong></p></td><td colspan="1" rowspan="1"><p><strong>Coincidencias</strong></p></td></tr><tr><td colspan="1" rowspan="1"><p><code>command</code></p></td><td colspan="1" rowspan="1"><p><code>command(prefix) or command(*)</code></p></td><td colspan="1" rowspan="1"><p>Coincide con los comandos por prefijo. command(git) coincide con <code>git add</code>, <code>git commi</code> t, etcétera.</p></td></tr><tr><td colspan="1" rowspan="1"><p><code>read_file</code></p></td><td colspan="1" rowspan="1"><p><code>read_file(/path)</code></p></td><td colspan="1" rowspan="1"><p>Coincide con el archivo o con todo lo que se encuentra en el directorio. Las rutas deben ser literales y absolutas. No se admiten los globs (<code>*.go</code>), las expresiones regulares ni <code>~</code>.</p></td></tr><tr><td colspan="1" rowspan="1"><p><code>write_file</code></p></td><td colspan="1" rowspan="1"><p><code>write_file(/path)</code></p></td><td colspan="1" rowspan="1"><p>El mismo que <code>read_file</code>. También abarca de forma implícita <code>read_file</code> para la misma ruta.</p></td></tr><tr><td colspan="1" rowspan="1"><p><code>read_url</code></p></td><td colspan="1" rowspan="1"><p><code>read_url(domain) or read_url(*)</code></p></td><td colspan="1" rowspan="1"><p>Coincide con el dominio y todos los subdominios. No coincide con las rutas de acceso de URL.</p></td></tr><tr><td colspan="1" rowspan="1"><p><code>mcp</code></p></td><td colspan="1" rowspan="1"><p><code>mcp(server/tool), mcp(server/<br>), or mcp()</code></p></td><td colspan="1" rowspan="1"><p>Coincide con el nombre exacto del servidor. <code>server/*</code> abarca todas las herramientas de ese servidor.</p></td></tr></tbody></table>

Puedes obtener más información [en la documentación](https://antigravity.google/docs/agent-permissions?hl=es-419).

### Prueba la lista de entidades permitidas

La **Lista de Permisos** se usa principalmente con la política de **Solicitud de Revisión**. Representa un modelo de seguridad positivo, lo que significa que todo está prohibido, a menos que se permita de forma expresa. Esta es la configuración más segura.

Ve a la sección `Permissions` y, luego, a `Always Allow`, y agrega lo siguiente: `command(ls)`

Luego, hazle la pregunta al agente y observa su comportamiento:

- Pregúntale al agente: `List the files in this directory`.
- El agente ejecuta `ls` automáticamente.
- Pregúntale al agente: `Delete the <some file>`.
- El agente intentará `rm <filepath>`, pero Antigravity forzará una revisión del usuario porque `rm` no está en la lista de entidades permitidas. Antigravity debería pedirte permiso antes de ejecutar el comando.

### Lista de entidades denegadas para pruebas

La **lista de bloqueo** es la protección de la política **Siempre continuar**. Representa un modelo de seguridad negativo, lo que significa que todo está permitido, a menos que se prohíba expresamente. Esto depende de que el desarrollador anticipe todos los peligros posibles, lo que es una propuesta arriesgada, pero que ofrece la máxima velocidad.

Ve a la sección `Permissions` y, luego, a la sección `Always Deny`, y agrega lo siguiente:

- `command(rm)`

Luego, hazle la pregunta al agente y observa su comportamiento:

- Pregúntale al agente: `Create a` [`main.py`](http://main.py/).
- El agente crea el archivo sin problemas.
- Pregúntale al agente: `Delete the` [`main.py`](http://main.py/) `file`.
- Antigravity bloquea la ejecución y te solicita aprobación manual.

## Política del navegador

La capacidad de Antigravity para navegar por la Web es un superpoder, pero también una vulnerabilidad. Un agente que visite un sitio de documentación vulnerado podría sufrir un ataque de inyección de instrucciones. Para evitarlo, puedes implementar una **lista de entidades permitidas de URLs del navegador** para el agente del navegador.

Para ver la configuración actual, ve a `Antigravity — Settings` y, luego, a `Browser`. Deberías ver la sección `Browser URL Allowlist` en la que puedes agregar URLs adicionales:

![7bd38b2b02df521d.png](https://codelabs.developers.google.com/static/getting-started-google-antigravity/img/7bd38b2b02df521d_2880.png?hl=es-419)

## 11\. Conclusión

¡Felicitaciones! Ya instalaste Antigravity correctamente, configuraste tu entorno y aprendiste a controlar tus agentes.

**Próximos pasos** Para ver Antigravity en acción en la compilación de aplicaciones del mundo real, puedes consultar los siguientes codelabs:

- [Compila con Google Antigravity](https://codelabs.developers.google.com/building-with-google-antigravity?hl=es-419): En este codelab, se muestra cómo compilar varias aplicaciones, incluido un sitio web dinámico de conferencias y una app de productividad.
- [Compila e implementa en Google Cloud con Antigravity](https://codelabs.developers.google.com/build-and-deploy-gcp-with-antigravity?hl=es-419): En este codelab, se muestra cómo diseñar, compilar e implementar una aplicación sin servidores en Google Cloud.

## Documentos de referencia

- Sitio oficial: [https://antigravity.google/](https://antigravity.google/?hl=es-419)
- Documentación: [https://antigravity.google/docs](https://antigravity.google/docs?hl=es-419)
- Casos de uso: [https://antigravity.google/use-cases](https://antigravity.google/use-cases?hl=es-419)
- Descargar: [https://antigravity.google/download](https://antigravity.google/download?hl=es-419)
- Canal de YouTube de Google Antigravity: [https://www.youtube.com/@googleantigravity](https://www.youtube.com/@googleantigravity?hl=es-419)