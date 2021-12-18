# Data Mining - Google Play Store
##### Proyeto de minería de datos para curso CC5206

## Motivacion

Durante los últimos años debido a la tecnología se han desarrollados miles de aplicaciones para los dispositivos emergentes por lo que distintas empresas han fijado sus ojo en esta nueva oportunidad de mercado, por ejemplo, durante el periodo de pandemia del COVID-19, la app store de la empresa apple a tenido un crecimiento en sus descargas de un 7% registrando 218 billones de descargas totales. Con este nuevo comportamiento las personas están pasando aproximadamente 4 horas al día en aplicaciones dentro de sus dispositivos móviles, independientemente de su rango de edad, generando así un crecimiento en este mercado, pues el 26% de los dólares generados globalmente están ligados a negocios que ofrecen algún tipo de aplicación móvil. 

Si bien este mercado presenta una oportunidad de negocio tentativa, si es que se presenta el capital necesario, primero se debe realizar un estudio de mercado, pues aproximadamente entre un 80 a 90% de las aplicaciones recién lanzadas al mercado son abandonadas en su primera etapa debido a la mala percepción popular, es por esta razón que se deben explorar puntos claves en el estudio de mercado.

En base a lo anterior, es de interés estudiar las relaciones existentes entre las aplicaciones, tales como las aplicaciones más utilizadas y mejor calificadas, aplicaciones con mayor número de instalaciones, aplicaciones con mejores (o peores) comentarios, etc. Dichas relaciones, podrían dar indicios del porqué una ciertas aplicaciones son las más descargadas (importancia de tales aplicaciones en la vida de hoy, y que tan necesarias son para las personas), o revelar, en base a las aplicaciones más utilizadas, conductas inducidas en las personas por las mismas. 

Al interés de estudiar datos del mercado de aplicaciones se suma, la preponderancia que tienen los smartphones y sus aplicaciones, en la vida cotidiana de las personas. Con el pasar del tiempo, las empresas desarrolladoras de aplicaciones han logrado vincularse con distintas costumbres y culturas humanas, teniendo una inmensa variedad de funcionalidades accesibles por las aplicaciones, que influyen en la vida de personas de formas muy variadas. Tales hábitos revelan factores importantes en los contextos sociales, económicos, políticos y culturales de los distintos países del mundo. 
Con los puntos a estudiar ya fijos, se tomó la decisión de estudiar un conjunto de datos recolectados de la tienda de aplicaciones Google Play, disponible en Kaggle. El dataset consiste básicamente de 2 tablas, la primera presenta información sobre las aplicaciones, como atributos de categoría, clasificación, cantidad de reseñas y tamaño aproximado. La segunda tabla contiene datos de las 100 reseñas más representativas de un conjunto de aplicaciones, en esta se encuentra información sobre la reseña, el carácter positivo, negativo o neutral de la reseña, que tan objetiva o subjetiva es y la calidad de redacción de esta.

## Preguntas y problemas

Basado en lo explicado en las secciones de motivación y análisis exploratorio surgen las siguientes preguntas/problemas que se podrían responder usando la refactorización de los datos para poder entender de mejor manera el comportamiento de las descargas de aplicaciones. Las preguntas son las siguientes:

- **1:** ¿Es posible predecir la cantidad de descargas de una aplicación, en base a los datos disponibles en Google Play?¿Se puede hacer la misma preducción con los datos de las reviews?

- **2:** ¿Es posible caracterizar tipos de aplicaciones respecto a los reviews recibidos por los usuarios? 

- **3:** ¿Existen características específicas de las aplicaciones que permitan tener mejor o peor aprobación del público?¿Es posible predecir el rating?

## Propuesta Metodológica Experimental Inicial

### Preprocesamiento

El dataframe 'Apps' se subdivide según categoría, escogiendo sólo las 10 con más descargas. En cada categoría se eliminan los outliers del número de descargas (en base al gráfico de caja). Se remueven las filas con valores faltantes de "Size" y "Rating".

Para porder utilizar "Content.Rating" en todos los modelos, se reemplazan sus valores cualitativos "Everyone", "Everyone 10+", "Teen", "Mature 17+" y "Adults only 18+", por 0.0, 0.5, 1.0, 1.5 y 2.0 respectivamente. Se eliminan filas con "Unrated".

Se agregan columnas que indican a que decil y cuartil de descargas y rating respectivamente, según cetegoría, pertenece cada aplicación.

De 'Reviews' se remueven las aplicaciones que no están en 'Apps'.

Del dataframe 'Apps/Reviews' se selecionan las aplicaciones por sobre el primer cuartil de reviews consideradas en 'Reviews' (más de 28) para realizar clustering en base a los 'scores' y los conteos de: caracteres, palabras, oraciones, espacios, caracteres alfa-numéricos, puntuaciones y sintagma nominal (noun phrase). No se utilizarán los conteos de emojis, duplicados, fechas y números enteros debido a que en la mayoría de los casos son 0, tampoco el promedio de caracteres no alfa-numéricos ya que es complemento de el de caracteres alfa-numéricos.

Los promedios de espacios, caracteres alfa-numéricos y puntuaciones se transforman a un espacio relativo dividiendolos por el promedio de caracteres. Los promedios de caracteres, palabras, oraciones y sintagma nominal se normalizan al espacio [0,1].

Se agregan los cuartiles 1 y 3 de los "scores". También se eliminan outliers del número de descargas y se agrega una columna que indica a que cuartil (de descargas) pertenece cada aplicación.

### Clasificación

Las preguntas 1 y 3 son problemas de predicción, por lo tanto para estas recurriremos a modelos de clasificación. Dado que para distintos tipos de aplicaciones, tanto el público objetivo como las expectativas y exigencias que se tiene sobre las mismas pueden variar considerablemenente, a fin de lograr mayor homogeneidad, se subdivide el dataset por categorías y se aplica la clasificación por separado sobre los 10 sub-datasets con más elementos.

__Aplicaciones:__

Se seleccionan las columnas "Rating", "Reviews", "Size", "Type" y "Content.Rating" para clasificar las descagas según cuartiles (pregunta 1). Se clasifica el rating según cuartiles con los mismo atributos, reemplazando "Rating" por "Installs" (Pregunta 3). No se concidera el género debido a que pueden existir combinaciones categoría/género no contempladas en los datos.

Se repiten las clasificaciones de descargas y rating para apliciones de pago, incluyendo el precio en los atributos a evaluar (extrayendo outliers).

__Reseñas:__

La clasificación en base a resseñas se divide en dos partes. Primero se evaluará la posibilidad de extrar información del comentario de las reviews. Para ello se vectoriza el texto y luego se reduce la dimensionalidad por medio de _Latent semantic analysis_. Se utilizan los datos del texto para clasificar "sentiment_polarity", "sentiment_subjectivity" y "spelling_quality". En la vectorización del texto no se consideran _stop words_, ya que palabras comunmente categorizadas como tal podrían aportar información para el nivel de subjetividad o facilidad de lectura.

Para la segunda parte se evaluan los datos de 'Apps/Reviews' para clasificar los cuartiles y deciles de rating y descargas. Para este caso no se subdivide por categoría debido a la menor cantidad de aplicaciones que contiene el dataframe.

Para todas las clasificaciones mencionadas se utlizan los modelos de árbol de decisión, KNN, naive bayes y support vector machines. Para evaluar los resultados de la clasificación se hará uso de cross validation con distribución 80/20 entre datos de entrenamiento y prueba, y considerarán las métricas: exactitud, precisión, recall y F1.

### Clustering

Sobre los datos de 'Apps/Reviews' se aplican los siguentes métodos: _K_-means, con valores de _k_ obtenidos según optimos observables en un gráfico de codo; clustering jerárquico aglomerativo, con criterios _complete_, _average_, _single_ y _ward_; y DBSCAN con valores de _eps_ y _minPts_ dados por la distancia media a los _k_ vecinos y el doble de la dimensionalidad del dataset respectivamente, ambos valores se toman como base y se variarán según los resultados.

Se tendrán dos enfoques para evaluar los clusters.

La primera se centra en los propios datos usados para crear los clusters, es decir el de las reviews. Los resultados se evaluan en base a cohesión (SSE) y coeficiente de Silhouette. Para visualizar los resultados se reduce la dimensionalidad con el método _principal component analisys_, de modo que sea posible crear un gráfico de disperción. También se estudia la distribución de loss atributos en los clusters resultantes, en específico las medias y cuartiles.

Para este caso se podrían aplicar más filtros y dividir los datos en subconjuntos más homogéneos (según número de descargas, categorías, content rating, ect.) sin embargo, el dataset actual ya es demasiado pequeño (607 filas).

La segunda opción es evaluar la posibilidad de que, por medio de sus reviews, se logre caracterizar las aplicaciones según datos externos a esta, como lo son la categoría o el rating. En este caso se evaluan los resultados comprobando las diferencias y similitudes de las aplicaciones de los clusters resultantes, según sus atributos en el dataframe 'Apps'. Para ello se considera que fracción de categorías, géneros y content rating se encuentran en cada cluster, junto con el porcentanje de la clase mayoritaria en cada caso. Se considerará también la media de rating de cada cluster. Estos resultados también pueden ser útiles para responder la pregunta 3.

Para ambos casos se evalua también con medida de información mutua, comparando los clusters con distintos atributos del dataset.

Adicionalmente, se realiza clustering con los métodos ya mencionados sobre el dataset 'Apps'. Si se encuentran clusters consistentes, se repite la parte de clasificación sobre estos datos.

### Notebooks

Más detalles sobre las metodologías usadas y un segundo análisis y reducción de atributos se encuentra en los respectivos notebooks de cada pregunta, donde se desarrollan los experimentos correspondientes.

## Contribucion (Hito 3)

- Benjamín Farías: Notebooks de los experimentos, presentación.
- Diego León: Reformulación preguntas (mejora Hito 1), Propuestas experimentales, Aporte de presentación.
- Tomás Letelier: Continuacion de exploracion de datos, Scatterplots, Aporte de la presentación.
- Joaquin Moraga: Presentacion, Informe conclusiones, limitaciones y proyecciones a futuro

