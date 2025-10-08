# Proyecto EDA - Análisis de Datos de Energía Solar en Colombia

## Descripción del Proyecto

Este proyecto realiza un análisis exploratorio de datos (EDA) de información satelital sobre energía solar en Colombia, utilizando datos del **Global Solar Atlas 2.0** desarrollado por **Solargis** y el **Banco Mundial (ESMAP)**.

## Fuente de Datos

Los datos provienen del Global Solar Atlas 2.0 con información satelital procesada entre **1994 y 2018**. Cada capa está en formato **GeoTIFF** y corresponde a un promedio de largo plazo, con resolución espacial de entre **9 y 120 arc-segundos**.

### Variables Disponibles

- **DIF.tif** → Irradiación horizontal difusa (kWh/m²)
- **DNI.tif** → Irradiación normal directa (kWh/m²)
- **GHI.tif** → Irradiación global horizontal (kWh/m²)
- **GTI.tif** → Irradiación global en el ángulo óptimo de inclinación (kWh/m²)
- **OPTA.tif** → Ángulo de inclinación óptimo para módulos FV (grados)
- **PVOUT.tif** → Producción fotovoltaica potencial (kWh/kWp)
- **TEMP.tif** → Temperatura media del aire (°C)

### Propósito de las Variables

- **DIF, DNI y GHI** → Caracterizan la disponibilidad de radiación solar en distintas condiciones
- **GTI y OPTA** → Permiten optimizar la captación de energía al definir la inclinación de los paneles
- **PVOUT** → Estima la producción energética de un sistema FV instalado
- **TEMP** → Describe el entorno térmico que influye en la eficiencia de los módulos solares

## Estructura del Proyecto

```
├── Scripts/
│   ├── Revicion_Inicial.ipynb     # Notebook de revisión inicial de datos
│   └── ProyectoEDA.ipynb          # Notebook con análisis exploratorio completo
├── data/
│   └── resultados_municipios.csv  # Datos agregados por municipio
└── README.md
```

## Análisis Realizados

### 1. Análisis Descriptivo
- Estadísticas descriptivas de todas las variables por municipio
- Distribución de radiación solar (histogramas)
- Análisis de outliers

### 2. Análisis de Correlaciones
- Matriz de correlación entre variables
- Identificación de relaciones entre variables solares y temperatura

### 3. Análisis Geográfico
- Distribución espacial de variables por departamento
- Identificación de regiones con mayor potencial solar

### 4. Análisis de Clustering
- Agrupación de municipios según características solares
- Identificación de patrones regionales

### 5. Análisis de Componentes Principales (PCA)
- Reducción de dimensionalidad
- Identificación de componentes principales

## Tecnologías Utilizadas

- **Python 3.10.18**
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Matplotlib/Seaborn** - Visualización de datos
- **Scikit-learn** - Machine Learning (PCA, Clustering)
- **Rasterio** - Procesamiento de datos raster GeoTIFF
- **GeoPandas** - Análisis geoespacial
- **Apache Airflow** - Orquestación de pipelines


## Instalación y Uso

### Prerrequisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn rasterio geopandas
```
```bash
# Ejecutar notebook de revisión inicial
jupyter notebook user_input_files/Revicion_Inicial.ipynb

# Ejecutar análisis EDA completo
jupyter notebook user_input_files/ProyectoEDA.ipynb
```

## Resultados

El análisis genera los siguientes outputs:

- **Estadísticas descriptivas** por municipio y variable
- **Visualizaciones** de distribuciones y correlaciones
- **Mapas de calor** de variables solares
- **Clusters** de municipios con características similares

## Licencia

Los datos están bajo **Creative Commons CC BY 4.0**, con origen en **Solargis (2021)** y publicados en la plataforma del Banco Mundial.

## Estado del Proyecto

**Fecha actual:** 8 de octubre de 2025

### ✅ Fases Completadas

- **Fase 1: Revisión Inicial** - Exploración preliminar de datos GeoTIFF
- **Fase 2: EDA Completo** - Análisis exploratorio detallado con pipeline automatizado
- **Fase 3: Pipeline** - Implementación de flujos para EDA

### 🚧 Estado Actual

Actualmente en **Fase de Consolidación EDA** con los siguientes logros:

- ✅ Pipeline de EDA automatizado funcionando
- ✅ Análisis estadístico completo de 1,118 municipios
- ✅ Identificación de patrones geográficos de radiación solar
- ✅ Clustering de municipios por características solares
- ✅ Análisis de componentes principales (PCA)
- ✅ Sistema de visualizaciones automatizado

## 🎯 Próximos Pasos

### Fase 4: Desarrollo de Modelo Predictivo (Planificado)

**Objetivo:** Desarrollar modelos de machine learning para predicción de potencial solar

#### 4.1 Preparación de Datos para Modelado
- [ ] Feature engineering de variables temporales
- [ ] Integración de datos meteorológicos adicionales
- [ ] Normalización y escalado de features

#### 4.2 Desarrollo de Modelos
- [ ] **Modelos de Regresión**
  - Regresión lineal múltiple
  - Random Forest Regressor
  - Gradient Boosting (XGBoost, LightGBM)
  - Support Vector Regression

- [ ] **Modelos de Clasificación**
  - Clasificación de zonas de alto/medio/bajo potencial
  - Identificación de ubicaciones óptimas para instalaciones

#### 4.3 Validación y Optimización
- [ ] Validación cruzada temporal
- [ ] Hyperparameter tuning
- [ ] Evaluación de métricas (RMSE, MAE, R²)
- [ ] Análisis de residuos y sesgo

#### 4.4 Implementación y Despliegue
- [ ] API para predicciones en tiempo real
- [ ] Dashboard interactivo para visualización
- [ ] Sistema de alertas para oportunidades
- [ ] Documentación técnica completa

### Fase 5: Validación con Datos Reales (Futuro)
- [ ] Comparación con instalaciones existentes
- [ ] Validación con estaciones meteorológicas
- [ ] Refinamiento de modelos basado en feedback

## 📊 Métricas del Proyecto

### Datos Procesados
- **Municipios analizados:** 1,118
- **Departamentos cubiertos:** 32
- **Variables solares:** 7 principales (DIF, DNI, GHI, GTI, OPTA, PVOUT, TEMP)
- **Período de datos:** 1994-2018 (24 años)

### Outputs Generados
- **Visualizaciones:** 12+ gráficos y mapas de calor
- **Modelos de clustering:** 4 clusters identificados
- **Componentes PCA:** 95% varianza explicada en 5 componentes

## 🛠️ Tecnologías Planificadas para Modelo Predictivo

- **Scikit-learn** - Modelos tradicionales de ML
- **XGBoost/LightGBM** - Gradient boosting avanzado
- **TensorFlow/PyTorch** - Deep learning (LSTM, Neural Networks)
- **Prophet** - Forecasting de series temporales
- **Optuna** - Optimización de hiperparámetros
- **MLflow** - Tracking de experimentos
- **FastAPI** - API para serving de modelos
- **Streamlit/Dash** - Dashboard interactivo

## 📈 Impacto Esperado

### Beneficios del Modelo Predictivo

1. **Planificación Energética**
   - Identificación óptima de ubicaciones para plantas solares
   - Estimación precisa de ROI en proyectos solares
   - Optimización de capacidad instalada

2. **Toma de Decisiones**
   - Soporte para políticas públicas energéticas
   - Evaluación de viabilidad de proyectos
   - Priorización de inversiones en infraestructura

3. **Reducción de Riesgos**
   - Predicción de variabilidad estacional
   - Identificación de factores críticos
   - Análisis de sensibilidad climática

## 🔄 Metodología de Desarrollo

### Enfoque Iterativo
1. **Desarrollo incremental** de modelos
2. **Validación continua** con métricas definidas
3. **Feedback loops** para mejora constante
4. **Documentación en tiempo real**

### Control de Calidad
- **Code reviews** para todos los modelos
- **Testing automatizado** de pipelines
- **Versionado** de modelos y datos
- **Monitoreo** de performance en producción

## 📋 Cronograma Estimado

- **Octubre 20 2025:** Finalización EDA y preparación datos
- **Octubre 30 2025:** Desarrollo modelos base
- **Noviembre 15 2025:** Optimización y validación
- **Noviembre 15 2026:** Producto Minimo

 ## Airflow
 | Orden | Task ID                               | Descripción                                                                                                                 | Input                                   | Output                                         | Dependencias                   |
| :---: | :------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------- | :--------------------------------------------- | :----------------------------- |
|   1   | `get_raw_data`                        | Carga los archivos base (raster de potencial solar, shapefile de municipios).                                               | Archivos `.tif`, `.shp` en `/data/raw/` | Datos crudos en memoria o carpeta temporal     | —                              |
|   2   | `preprocess_raster`                   | Procesa y estandariza los archivos raster (reproyección, recorte, máscara).                                                 | Raster original                         | Raster limpio y alineado en `/data/processed/` | get_raw_data                   |
|   3   | `merge_with_municipalities`           | Une los valores del raster con las áreas de los municipios.                                                                 | Raster limpio + shapefile               | GeoDataFrame con valores por municipio         | preprocess_raster              |
|   4   | `aggregate_data`                      | Calcula estadísticas agregadas (media, suma, desviación) del potencial solar por municipio.                                 | GeoDataFrame con valores por pixel      | DataFrame tabular por municipio                | merge_with_municipalities      |
|   5   | `eda_analysis`                        | Análisis exploratorio: distribución, correlaciones, mapas y gráficas.                                                       | DataFrame final por municipio           | Visualizaciones y conclusiones EDA             | aggregate_data                 |
|   6   | `save_clean_dataset`                  | Guarda el dataset final limpio y consolidado para análisis posteriores.                                                     | DataFrame tabular                       | Archivo `.csv` o `.parquet` en `/data/final/`  | aggregate_data                 |
|   7   | `model_training` *(siguiente paso)*   | Entrena modelo predictivo (p. ej. regresión o ML) para estimar potencial en municipios sin datos o bajo escenarios futuros. | Dataset limpio                          | Modelo entrenado (`.pkl`, `.joblib`)           | save_clean_dataset             |
|   8   | `model_validation` *(siguiente paso)* | Evalúa el modelo: métricas (MAE, RMSE, R²), validación cruzada y ajuste.                                                    | Modelo entrenado + dataset de prueba    | Reporte de desempeño del modelo                | model_training                 |
|   9   | `generate_report` *(Presentacion final)*  | Crea un informe automatizado (PDF/HTML) con resultados EDA, modelo y validación.                                            | Resultados EDA + métricas de validación | Reporte final en `/reports/`                   | eda_analysis, model_validation |



# Análisis y Conclusiones
## Conclusión Analítica del Análisis Exploratorio de Datos (EDA)

El análisis exploratorio de datos (EDA) revela información crucial sobre el **potencial de energía solar en Colombia**, identificando distribuciones, correlaciones, y patrones geográficos que son fundamentales para la planificación energética y la toma de decisiones.

---

### Distribución y Características de las Variables

Los histogramas de la **Distribución de Variables de Energía Solar** muestran lo siguiente:

* **Irradiación Difusa (DIF)**: Presenta una distribución relativamente simétrica y estrecha alrededor de su media ($\approx 901.5$), indicando una **baja variabilidad** (Std: $37.9$).
* **Irradiación Normal Directa (DNI)**: Muestra una distribución más dispersa, con una media de $\approx 1235.9$ y una **variabilidad significativamente alta** (Std: $267.1$). Esto sugiere que el DNI es la variable solar con mayor dispersión y, por lo tanto, un factor de mayor riesgo o potencial, dependiendo de la región.
* **Producción Fotovoltaica (PVOUT)**: La producción estimada ($\text{Media} \approx 1445.2$) sigue una distribución con una cola hacia la izquierda, lo que podría indicar que una **gran parte de los municipios tienen una alta producción** o que los valores de producción más bajos son menos frecuentes.
* **Temperatura (TEMP)**: Posee una distribución amplia y notablemente sesgada a la derecha, con la mayoría de los municipios experimentando **temperaturas más altas** ($\text{Media} \approx 20.7$, $\text{Std}: 5.3$), lo cual es relevante para la eficiencia de los paneles fotovoltaicos.

---

### Correlaciones entre Variables

La **Matriz de Correlación** proporciona *insights* clave sobre la interdependencia de las variables:

* **Producción FV (PVOUT) y DNI:** Existe una **correlación muy fuerte y positiva** ($+0.984$), lo que confirma que la **Irradiación Normal Directa es el factor dominante** que impulsa la producción de energía fotovoltaica en Colombia. Las regiones con alto DNI tendrán el mayor PVOUT.
* **DIF y DNI/PVOUT:** La Irradiación Difusa tiene una **correlación negativa** tanto con DNI ($\approx -0.393$) como con PVOUT ($\approx -0.273$). Esto es típico, ya que las regiones con cielos más despejados (alto DNI) suelen tener menor nubosidad, lo que reduce la componente difusa.
* **Temperatura y PVOUT:** La correlación entre Temperatura y PVOUT es **positiva pero baja** ($+0.333$). Aunque la temperatura puede aumentar en las regiones soleadas, el factor determinante para la producción sigue siendo la radiación solar, mientras que el efecto negativo del calor en la eficiencia de los paneles no anula completamente la ventaja de la alta insolación.

---

### Potencial Geográfico y Regionalización

El análisis geográfico y de *clustering* destaca la distribución heterogénea del potencial:

* **Potencial Fotovoltaico (Mapa de Círculos)**: Muestra que las regiones con **potencial Alto ($> 1400 \text{ kWh/m}^2$)** se concentran principalmente en la **Región Caribe** y la **Región Andina central y oriental**.
* **Municipios de Alta Producción (Gráfico de Barras)**: Los 20 municipios con **mayor producción FV estimada** (superando los $1700 \text{ kWh/kWp}$) se encuentran en departamentos como Santander, La Guajira, y Cesar, como lo demuestran **Guapota** y **Palmas del Socorro**. Estas son las ubicaciones **óptimas** para la inversión en grandes plantas solares.
* **Municipios de Baja Producción (Gráfico de Barras)**: Los municipios con **menor producción** (cerca de $1050 \text{ kWh/kWp}$) se encuentran principalmente en zonas de la costa Pacífica y partes de la Amazonía, como **El Carmen de Atrato** y **Ricaurte**, que probablemente experimentan mayor nubosidad o menor radiación.
* **Clustering (Mapa de Puntos Agrupados)**: La clasificación regional (**Clusters de Potencial Energético**) confirma que el **"Excelente Potencial" (púrpura/rojo)** se ubica predominantemente en el **norte de Colombia** y partes de la región andina central (Boyacá, Santander), mientras que el **"Bajo Potencial" (azul)** se localiza en las vastas regiones de la **Amazonía y el Orinoco**, y el **"Bajo" a "Moderado" (rojo/naranja)** en la **Costa Pacífica**.

El análisis demuestra que el **alto potencial de energía fotovoltaica en Colombia se concentra en la región Andina oriental y el Caribe**, fuertemente correlacionado con una alta **Irradiación Normal Directa (DNI)**. Esta información es esencial para la **Identificación óptima de ubicaciones** y la **Estimación precisa de ROI** que se espera del proyecto, permitiendo dirigir las inversiones hacia las zonas de mayor rendimiento.

## Contacto

Para preguntas o colaboraciones sobre el proyecto de energía solar, contactar al equipo de desarrollo.

*Este proyecto forma parte del análisis de potencial de energía solar en Colombia utilizando datos satelitales de alta calidad y técnicas avanzadas de machine learning para predicción de rendimiento solar.*