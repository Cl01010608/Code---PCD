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
├── airflow_pipeline/
│   └── eda_solar_pipeline.py      # Pipeline de Airflow para EDA automatizado
├── data/
│   └── resultados_municipios.csv  # Datos agregados por municipio
├── output/
│   └── [resultados del pipeline]
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

## Pipeline de Airflow

El pipeline automatizado incluye las siguientes tareas:

1. **Validación de datos** - Verificación de integridad
2. **Limpieza de datos** - Tratamiento de valores faltantes
3. **Análisis estadístico** - Generación de estadísticas descriptivas
4. **Análisis de correlaciones** - Matriz de correlación
5. **Detección de outliers** - Identificación de valores atípicos
6. **Visualizaciones** - Generación de gráficos
7. **Clustering** - Agrupación de municipios
8. **PCA** - Análisis de componentes principales
9. **Reporte final** - Consolidación de resultados

## Instalación y Uso

### Prerrequisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn rasterio geopandas apache-airflow
```

### Ejecución del Pipeline

1. **Configurar Airflow:**
```bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

2. **Copiar el pipeline:**
```bash
cp airflow_pipeline/eda_solar_pipeline.py $AIRFLOW_HOME/dags/
```

3. **Iniciar Airflow:**
```bash
airflow webserver --port 8080
airflow scheduler
```

4. **Ejecutar el DAG desde la interfaz web de Airflow**

### Ejecución Manual

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
- **Reporte consolidado** con hallazgos principales

## Licencia

Los datos están bajo **Creative Commons CC BY 4.0**, con origen en **Solargis (2021)** y publicados en la plataforma del Banco Mundial.

## Estado del Proyecto

**Fecha actual:** 8 de octubre de 2025

### ✅ Fases Completadas

- **Fase 1: Revisión Inicial** - Exploración preliminar de datos GeoTIFF
- **Fase 2: EDA Completo** - Análisis exploratorio detallado con pipeline automatizado
- **Fase 3: Pipeline Airflow** - Implementación de flujos automatizados para EDA

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
- [ ] Creación de variables lag y rolling statistics
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
- **Resolución espacial:** 9-120 arc-segundos

### Outputs Generados
- **Reportes:** 8 reportes automatizados
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

## Contacto

Para preguntas o colaboraciones sobre el proyecto de energía solar, contactar al equipo de desarrollo.

*Este proyecto forma parte del análisis de potencial de energía solar en Colombia utilizando datos satelitales de alta calidad y técnicas avanzadas de machine learning para predicción de rendimiento solar.*