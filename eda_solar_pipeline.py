
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
from pathlib import Path

# Configuración de matplotlib para renderizado
def setup_matplotlib_for_plotting():
    """
    Setup matplotlib y seaborn para plotting con configuración adecuada.
    Llamar esta función antes de crear cualquier plot para asegurar renderizado correcto.
    """
    warnings.filterwarnings('default')  # Mostrar todas las advertencias
    
    # Configurar matplotlib para modo no interactivo
    plt.switch_backend("Agg")
    
    # Establecer estilo de gráficos
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Configurar fuentes apropiadas para compatibilidad multiplataforma
    # Debe establecerse después de style.use, sino será sobrescrito
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

# Argumentos por defecto del DAG
default_args = {
    'owner': 'MiniMax Agent',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 8),
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Definición del DAG
dag = DAG(
    'eda_solar_pipeline',
    default_args=default_args,
    description='Pipeline EDA para datos de energía solar Colombia',
    schedule_interval=timedelta(days=1),
    tags=['eda', 'solar', 'colombia', 'analytics']
)

# Variables globales
DATA_PATH = '/workspace/data/resultados_municipios.csv'
OUTPUT_PATH = '/workspace/output'
FIGURES_PATH = f'{OUTPUT_PATH}/figures'
REPORTS_PATH = f'{OUTPUT_PATH}/reports'

# === FUNCIONES DE LAS TAREAS ===

def setup_directories(**context):
    """Crear directorios necesarios para el pipeline"""
    directories = [OUTPUT_PATH, FIGURES_PATH, REPORTS_PATH]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado/verificado: {directory}")
    
    print("Configuración de directorios completada")

def validate_data(**context):
    """Validar integridad y calidad de los datos"""
    print("Iniciando validación de datos...")
    
    # Verificar existencia del archivo
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Archivo de datos no encontrado: {DATA_PATH}")
    
    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    
    # Validaciones básicas
    validations = {
        'total_registros': len(df),
        'total_columnas': len(df.columns),
        'valores_faltantes': df.isnull().sum().sum(),
        'duplicados': df.duplicated().sum(),
        'departamentos_unicos': df['Departamento'].nunique(),
        'municipios_unicos': df['Municipio'].nunique()
    }
    
    print("=== REPORTE DE VALIDACIÓN ===")
    for key, value in validations.items():
        print(f"{key}: {value}")
    
    # Guardar reporte de validación
    validation_report = pd.DataFrame.from_dict(validations, orient='index', columns=['Valor'])
    validation_report.to_csv(f"{REPORTS_PATH}/validation_report.csv")
    
    # Verificar columnas esperadas
    expected_columns = ['Departamento', 'Municipio']
    solar_variables = ['DIF', 'DNI', 'GHI', 'GTI', 'OPTA', 'PVOUT', 'TEMP']
    stats_suffixes = ['_min', '_max', '_mean', '_std', '_median']
    
    for var in solar_variables:
        for suffix in stats_suffixes:
            expected_columns.append(f"{var}{suffix}")
    
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columnas faltantes: {missing_columns}")
    
    print("Validación de datos completada exitosamente")
    return validations

def clean_data(**context):
    """Limpieza y preprocesamiento de datos"""
    print("Iniciando limpieza de datos...")
    
    df = pd.read_csv(DATA_PATH)
    
    # Estadísticas antes de limpieza
    before_cleaning = {
        'registros': len(df),
        'valores_faltantes': df.isnull().sum().sum()
    }
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Manejar valores faltantes (si los hay)
    # Para este dataset, usar forward fill o interpolación según el caso
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Verificar valores negativos en variables que no deberían tenerlos
    solar_vars = ['DIF_mean', 'DNI_mean', 'GHI_mean', 'GTI_mean', 'PVOUT_mean']
    for var in solar_vars:
        if var in df.columns:
            negative_count = (df[var] < 0).sum()
            if negative_count > 0:
                print(f"Advertencia: {negative_count} valores negativos en {var}")
    
    # Estadísticas después de limpieza
    after_cleaning = {
        'registros': len(df),
        'valores_faltantes': df.isnull().sum().sum()
    }
    
    # Guardar datos limpios
    clean_data_path = f"{OUTPUT_PATH}/datos_limpios.csv"
    df.to_csv(clean_data_path, index=False)
    
    # Reporte de limpieza
    cleaning_report = pd.DataFrame({
        'Antes': before_cleaning,
        'Después': after_cleaning
    })
    cleaning_report.to_csv(f"{REPORTS_PATH}/cleaning_report.csv")
    
    print("Limpieza de datos completada")
    print(f"Registros antes: {before_cleaning['registros']}, después: {after_cleaning['registros']}")
    
    return clean_data_path

def descriptive_statistics(**context):
    """Generar estadísticas descriptivas"""
    print("Generando estadísticas descriptivas...")
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    
    # Variables de interés para análisis
    mean_vars = [col for col in df.columns if '_mean' in col]
    
    # Estadísticas descriptivas básicas
    stats_desc = df[mean_vars].describe()
    stats_desc.to_csv(f"{REPORTS_PATH}/estadisticas_descriptivas.csv")
    
    # Estadísticas por departamento
    stats_by_dept = df.groupby('Departamento')[mean_vars].mean()
    stats_by_dept.to_csv(f"{REPORTS_PATH}/estadisticas_por_departamento.csv")
    
    # Top municipios por variables
    top_reports = {}
    for var in ['GHI_mean', 'DNI_mean', 'PVOUT_mean']:
        if var in df.columns:
            top_10 = df.nlargest(10, var)[['Departamento', 'Municipio', var]]
            top_10.to_csv(f"{REPORTS_PATH}/top_10_{var.replace('_mean', '').lower()}.csv", index=False)
            top_reports[var] = top_10
    
    print("Estadísticas descriptivas generadas")
    return f"{REPORTS_PATH}/estadisticas_descriptivas.csv"

def correlation_analysis(**context):
    """Análisis de correlaciones entre variables"""
    print("Realizando análisis de correlaciones...")
    
    setup_matplotlib_for_plotting()
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    mean_vars = [col for col in df.columns if '_mean' in col]
    
    # Matriz de correlación
    correlation_matrix = df[mean_vars].corr()
    correlation_matrix.to_csv(f"{REPORTS_PATH}/matriz_correlacion.csv")
    
    # Visualización de matriz de correlación
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, mask=mask, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación - Variables Solares', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/matriz_correlacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlaciones más altas (excluyendo auto-correlaciones)
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            corr_pairs.append({'Variable_1': var1, 'Variable_2': var2, 'Correlacion': corr_value})
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Correlacion', key=abs, ascending=False)
    corr_df.to_csv(f"{REPORTS_PATH}/correlaciones_ordenadas.csv", index=False)
    
    print("Análisis de correlaciones completado")
    return f"{REPORTS_PATH}/matriz_correlacion.csv"

def outlier_detection(**context):
    """Detección de outliers usando IQR y Z-score"""
    print("Detectando outliers...")
    
    setup_matplotlib_for_plotting()
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    mean_vars = [col for col in df.columns if '_mean' in col]
    
    outliers_report = []
    
    # Método IQR para cada variable
    for var in mean_vars:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
        
        outliers_report.append({
            'Variable': var,
            'Metodo': 'IQR',
            'Total_Outliers': len(outliers),
            'Porcentaje': (len(outliers) / len(df)) * 100,
            'Limite_Inferior': lower_bound,
            'Limite_Superior': upper_bound
        })
    
    # Z-score method
    for var in mean_vars:
        z_scores = np.abs(stats.zscore(df[var]))
        outliers = df[z_scores > 3]
        
        outliers_report.append({
            'Variable': var,
            'Metodo': 'Z-Score',
            'Total_Outliers': len(outliers),
            'Porcentaje': (len(outliers) / len(df)) * 100,
            'Limite_Inferior': -3,
            'Limite_Superior': 3
        })
    
    outliers_df = pd.DataFrame(outliers_report)
    outliers_df.to_csv(f"{REPORTS_PATH}/outliers_detection.csv", index=False)
    
    # Visualización de boxplots para variables principales
    main_vars = ['GHI_mean', 'DNI_mean', 'PVOUT_mean', 'TEMP_mean']
    available_vars = [var for var in main_vars if var in df.columns]
    
    if available_vars:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(available_vars):
            if i < len(axes):
                df.boxplot(column=var, ax=axes[i])
                axes[i].set_title(f'Boxplot - {var}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(len(available_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/outliers_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Detección de outliers completada")
    return f"{REPORTS_PATH}/outliers_detection.csv"

def generate_visualizations(**context):
    """Generar visualizaciones principales"""
    print("Generando visualizaciones...")
    
    setup_matplotlib_for_plotting()
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    
    # 1. Distribuciones de variables principales
    main_vars = ['GHI_mean', 'DNI_mean', 'PVOUT_mean', 'TEMP_mean']
    available_vars = [var for var in main_vars if var in df.columns]
    
    if available_vars:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(available_vars):
            if i < len(axes):
                axes[i].hist(df[var], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribución de {var}', fontweight='bold')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(available_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/distribuciones.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Comparación por departamentos (top 10)
    if 'GHI_mean' in df.columns:
        dept_means = df.groupby('Departamento')['GHI_mean'].mean().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        dept_means.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Top 10 Departamentos - Irradiación Global Horizontal Media (GHI)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Departamento')
        plt.ylabel('GHI Mean (kWh/m²)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/top_departamentos_ghi.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Scatter plot GHI vs PVOUT
    if 'GHI_mean' in df.columns and 'PVOUT_mean' in df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(df['GHI_mean'], df['PVOUT_mean'], alpha=0.6, s=50)
        plt.xlabel('GHI Mean (kWh/m²)')
        plt.ylabel('PVOUT Mean (kWh/kWp)')
        plt.title('Relación entre Irradiación Global y Producción Fotovoltaica', 
                 fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Agregar línea de tendencia
        z = np.polyfit(df['GHI_mean'], df['PVOUT_mean'], 1)
        p = np.poly1d(z)
        plt.plot(df['GHI_mean'], p(df['GHI_mean']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/ghi_vs_pvout.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizaciones generadas exitosamente")
    return f"{FIGURES_PATH}/"

def clustering_analysis(**context):
    """Análisis de clustering de municipios"""
    print("Realizando análisis de clustering...")
    
    setup_matplotlib_for_plotting()
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    mean_vars = [col for col in df.columns if '_mean' in col]
    
    # Preparar datos para clustering
    X = df[mean_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determinar número óptimo de clusters usando elbow method
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar k Óptimo', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/elbow_method.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aplicar K-means con k=4 (valor típico para análisis geográfico)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Agregar clusters al dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    df_clustered.to_csv(f"{OUTPUT_PATH}/datos_con_clusters.csv", index=False)
    
    # Análisis de clusters
    cluster_summary = df_clustered.groupby('Cluster')[mean_vars].mean()
    cluster_summary.to_csv(f"{REPORTS_PATH}/cluster_summary.csv")
    
    # Conteo de municipios por cluster
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    cluster_counts.to_csv(f"{REPORTS_PATH}/cluster_counts.csv")
    
    # Visualización de clusters en 2D (usando las dos primeras componentes principales)
    if len(mean_vars) >= 2:
        pca_temp = PCA(n_components=2)
        X_pca = pca_temp.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i in range(optimal_k):
            cluster_points = X_pca[clusters == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)
        
        plt.xlabel('Primera Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.title('Clusters de Municipios (Proyección PCA)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}/clusters_pca.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Clustering completado con {optimal_k} clusters")
    return f"{OUTPUT_PATH}/datos_con_clusters.csv"

def pca_analysis(**context):
    """Análisis de Componentes Principales"""
    print("Realizando análisis PCA...")
    
    setup_matplotlib_for_plotting()
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    mean_vars = [col for col in df.columns if '_mean' in col]
    
    # Preparar datos
    X = df[mean_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Varianza explicada
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Guardar resultados
    pca_results = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Varianza_Explicada': explained_variance_ratio,
        'Varianza_Acumulada': cumulative_variance
    })
    pca_results.to_csv(f"{REPORTS_PATH}/pca_variance_explained.csv", index=False)
    
    # Loadings (correlaciones entre variables originales y componentes)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        index=mean_vars
    )
    loadings.to_csv(f"{REPORTS_PATH}/pca_loadings.csv")
    
    # Visualización de varianza explicada
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Varianza por componente
    plt.subplot(2, 1, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente Principal', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Varianza acumulada
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Varianza')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Varianza')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada')
    plt.title('Varianza Acumulada por Número de Componentes', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/pca_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap de loadings para las primeras 4 componentes
    n_components_to_show = min(4, len(explained_variance_ratio))
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings.iloc[:, :n_components_to_show], 
                annot=True, cmap='coolwarm', center=0, 
                cbar_kws={"shrink": .8})
    plt.title('Loadings de las Primeras Componentes Principales', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/pca_loadings_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Análisis PCA completado")
    print(f"Primeras 3 componentes explican {cumulative_variance[2]:.2%} de la varianza")
    
    return f"{REPORTS_PATH}/pca_variance_explained.csv"

def generate_final_report(**context):
    """Generar reporte final consolidado"""
    print("Generando reporte final...")
    
    df = pd.read_csv(f"{OUTPUT_PATH}/datos_limpios.csv")
    
    # Leer resultados de análisis previos
    validation_report = pd.read_csv(f"{REPORTS_PATH}/validation_report.csv", index_col=0)
    stats_desc = pd.read_csv(f"{REPORTS_PATH}/estadisticas_descriptivas.csv", index_col=0)
    outliers_report = pd.read_csv(f"{REPORTS_PATH}/outliers_detection.csv")
    pca_results = pd.read_csv(f"{REPORTS_PATH}/pca_variance_explained.csv")
    
    # Crear reporte en texto
    report_lines = [
        "===== REPORTE FINAL - ANÁLISIS EXPLORATORIO DE DATOS =====",
        "Datos de Energía Solar en Colombia - Global Solar Atlas 2.0",
        f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n=== RESUMEN EJECUTIVO ===",
        f"• Total de municipios analizados: {validation_report.loc['municipios_unicos', 'Valor']}",
        f"• Total de departamentos: {validation_report.loc['departamentos_unicos', 'Valor']}",
        f"• Variables analizadas: {len([col for col in df.columns if '_mean' in col])}",
        "\n=== ESTADÍSTICAS PRINCIPALES ==="
    ]
    
    # Agregar estadísticas de variables principales
    main_vars = ['GHI_mean', 'DNI_mean', 'PVOUT_mean', 'TEMP_mean']
    for var in main_vars:
        if var in stats_desc.columns:
            mean_val = stats_desc.loc['mean', var]
            std_val = stats_desc.loc['std', var]
            report_lines.extend([
                f"• {var}:",
                f"  - Media: {mean_val:.2f}",
                f"  - Desviación estándar: {std_val:.2f}"
            ])
    
    # Agregar información de outliers
    report_lines.extend([
        "\n=== OUTLIERS DETECTADOS ==="
    ])
    
    for var in main_vars:
        if var in outliers_report['Variable'].values:
            iqr_outliers = outliers_report[
                (outliers_report['Variable'] == var) & 
                (outliers_report['Metodo'] == 'IQR')
            ]['Total_Outliers'].iloc[0]
            report_lines.append(f"• {var}: {iqr_outliers} outliers (método IQR)")
    
    # Agregar información de PCA
    report_lines.extend([
        "\n=== ANÁLISIS DE COMPONENTES PRINCIPALES ===",
        f"• Primera componente explica: {pca_results.iloc[0]['Varianza_Explicada']:.2%} de la varianza",
        f"• Primeras 3 componentes explican: {pca_results.iloc[2]['Varianza_Acumulada']:.2%} de la varianza"
    ])
    
    # Top municipios
    if 'GHI_mean' in df.columns:
        top_ghi = df.nlargest(5, 'GHI_mean')
        report_lines.extend([
            "\n=== TOP 5 MUNICIPIOS - IRRADIACIÓN GLOBAL HORIZONTAL ==="
        ])
        for idx, row in top_ghi.iterrows():
            report_lines.append(
                f"• {row['Municipio']}, {row['Departamento']}: {row['GHI_mean']:.2f} kWh/m²"
            )
    
    # Archivos generados
    report_lines.extend([
        "\n=== ARCHIVOS GENERADOS ===",
        "• Datos limpios: output/datos_limpios.csv",
        "• Datos con clusters: output/datos_con_clusters.csv",
        "• Figuras: output/figures/",
        "• Reportes detallados: output/reports/",
        "\n=== FIN DEL REPORTE ==="
    ])
    
    # Guardar reporte
    with open(f"{REPORTS_PATH}/reporte_final.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    # También crear versión en CSV con métricas clave
    summary_metrics = {
        'Metrica': [
            'Total_Municipios', 'Total_Departamentos', 'Variables_Analizadas',
            'GHI_Media', 'PVOUT_Media', 'Outliers_GHI', 'Varianza_PC1'
        ],
        'Valor': [
            validation_report.loc['municipios_unicos', 'Valor'],
            validation_report.loc['departamentos_unicos', 'Valor'],
            len([col for col in df.columns if '_mean' in col]),
            stats_desc.loc['mean', 'GHI_mean'] if 'GHI_mean' in stats_desc.columns else 0,
            stats_desc.loc['mean', 'PVOUT_mean'] if 'PVOUT_mean' in stats_desc.columns else 0,
            outliers_report[
                (outliers_report['Variable'] == 'GHI_mean') & 
                (outliers_report['Metodo'] == 'IQR')
            ]['Total_Outliers'].iloc[0] if 'GHI_mean' in outliers_report['Variable'].values else 0,
            pca_results.iloc[0]['Varianza_Explicada']
        ]
    }
    
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(f"{REPORTS_PATH}/resumen_ejecutivo.csv", index=False)
    
    print("Reporte final generado exitosamente")
    print(f"Archivos disponibles en: {REPORTS_PATH}")
    
    return f"{REPORTS_PATH}/reporte_final.txt"

# === DEFINICIÓN DE TAREAS ===

# Tarea de configuración inicial
setup_task = PythonOperator(
    task_id='setup_directories',
    python_callable=setup_directories,
    dag=dag
)

# Tarea de validación de datos
validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

# Tarea de limpieza de datos
clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    dag=dag
)

# Tarea de estadísticas descriptivas
stats_task = PythonOperator(
    task_id='descriptive_statistics',
    python_callable=descriptive_statistics,
    dag=dag
)

# Tarea de análisis de correlaciones
corr_task = PythonOperator(
    task_id='correlation_analysis',
    python_callable=correlation_analysis,
    dag=dag
)

# Tarea de detección de outliers
outliers_task = PythonOperator(
    task_id='outlier_detection',
    python_callable=outlier_detection,
    dag=dag
)

# Tarea de generación de visualizaciones
viz_task = PythonOperator(
    task_id='generate_visualizations',
    python_callable=generate_visualizations,
    dag=dag
)

# Tarea de clustering
cluster_task = PythonOperator(
    task_id='clustering_analysis',
    python_callable=clustering_analysis,
    dag=dag
)

# Tarea de PCA
pca_task = PythonOperator(
    task_id='pca_analysis',
    python_callable=pca_analysis,
    dag=dag
)

# Tarea de reporte final
report_task = PythonOperator(
    task_id='generate_final_report',
    python_callable=generate_final_report,
    dag=dag
)

# Tarea de notificación (opcional)
notify_task = BashOperator(
    task_id='notify_completion',
    bash_command='echo "Pipeline EDA Solar completado exitosamente en $(date)"',
    dag=dag
)

# === DEFINICIÓN DE DEPENDENCIAS ===

# Configuración de la secuencia de tareas
setup_task >> validate_task >> clean_task

# Análisis en paralelo después de limpieza
clean_task >> [stats_task, corr_task, outliers_task]

# Visualizaciones después de estadísticas
stats_task >> viz_task

# Análisis avanzados en paralelo
clean_task >> [cluster_task, pca_task]

# Reporte final después de todos los análisis
[viz_task, corr_task, outliers_task, cluster_task, pca_task] >> report_task >> notify_task

if __name__ == "__main__":
    print("DAG EDA Solar Pipeline definido correctamente")
    print(f"Tareas incluidas: {len(dag.task_dict)} tareas")
    for task_id in dag.task_dict:
        print(f"  - {task_id}")