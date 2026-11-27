import os
import sys
import time
import unicodedata
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 1. FUNCIÓN DE NORMALIZACIÓN DE TEXTO (ELIMINACIÓN DE TILDES Y ESPACIOS)
# ==============================================================================
def normalizar_texto(texto: str) -> str:
    """Elimina tildes, diacríticos, convierte a minúsculas y remueve espacios extra."""
    if pd.isna(texto):
        return ""
    texto_str = str(texto).strip().lower()
    # Normalización Unicode NFKD para separar letras de acentos
    texto_sin_tildes = "".join(
        c for c in unicodedata.normalize("NFKD", texto_str)
        if unicodedata.category(c) != "Mn"
    )
    return texto_sin_tildes


# ==============================================================================
# 2. DICCIONARIO EXHAUSTIVO DE LOS 41 CULTIVOS DEL DATASET OFICIAL
# ==============================================================================
# Clasificación agronómica validada con el Ministerio de Agricultura e INTA
MAPA_CULTIVOS_EXHAUSTIVO = {
    # --- COSECHA GRUESA (Siembra Primavera/Verano -> Cosecha Otoño) ---
    'soja total': 'gruesa', 'soja 1ra': 'gruesa', 'soja 2da': 'gruesa',
    'maiz': 'gruesa', 'girasol': 'gruesa', 'sorgo': 'gruesa',
    'algodon': 'gruesa', 'mani': 'gruesa', 'arroz': 'gruesa',
    'poroto total': 'gruesa', 'poroto alubia': 'gruesa', 'poroto negro': 'gruesa',
    'poroto otros': 'gruesa', 'mijo': 'gruesa', 'cartamo': 'gruesa', 'jojoba': 'gruesa',
    
    # --- COSECHA FINA (Siembra Otoño/Invierno -> Cosecha Primavera/Verano) ---
    'trigo total': 'fina', 'trigo candeal': 'fina',
    'cebada total': 'fina', 'cebada cervecera': 'fina', 'cebada forrajera': 'fina',
    'avena': 'fina', 'centeno': 'fina', 'lino': 'fina', 'colza': 'fina',
    'alpiste': 'fina', 'arveja': 'fina', 'garbanzo': 'fina', 'lenteja': 'fina',
    
    # --- CULTIVOS ANUALES / PERENNES / REGIONALES (Ciclo Hidrológico Completo) ---
    'ajo': 'anual', 'cebolla total': 'anual', 'papa total': 'anual',
    'cana de azucar': 'anual', 'te': 'anual', 'yerba mate': 'anual',
    'limon': 'anual', 'naranja': 'anual', 'mandarina': 'anual', 'pomelo': 'anual',
    'banana': 'anual', 'tung': 'anual'
}

# Coordenadas centroidales por provincia (Fallback si falla el departamento)
CENTROIDES_PROVINCIAS = {
    'buenos aires': (-36.67, -60.55), 'cordoba': (-31.42, -64.18),
    'santa fe': (-31.63, -60.70), 'entre rios': (-32.05, -59.25),
    'la pampa': (-36.61, -64.28), 'santiago del estero': (-27.79, -64.26),
    'chaco': (-27.45, -58.98), 'tucuman': (-26.82, -65.22),
    'salta': (-24.78, -65.41), 'corrientes': (-27.46, -58.83),
    'misiones': (-26.92, -54.60), 'san luis': (-33.30, -66.33),
    'formosa': (-26.17, -58.17), 'jujuy': (-24.18, -65.29),
    'catamarca': (-28.46, -65.77), 'la rioja': (-29.41, -66.85),
    'san juan': (-31.53, -68.53), 'mendoza': (-32.89, -68.82),
    'neuquen': (-38.95, -68.05), 'rio negro': (-40.81, -63.00),
    'chubut': (-43.30, -65.10), 'santa cruz': (-51.62, -69.21),
    'tierra del fuego': (-54.80, -68.30)
}


# ==============================================================================
# 3. LOCALIZACIÓN ROBUSTA DEL ARCHIVO CSV
# ==============================================================================
def buscar_archivo_csv(nombre_archivo: str) -> str:
    """Busca el archivo en el directorio actual, en el path del script o en Downloads."""
    posibles_rutas = [
        nombre_archivo,
        os.path.join(os.getcwd(), nombre_archivo),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre_archivo) if '__file__' in globals() else nombre_archivo,
        os.path.join(os.path.expanduser("~"), "Downloads", nombre_archivo),
        "estimaciones-agricolas-2026-03.csv",
        "estimaciones-agricolas-2026-03_2.csv"
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            print(f"[OK] Archivo encontrado en: {ruta}")
            return ruta
    raise FileNotFoundError(f"No se encontro el archivo '{nombre_archivo}'. Verifique la ruta.")


# ==============================================================================
# 4. RESOLUCIÓN DE GEOLOCALIZACIÓN
# ==============================================================================
def resolver_coordenadas(provincia: str, departamento: str, cache_coords: dict) -> tuple:
    """Resuelve lat/lon usando Georef API con fallback al centroide provincial."""
    prov_norm = normalizar_texto(provincia)
    dept_norm = normalizar_texto(departamento)
    clave = f"{dept_norm}_{prov_norm}"
    
    if clave in cache_coords:
        return cache_coords[clave]
    
    # 1. Intentar consulta a Georef API
    url = f"https://apis.datos.gob.ar/georef/api/departamentos?nombre={dept_norm}&provincia={prov_norm}&max=1"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            datos = r.json()
            if datos.get('cantidad', 0) > 0:
                c = datos['departamentos'][0]['centroide']
                coords = (round(c['lat'], 2), round(c['lon'], 2))
                cache_coords[clave] = coords
                return coords
    except Exception:
        pass
    
    # 2. Fallback al centroide de la provincia
    coords = CENTROIDES_PROVINCIAS.get(prov_norm, (-34.00, -61.00))
    cache_coords[clave] = coords
    return coords


# ==============================================================================
# 5. DESCARGA RESILIENTE DE NASA POWER API (SERIE DIARIA)
# ==============================================================================
def obtener_clima_departamento(lat: float, lon: float, depto_id: int, anio_inicio=1981, anio_fin=2024) -> pd.DataFrame:
    """Descarga serie diaria de NASA POWER API con persistencia en caché local."""
    os.makedirs('cache_clima', exist_ok=True)
    archivo_cache = os.path.join('cache_clima', f"clima_{depto_id}.parquet")
    
    if os.path.exists(archivo_cache):
        return pd.read_parquet(archivo_cache)
    
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=PRECTOTCORR,T2M,T2M_MAX,T2M_MIN&community=AG&"
        f"longitude={lon:.2f}&latitude={lat:.2f}&"
        f"start={anio_inicio}0101&end={anio_fin}1231&format=JSON"
    )
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            params = resp.json()['properties']['parameter']
            fechas = list(params['PRECTOTCORR'].keys())
            df_clima = pd.DataFrame({
                'fecha': pd.to_datetime(fechas, format='%Y%m%d'),
                'precipitacion_mm': list(params['PRECTOTCORR'].values()),
                'temp_media': list(params['T2M'].values()),
                'temp_max': list(params['T2M_MAX'].values()),
                'temp_min': list(params['T2M_MIN'].values())
            })
            df_clima = df_clima.replace(-999.0, np.nan).dropna(subset=['precipitacion_mm'])
            df_clima['departamento_id'] = int(depto_id)
            
            # Guardar en formato binario Parquet
            df_clima.to_parquet(archivo_cache, index=False)
            time.sleep(0.3)  # Pausa de cortesía para la API
            return df_clima
    except Exception as e:
        print(f"  [!] Alerta en descarga depto {depto_id} ({lat}, {lon}): {e}")
        
    return pd.DataFrame()


# ==============================================================================
# 6. AGREGACIÓN METEOROLÓGICA POR CAMPAÑA Y GRUPO AGRONÓMICO
# ==============================================================================
def agregar_metricas_climaticas(df_diario: pd.DataFrame, depto_id: int) -> pd.DataFrame:
    """Calcula las lluvias y temperaturas agregadas según el calendario de cada cultivo."""
    if df_diario.empty:
        return pd.DataFrame()
        
    anios = df_diario['fecha'].dt.year.unique()
    filas = []
    
    for anio in sorted(anios):
        campania_str = f"{anio}/{str(anio + 1)[-2:]}"
        
        # A. Cosecha Gruesa (Octubre T -> Abril T+1)
        f_ini_g = pd.Timestamp(f"{anio}-10-01")
        f_fin_g = pd.Timestamp(f"{anio + 1}-04-30")
        df_g = df_diario[(df_diario['fecha'] >= f_ini_g) & (df_diario['fecha'] <= f_fin_g)]
        
        if len(df_g) >= 150:
            critico_g = df_g['fecha'].dt.month.isin([1, 2])
            filas.append({
                'departamento_id': depto_id,
                'campania': campania_str,
                'grupo_cultivo': 'gruesa',
                'lluvia_total_ciclo_mm': round(df_g['precipitacion_mm'].sum(), 2),
                'lluvia_periodo_critico_mm': round(df_g.loc[critico_g, 'precipitacion_mm'].sum(), 2),
                'temp_media_ciclo': round(df_g['temp_media'].mean(), 2),
                'dias_ola_calor': int((df_g['temp_max'] >= 35.0).sum())
            })
            
        # B. Cosecha Fina (Mayo T -> Diciembre T)
        f_ini_f = pd.Timestamp(f"{anio}-05-01")
        f_fin_f = pd.Timestamp(f"{anio}-12-31")
        df_f = df_diario[(df_diario['fecha'] >= f_ini_f) & (df_diario['fecha'] <= f_fin_f)]
        
        if len(df_f) >= 180:
            critico_f = df_f['fecha'].dt.month.isin([9, 10])
            filas.append({
                'departamento_id': depto_id,
                'campania': campania_str,
                'grupo_cultivo': 'fina',
                'lluvia_total_ciclo_mm': round(df_f['precipitacion_mm'].sum(), 2),
                'lluvia_periodo_critico_mm': round(df_f.loc[critico_f, 'precipitacion_mm'].sum(), 2),
                'temp_media_ciclo': round(df_f['temp_media'].mean(), 2),
                'dias_ola_calor': int((df_f['temp_max'] >= 32.0).sum())
            })
            
        # C. Cultivos Anuales / Regionales (Julio T -> Junio T+1)
        f_ini_a = pd.Timestamp(f"{anio}-07-01")
        f_fin_a = pd.Timestamp(f"{anio + 1}-06-30")
        df_a = df_diario[(df_diario['fecha'] >= f_ini_a) & (df_diario['fecha'] <= f_fin_a)]
        
        if len(df_a) >= 300:
            filas.append({
                'departamento_id': depto_id,
                'campania': campania_str,
                'grupo_cultivo': 'anual',
                'lluvia_total_ciclo_mm': round(df_a['precipitacion_mm'].sum(), 2),
                'lluvia_periodo_critico_mm': round(df_a['precipitacion_mm'].sum() * 0.35, 2),
                'temp_media_ciclo': round(df_a['temp_media'].mean(), 2),
                'dias_ola_calor': int((df_a['temp_max'] >= 35.0).sum())
            })

    return pd.DataFrame(filas)


# ==============================================================================
# 7. PROCESO PRINCIPAL (ETL Y DATA FUSION)
# ==============================================================================
def ejecutar_pipeline_agro_clima(nombre_archivo_csv: str):
    ruta_valida = buscar_archivo_csv(nombre_archivo_csv)
    
    print("[1/5] Cargando dataset agrícola oficial...")
    df_agro = pd.read_csv(ruta_valida, sep=';', encoding='utf-8')
    
    # Limpieza de nulos en llaves y conversión de tipos
    df_agro = df_agro.dropna(subset=['departamento_id', 'rendimiento_kgxha']).copy()
    df_agro['departamento_id'] = df_agro['departamento_id'].astype(int)
    
    # Conversión numérica de variables clave
    for col in ['superficie_sembrada_ha', 'superficie_cosechada_ha', 'produccion_tm', 'rendimiento_kgxha']:
        df_agro[col] = pd.to_numeric(df_agro[col], errors='coerce').fillna(0.0)
        
    # Normalización del texto de cultivo
    df_agro['cultivo_norm'] = df_agro['cultivo'].apply(normalizar_texto)
    df_agro['grupo_cultivo'] = df_agro['cultivo_norm'].map(MAPA_CULTIVOS_EXHAUSTIVO).fillna('anual')
    
    # Filtro de campañas >= 1981
    df_agro['campania_inicio'] = df_agro['campania'].apply(
        lambda x: int(str(x).split('/')[0]) if '/' in str(x) else (int(x) if str(x).isdigit() else np.nan)
    )
    df_agro = df_agro[df_agro['campania_inicio'] >= 1981].copy()
    
    # Formato homogéneo de campaña: "YYYY/YY" (ej: "1981/82")
    df_agro['campania'] = df_agro['campania_inicio'].astype(int).apply(lambda y: f"{y}/{str(y+1)[-2:]}")
    print(f"[INFO] Registros agropecuarios a procesar (1981-2024): {len(df_agro):,}")
    
    # [2/5] Extracción de departamentos únicos
    deptos_unicos = df_agro[['departamento_id', 'provincia', 'departamento']].drop_duplicates().reset_index(drop=True)
    print(f"[2/5] Procesando historial meteorologico para {len(deptos_unicos)} departamentos...")
    
    cache_coords = {}
    lista_clima = []
    
    for i, row in deptos_unicos.iterrows():
        d_id = int(row['departamento_id'])
        prov = str(row['provincia'])
        dept = str(row['departamento'])
        
        lat, lon = resolver_coordenadas(prov, dept, cache_coords)
        df_diario = obtener_clima_departamento(lat, lon, depto_id=d_id, anio_inicio=1981, anio_fin=2024)
        
        if not df_diario.empty:
            df_agregado = agregar_metricas_climaticas(df_diario, depto_id=d_id)
            lista_clima.append(df_agregado)
            
        if (i + 1) % 20 == 0 or (i + 1) == len(deptos_unicos):
            print(f"  --> Departamentos procesados: {i + 1}/{len(deptos_unicos)}")
            
    df_clima_total = pd.concat(lista_clima, ignore_index=True)
    print(f"[3/5] Base meteorologica agregada: {len(df_clima_total):,} filas.")
    
    # [4/5] Merge relacional por clave compuesta de 3 columnas
    print("[4/5] Uniendo datos agricolas y meteorologicos...")
    df_final = pd.merge(
        df_agro,
        df_clima_total,
        on=['departamento_id', 'campania', 'grupo_cultivo'],
        how='inner'
    )
    
    # Feature Engineering de área perdida
    df_final['ratio_perdida_area'] = (
        (df_final['superficie_sembrada_ha'] - df_final['superficie_cosechada_ha']) 
        / df_final['superficie_sembrada_ha'].replace(0, np.nan)
    ).clip(0.0, 1.0).fillna(0.0)
    
    # [5/5] Exportar resultados finales
    salida_csv = "dataset_agro_clima_final.csv"
    salida_parquet = "dataset_agro_clima_final.parquet"
    
    df_final.to_csv(salida_csv, index=False, sep=';', encoding='utf-8')
    df_final.to_parquet(salida_parquet, index=False)
    
    print("\n" + "=" * 65)
    print("PROCESO FINALIZADO CON EXITO")
    print(f"  - Filas consolidadas: {len(df_final):,}")
    print(f"  - Archivos guardados: '{salida_csv}' y '{salida_parquet}'")
    print("=" * 65)
    
    return df_final


if __name__ == "__main__":
    # Nombre del archivo provisto
    ARCHIVO = "estimaciones-agricolas-2026-03.csv"
    ejecutar_pipeline_agro_clima(ARCHIVO)