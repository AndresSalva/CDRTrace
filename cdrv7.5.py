# -*- coding: utf-8 -*-
"""
Analiza y visualiza datos de Registros de Detalle de Llamadas (CDR).

Funcionalidades:
- Carga y limpia datos CDR y Puntos de Interés (POI).
- Calcula estadísticas de uso de teléfonos.
- Genera animaciones:
    1. Heatmap de uso de antenas con punto móvil y POIs.
    2. Ruta de un teléfono con segmentos coloreados por velocidad (flechas) y POIs.
"""

import math
import warnings
from pathlib import Path

import contextily as ctx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Configuraciones ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

# Rutas a los archivos de datos (ajusta si es necesario)
CDR_CSV_PATH = Path('cleaned_cdrs.csv')
POI_CSV_PATH = Path('puntos_pericia.csv')

# --- Funciones Auxiliares de Carga y Cálculo ---

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Carga y preprocesa los datos CDR desde un archivo CSV.

    Args:
        file_path: Ruta al archivo CSV de CDRs.

    Returns:
        DataFrame de pandas con los datos cargados y limpiados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si ocurre un error durante la carga o el preprocesamiento.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path.resolve()}")
    try:
        df = pd.read_csv(
            file_path,
            parse_dates=['datetime'],
            dtype={'origin': str, 'destination': str, 'cell_site': str}
        )
        # Asegurar tipos correctos y manejar errores de conversión numérica
        df[['origin', 'destination']] = df[['origin', 'destination']].astype(str)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

        # Limpieza básica: eliminar filas sin coordenadas válidas
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)] # Longitud también

        print(f"Datos CDR cargados y limpiados desde {file_path.name}.")
        return df
    except Exception as e:
        print(f"Error cargando {file_path.name}: {e}")
        raise

def generate_circle(lon_center: float, lat_center: float, radius_km: float, n_points: int = 50) -> np.ndarray:
    """
    Genera los puntos (lon, lat) de un círculo en la superficie terrestre.

    Args:
        lon_center: Longitud del centro del círculo.
        lat_center: Latitud del centro del círculo.
        radius_km: Radio del círculo en kilómetros.
        n_points: Número de puntos para aproximar el círculo.

    Returns:
        Array de numpy de forma (n_points, 2) con [longitud, latitud] para cada punto.
    """
    R = 6371.0  # Radio medio de la Tierra en km
    lat_rad = np.radians(lat_center)
    lon_rad = np.radians(lon_center)
    angular_dist = radius_km / R
    lats, lons = [], []

    for angle in np.linspace(0, 2 * np.pi, n_points):
        lat_p_rad = np.arcsin(np.sin(lat_rad) * np.cos(angular_dist) +
                              np.cos(lat_rad) * np.sin(angular_dist) * np.cos(angle))
        cos_lat_p = np.cos(lat_p_rad)

        # Manejar división por cero cerca de los polos si es necesario (aunque poco probable con datos CDR)
        if np.isclose(cos_lat_p, 0):
             # Si estamos exactamente en un polo, el cambio de longitud no está bien definido de esta forma,
             # pero la longitud del centro es una aproximación razonable.
            lon_p_rad = lon_rad
        else:
            # Fórmula más estable numéricamente para el cambio de longitud
            sin_dlon_num = np.sin(angle) * np.sin(angular_dist) * np.cos(lat_rad)
            cos_dlon_num = np.cos(angular_dist) - np.sin(lat_rad) * np.sin(lat_p_rad)
            dlon_rad = np.arctan2(sin_dlon_num, cos_dlon_num)
            lon_p_rad = (lon_rad + dlon_rad + np.pi) % (2 * np.pi) - np.pi # Asegurar rango [-pi, pi]

        lats.append(np.degrees(lat_p_rad))
        lons.append(np.degrees(lon_p_rad))

    return np.vstack([lons, lats]).T

def list_phone_stats(df: pd.DataFrame, n: int = 10):
    """
    Muestra estadísticas básicas de los n números de teléfono más frecuentes.

    Args:
        df: DataFrame de CDRs.
        n: Número de teléfonos top a mostrar.
    """
    all_numbers = pd.concat([df['origin'].astype(str), df['destination'].astype(str)])
    # Filtro más robusto para números de teléfono (opcionalmente empieza con +)
    valid_phones_mask = all_numbers.str.match(r'^\+?\d{5,}$', na=False) # Mínimo 5 dígitos
    counts = all_numbers[valid_phones_mask].value_counts()

    if counts.empty:
        print("No se encontraron números de teléfono válidos para generar estadísticas.")
        return

    top = counts.head(n)
    stats = []
    print(f"\n--- Top {min(n, len(top))} Teléfonos Más Frecuentes ---")
    for phone in top.index:
        # Buscar el teléfono tanto en origen como en destino
        df_p = df[(df['origin'] == phone) | (df['destination'] == phone)]
        if not df_p.empty:
            stats.append({
                'phone': phone,
                'count': top[phone],
                'first_record': df_p['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'last_record': df_p['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            # Esto no debería ocurrir si el teléfono está en `counts`, pero por si acaso
            stats.append({
                'phone': phone,
                'count': top[phone],
                'first_record': 'N/A',
                'last_record': 'N/A'
            })

    if stats:
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
    else:
        print("No se pudieron generar estadísticas (inesperado).")
    print("------------------------------------------")

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calcula la distancia geodésica (Haversine) entre dos puntos (lon, lat).

    Args:
        lon1, lat1: Longitud y latitud del primer punto (grados).
        lon2, lat2: Longitud y latitud del segundo punto (grados).

    Returns:
        Distancia en kilómetros.
    """
    # Convertir grados a radianes
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Fórmula Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0  # Radio de la Tierra en kilómetros
    distance = R * c
    return distance

def load_puntos_pericia(file_path: Path) -> pd.DataFrame | None:
    """
    Carga los Puntos de Interés (POI) desde un archivo CSV.

    Args:
        file_path: Ruta al archivo CSV de POIs.

    Returns:
        DataFrame de pandas con los POIs o None si hay errores o no se encuentra.
    """
    if not file_path.exists():
        print(f"Advertencia: Archivo de Puntos de Pericia no encontrado en {file_path.resolve()}. "
              "No se mostrarán POIs.")
        return None
    try:
        poi_df = pd.read_csv(file_path, dtype={'name': str})

        # Validar columnas necesarias
        required_cols = ['name', 'latitude', 'longitude']
        if not all(col in poi_df.columns for col in required_cols):
            print("Error: El archivo de Puntos de Pericia debe contener las columnas "
                  f"{required_cols}.")
            return None

        # Convertir coordenadas a numérico, manejando errores
        poi_df['latitude'] = pd.to_numeric(poi_df['latitude'], errors='coerce')
        poi_df['longitude'] = pd.to_numeric(poi_df['longitude'], errors='coerce')

        # Eliminar filas con coordenadas inválidas o faltantes
        poi_df.dropna(subset=['latitude', 'longitude'], inplace=True)
        poi_df = poi_df[poi_df['latitude'].between(-90, 90) & poi_df['longitude'].between(-180, 180)]

        if poi_df.empty:
            print("Advertencia: No se encontraron puntos de pericia válidos en el archivo.")
            return None

        print(f"Puntos de Pericia cargados desde {file_path.name}.")
        return poi_df
    except Exception as e:
        print(f"Error cargando Puntos de Pericia desde {file_path.name}: {e}")
        return None

def calculate_speeds(df_phone: pd.DataFrame, min_time_seconds_threshold: float = 10) -> tuple[list, list]:
    """
    Calcula la velocidad entre puntos consecutivos de un DataFrame filtrado.

    Args:
        df_phone: DataFrame ordenado por tiempo para un solo teléfono.
        min_time_seconds_threshold: Tiempo mínimo (segundos) entre registros
                                    para calcular una velocidad válida.

    Returns:
        Tupla conteniendo:
        - Lista de timestamps (correspondiente al punto final de cada segmento).
        - Lista de velocidades calculadas en km/h (NaN si el tiempo es menor al umbral).
    """
    times = []
    speeds = []
    if len(df_phone) < 2:
        return times, speeds # No se puede calcular velocidad con menos de 2 puntos

    filtered_count = 0
    # Iterar desde el segundo registro
    for i in range(1, len(df_phone)):
        prev_row = df_phone.iloc[i - 1]
        curr_row = df_phone.iloc[i]

        # Calcular diferencia de tiempo
        time_diff = curr_row['datetime'] - prev_row['datetime']
        time_seconds = time_diff.total_seconds()

        # Añadir timestamp actual a la lista de tiempos
        times.append(curr_row['datetime'])

        # Calcular velocidad si el tiempo es suficiente y las coordenadas son válidas
        speed_kph = np.nan # Valor por defecto
        if time_seconds >= min_time_seconds_threshold:
             # Asegurarse de que las coordenadas son válidas antes de Haversine
            if pd.notna(prev_row['longitude']) and pd.notna(prev_row['latitude']) and \
               pd.notna(curr_row['longitude']) and pd.notna(curr_row['latitude']):
                distance_km = haversine(
                    prev_row['longitude'], prev_row['latitude'],
                    curr_row['longitude'], curr_row['latitude']
                )
                # Evitar división por cero si time_seconds fuera exactamente 0 (improbable con el umbral)
                if time_seconds > 0:
                    speed_kph = (distance_km / time_seconds) * 3600 # km/h
                else:
                    speed_kph = 0.0 # Si la distancia es 0 y el tiempo es 0
            # else: speed_kph remains np.nan
        else:
            filtered_count += 1 # Contar cuántos intervalos fueron demasiado cortos

        speeds.append(speed_kph)

    if filtered_count > 0:
        print(f"Nota: {filtered_count} segmentos de velocidad ignorados "
              f"(intervalo < {min_time_seconds_threshold}s).")

    # Asegurar que speeds tenga longitud len(df_phone) - 1
    # Esto ya debería ser así por cómo está construido el bucle
    assert len(speeds) == len(df_phone) - 1, "Error interno: longitud de velocidades incorrecta."

    return times, speeds


# --- Funciones Auxiliares Comunes para Animaciones ---

def _add_basemap_to_ax(ax: plt.Axes):
    """Añade un mapa base de OpenStreetMap al eje dado."""
    print("Añadiendo mapa base...")
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:4326", # Coordenadas geográficas estándar
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution_size=7,
            zoom='auto' # Deja que contextily elija el zoom
        )
    except Exception as e:
        print(f"Advertencia: No se pudo añadir el mapa base. Error: {e}.")

def _plot_pois_on_ax(ax: plt.Axes, poi_df: pd.DataFrame | None, x_margin: float, y_margin: float,
                     poi_marker: str, poi_size: int, poi_color: str):
    """Dibuja los Puntos de Interés (POIs) en el eje."""
    if poi_df is not None and not poi_df.empty:
        print("Añadiendo Puntos de Pericia (POIs)...")
        ax.scatter(
            poi_df['longitude'], poi_df['latitude'],
            marker=poi_marker, s=poi_size, color=poi_color,
            edgecolor='black', label='Punto de Pericia (POI)',
            zorder=10 # Asegurar que los POIs estén encima de otros elementos
        )
        # Añadir etiquetas a los POIs
        for i, point in poi_df.iterrows():
            ax.text(
                point['longitude'] + x_margin * 0.02, # Pequeño offset para legibilidad
                point['latitude'] + y_margin * 0.02,
                point['name'],
                fontsize=9, color=poi_color, weight='bold',
                # Caja de texto semitransparente para mejorar legibilidad sobre el mapa
                bbox=dict(facecolor='white', alpha=0.6, pad=0.2, edgecolor='none'),
                zorder=11 # Etiquetas encima de los marcadores POI
            )

def _save_or_show_animation(fig: plt.Figure, ani: FuncAnimation, save_path: str | None,
                           interval: int, plot_type_name: str):
    """Guarda la animación en un archivo o la muestra en una ventana."""
    if save_path:
        output_path = Path(save_path)
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Calcular FPS (asegurando al menos 1 FPS)
            fps = max(1, 1000 // interval if interval > 0 else 1)
            print(f"Guardando animación {plot_type_name} en: {output_path} (FPS: {fps})...")
            # Usar dpi para mejor calidad si se guarda
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Animación {plot_type_name} guardada.")
        except Exception as e:
            print(f"\nError guardando animación {plot_type_name}: {e}\n"
                  "Mostrando en ventana como alternativa.")
            plt.tight_layout()
            plt.show()
        finally:
            plt.close(fig) # Cerrar la figura después de intentar guardar o mostrar
    else:
        print(f"Mostrando animación {plot_type_name}...")
        plt.tight_layout()
        plt.show()
        plt.close(fig) # Cerrar la figura después de mostrarla


# --- Función 1: Animación Heatmap ---

def animate_phone_heatmap(
    df: pd.DataFrame,
    poi_df: pd.DataFrame | None,
    phone_number: str,
    start_time: str,
    end_time: str,
    heatmap_cmap: str = 'viridis',
    circle_radius_km: float = 2.0,
    heatmap_base_size: float = 50,
    heatmap_scale_factor: float = 15,
    interval: int = 500,
    save_path: str | None = None,
    poi_marker: str = '*',
    poi_size: int = 150,
    poi_color: str = 'fuchsia'
):
    """
    Genera una animación mostrando un heatmap de uso de antenas, círculos de
    cobertura estimada, un punto móvil para la posición del teléfono y POIs.
    """
    print("\n--- Iniciando Animación: Heatmap, Punto Móvil y POIs ---")

    # 1. Filtrar datos para el teléfono y rango de tiempo
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    mask_phone_time = (
        ((df['origin'] == phone_number) | (df['destination'] == phone_number)) &
        (df['datetime'] >= start) &
        (df['datetime'] <= end) &
        df['latitude'].notna() & df['longitude'].notna()
    )
    df_phone = df.loc[mask_phone_time].sort_values('datetime').reset_index(drop=True)

    if df_phone.empty:
        print(f"No hay registros válidos para la animación heatmap de {phone_number} "
              f"entre {start_time} y {end_time}.")
        return

    # 2. Calcular uso de antenas
    antenna_usage_counts = df_phone.groupby(
        ['latitude', 'longitude', 'cell_site'], # Agrupar por coordenadas y ID de celda
        observed=True # Recomendado para tipos categóricos si los hubiera
    ).size().reset_index(name='usage_count')

    if antenna_usage_counts.empty:
        print(f"No se encontraron ubicaciones de antena para el heatmap de {phone_number}.")
        return

    # 3. Configurar el gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    title = (f'Actividad {phone_number} ({start_time} - {end_time})\n'
             f'Densidad de Antenas (Color/Tamaño) | Cobertura Estimada ({circle_radius_km}km) | POIs')
    ax.set_title(title)

    # 4. Calcular límites del mapa
    lon_min_ant = antenna_usage_counts['longitude'].min(); lon_max_ant = antenna_usage_counts['longitude'].max()
    lat_min_ant = antenna_usage_counts['latitude'].min(); lat_max_ant = antenna_usage_counts['latitude'].max()

    # Considerar también la extensión de los círculos y POIs para los límites
    all_circle_points = []
    for _, antenna in antenna_usage_counts.iterrows():
        try:
            all_circle_points.append(generate_circle(antenna['longitude'], antenna['latitude'], circle_radius_km))
        except Exception: # Ignorar si falla la generación de algún círculo
             print(f"Advertencia: No se pudo generar círculo para antena en ({antenna['longitude']}, {antenna['latitude']})")
             pass

    lon_min, lon_max = lon_min_ant, lon_max_ant
    lat_min, lat_max = lat_min_ant, lat_max_ant

    if all_circle_points:
        circle_pts_all = np.vstack(all_circle_points)
        lon_min = min(lon_min, circle_pts_all[:, 0].min()); lon_max = max(lon_max, circle_pts_all[:, 0].max())
        lat_min = min(lat_min, circle_pts_all[:, 1].min()); lat_max = max(lat_max, circle_pts_all[:, 1].max())

    if poi_df is not None and not poi_df.empty:
        lon_min = min(lon_min, poi_df['longitude'].min()); lon_max = max(lon_max, poi_df['longitude'].max())
        lat_min = min(lat_min, poi_df['latitude'].min()); lat_max = max(lat_max, poi_df['latitude'].max())

    # Añadir margen a los límites
    lon_range = max(lon_max - lon_min, 0.01); lat_range = max(lat_max - lat_min, 0.01)
    x_margin = max(lon_range * 0.1, 0.02); y_margin = max(lat_range * 0.1, 0.02)
    ax.set_xlim(lon_min - x_margin, lon_max + x_margin)
    ax.set_ylim(lat_min - y_margin, lat_max + y_margin)

    # 5. Añadir mapa base
    _add_basemap_to_ax(ax)

    # 6. Dibujar Heatmap (Scatter) y Círculos de Cobertura
    print("Generando mapa de calor y círculos de cobertura...")
    norm = Normalize(vmin=antenna_usage_counts['usage_count'].min(), vmax=antenna_usage_counts['usage_count'].max())
    sizes = antenna_usage_counts['usage_count'] * heatmap_scale_factor + heatmap_base_size
    heatmap_scatter = ax.scatter(
        antenna_usage_counts['longitude'], antenna_usage_counts['latitude'],
        c=antenna_usage_counts['usage_count'], s=sizes, cmap=heatmap_cmap,
        norm=norm, alpha=0.75,
        label='Antena Usada (Color/Tamaño = # Conexiones)',
        zorder=3 # Heatmap encima del mapa base pero debajo de POIs/punto móvil
    )
    cbar = fig.colorbar(heatmap_scatter, ax=ax, shrink=0.7)
    cbar.set_label('Número de Conexiones')

    # Dibujar círculos
    first_circle = True
    for idx, antenna in antenna_usage_counts.iterrows():
        try:
            circle_pts = generate_circle(antenna['longitude'], antenna['latitude'], circle_radius_km)
            label = f'{circle_radius_km}km Cobertura Estimada' if first_circle else None
            ax.plot(circle_pts[:, 0], circle_pts[:, 1], linestyle='--', linewidth=1.0,
                    color='gray', alpha=0.6, label=label, zorder=4) # Círculos encima del heatmap
            first_circle = False
        except Exception:
            # Ya se advirtió antes si la generación falló
            pass

    # 7. Dibujar POIs
    _plot_pois_on_ax(ax, poi_df, x_margin, y_margin, poi_marker, poi_size, poi_color)

    # 8. Configurar elementos de la animación (punto móvil y texto de tiempo)
    mobile_point = ax.scatter(
        [], [], s=100, color='red', edgecolor='white',
        label=f'Posición Actual {phone_number}', zorder=6 # Punto móvil encima de casi todo
    )
    time_text = ax.text(
        0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
        backgroundcolor='white', zorder=12 # Texto encima de todo
    )

    # 9. Crear leyenda unificada (evitando duplicados)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Diccionario para eliminar duplicados
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

    # 10. Definir función de actualización para la animación
    def update(frame):
        """Actualiza la posición del punto móvil y el texto del tiempo."""
        if frame < len(df_phone):
            row = df_phone.iloc[frame]
            # Asegurarse de que hay coordenadas válidas para este frame
            if pd.notna(row['longitude']) and pd.notna(row['latitude']):
                mobile_point.set_offsets([[row['longitude'], row['latitude']]])
                time_text.set_text(row['datetime'].strftime('%Y-%m-%d %H:%M:%S'))
            else:
                # Opcional: ocultar el punto si no hay coordenadas?
                # mobile_point.set_offsets([[], []]) # O dejarlo donde estaba
                time_text.set_text(f"{row['datetime'].strftime('%Y-%m-%d %H:%M:%S')} (Sin Coordenadas)")
        # Devolver los elementos que cambian (blit=True necesita esto)
        return mobile_point, time_text

    # 11. Crear y manejar la animación
    print("Creando animación heatmap/punto/POIs...")
    num_frames = len(df_phone)
    if num_frames == 0:
        print("No hay frames para animar el heatmap.")
        plt.close(fig)
        return

    ani = FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval, blit=True, # blit=True para rendimiento
        repeat=False
    )

    # 12. Guardar o Mostrar
    _save_or_show_animation(fig, ani, save_path, interval, "Heatmap")


# --- Función 2: Animación Ruta Coloreada con FLECHAS ---

def animate_phone_path_with_speed_color(
    df_phone: pd.DataFrame,
    poi_df: pd.DataFrame | None,
    speeds: list,
    phone_number: str,
    start_time: str,
    end_time: str,
    cmap_realistic: str = 'viridis',
    speed_threshold_low: float = 150.0,
    speed_threshold_high: float = 800.0,
    color_nan: str = 'lightgray',
    color_unrealistic: str = 'red',
    color_high: str = 'orange',
    path_linewidth: float = 1.5,
    arrow_style: str = '->', # Estilo de flecha de matplotlib
    interval: int = 100,
    save_path: str | None = None,
    poi_marker: str = '*',
    poi_size: int = 150,
    poi_color: str = 'fuchsia'
):
    """
    Genera una animación mostrando la ruta de un teléfono con flechas
    coloreadas según la velocidad calculada, junto con POIs.
    """
    print("\n--- Iniciando Animación: Ruta con Flechas Coloreadas y POIs ---")

    # 1. Validar datos de entrada
    if df_phone.empty or len(df_phone) < 2:
        print("No hay suficientes datos (mínimo 2 puntos) para animar la ruta.")
        return
    if len(speeds) != len(df_phone) - 1:
        print(f"Error crítico: Discrepancia entre número de puntos ({len(df_phone)}) "
              f"y número de velocidades ({len(speeds)}). No se puede animar.")
        return

    # 2. Configurar colormap y normalización para la velocidad
    cmap = plt.get_cmap(cmap_realistic)
    # Normalizar solo hasta el umbral bajo para el colormap principal
    norm = Normalize(vmin=0, vmax=speed_threshold_low)
    scalar_mappable = ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([]) # Necesario para que funcione el colorbar

    # 3. Configurar el gráfico
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    title = (f'Ruta {phone_number} ({start_time} - {end_time})\n'
             f'Flecha Color = Velocidad Estimada | POIs')
    ax.set_title(title)

    # 4. Calcular límites del mapa (basado en la ruta y POIs)
    lon_coords = df_phone['longitude']; lat_coords = df_phone['latitude']
    lon_min, lon_max = lon_coords.min(), lon_coords.max()
    lat_min, lat_max = lat_coords.min(), lat_coords.max()

    if poi_df is not None and not poi_df.empty:
        lon_min = min(lon_min, poi_df['longitude'].min()); lon_max = max(lon_max, poi_df['longitude'].max())
        lat_min = min(lat_min, poi_df['latitude'].min()); lat_max = max(lat_max, poi_df['latitude'].max())

    lon_range = max(lon_max - lon_min, 0.01); lat_range = max(lat_max - lat_min, 0.01)
    x_margin = max(lon_range * 0.1, 0.02); y_margin = max(lat_range * 0.1, 0.02)
    ax.set_xlim(lon_min - x_margin, lon_max + x_margin)
    ax.set_ylim(lat_min - y_margin, lat_max + y_margin)

    # 5. Añadir mapa base
    _add_basemap_to_ax(ax)

    # 6. Dibujar POIs
    _plot_pois_on_ax(ax, poi_df, x_margin, y_margin, poi_marker, poi_size, poi_color)

    # 7. Configurar elementos de la animación (texto de tiempo)
    # Las flechas se añadirán dinámicamente en update
    time_text = ax.text(
        0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
        backgroundcolor='white', zorder=12 # Texto encima de todo
    )

    # Lista para almacenar los objetos de flecha (Annotations) creados
    # No se usa directamente para blit=False, pero puede ser útil para debug
    # arrow_annotations = []

    # 8. Definir función de actualización para la animación (usando annotate)
    def update(frame):
        """Dibuja el segmento de ruta (flecha) correspondiente al frame actual."""
        current_time = df_phone.iloc[frame]['datetime']
        time_text.set_text(current_time.strftime('%Y-%m-%d %H:%M:%S'))

        # Las flechas representan el movimiento *hacia* el punto 'frame'
        # desde el punto 'frame-1'. Necesitamos frame > 0.
        new_arrows_in_frame = []
        if frame > 0:
            prev_row = df_phone.iloc[frame - 1]
            curr_row = df_phone.iloc[frame]
            speed = speeds[frame - 1] # speed[i] es la velocidad para ir de punto i a i+1

            # Determinar color del segmento/flecha
            segment_color = color_nan # Color por defecto si la velocidad es NaN
            if pd.notna(speed):
                if speed <= speed_threshold_low:
                    segment_color = cmap(norm(speed)) # Color del colormap
                elif speed <= speed_threshold_high:
                    segment_color = color_high # Color para velocidad alta
                else:
                    segment_color = color_unrealistic # Color para velocidad muy alta/irreal

            # Dibujar flecha solo si los puntos son distintos y válidos
            if pd.notna(prev_row['longitude']) and pd.notna(prev_row['latitude']) and \
               pd.notna(curr_row['longitude']) and pd.notna(curr_row['latitude']) and \
               not (np.isclose(prev_row['longitude'], curr_row['longitude']) and \
                    np.isclose(prev_row['latitude'], curr_row['latitude'])):

                arrow = ax.annotate(
                    "", # Sin texto en la anotación
                    xy=(curr_row['longitude'], curr_row['latitude']), # Punta de la flecha
                    xytext=(prev_row['longitude'], prev_row['latitude']), # Cola de la flecha
                    arrowprops=dict(
                        arrowstyle=arrow_style,
                        color=segment_color,
                        lw=path_linewidth,
                        alpha=0.9,
                        shrinkA=0, # No encoger desde la punta
                        shrinkB=0  # No encoger desde la cola
                    ),
                    zorder=5 # Flechas encima del mapa base, debajo de POIs/texto
                )
                new_arrows_in_frame.append(arrow)
                # arrow_annotations.append(arrow) # Guardar referencia si fuera necesario

        # Con blit=False, solo necesitamos devolver los artistas *nuevos* o *modificados*
        # en este frame. El texto se modifica siempre. Las flechas son nuevas cada vez.
        # PERO: `annotate` no funciona bien con blit=True. Forzamos blit=False.
        # Con blit=False, la función update puede simplemente dibujar y no necesita
        # devolver los artistas necesariamente, pero devolverlos no hace daño.
        # Devolver solo el texto es suficiente aquí ya que las flechas se añaden al eje.
        return [time_text] # Devolver al menos un artista para que la animación sepa que algo cambió

    # 9. Crear y manejar la animación
    print("Creando animación de ruta con flechas coloreadas y POIs...")
    num_frames = len(df_phone)
    if num_frames < 2: # Ya se verificó antes, pero por si acaso
        print("No hay suficientes frames para animar la ruta con flechas.")
        plt.close(fig)
        return

    # IMPORTANTE: blit=False es necesario para que ax.annotate funcione correctamente
    # en cada frame de la animación. blit=True tiende a no redibujar bien las anotaciones.
    ani = FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval, blit=False, # ¡¡¡ blit DEBE ser False para annotate !!!
        repeat=False
    )

    # 10. Añadir Barra de Color y Leyenda
    # Crear un eje separado para la barra de color
    # Posición: [left, bottom, width, height] en coordenadas de figura (0 a 1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label(
        f'Velocidad Normal ({cmap_realistic}, 0-{int(speed_threshold_low)} km/h)',
        rotation=270, labelpad=15
    )

    # Crear leyenda personalizada para los colores de velocidad y POIs
    # Usar Line2D para representar los colores en la leyenda
    legend_handles = [
        plt.Line2D([0], [0], color=scalar_mappable.cmap(scalar_mappable.norm(speed_threshold_low * 0.5)), lw=path_linewidth + 1, label=f'Normal (0-{int(speed_threshold_low)} km/h)'),
        plt.Line2D([0], [0], color=color_high, lw=path_linewidth + 1, label=f'Alta ({int(speed_threshold_low)}-{int(speed_threshold_high)} km/h)'),
        plt.Line2D([0], [0], color=color_unrealistic, lw=path_linewidth + 1, label=f'Muy Alta (> {int(speed_threshold_high)} km/h)'),
        plt.Line2D([0], [0], color=color_nan, lw=path_linewidth + 1, label='Inválida/Rápida (< umbral t)')
    ]
    if poi_df is not None and not poi_df.empty:
        # Usar un marcador scatter para representar POIs en la leyenda
        legend_handles.append(
            plt.scatter([], [], marker=poi_marker, s=poi_size*0.5, # Tamaño reducido para leyenda
                        color=poi_color, edgecolor='black', label='Punto de Pericia (POI)')
         )
         # Alternativa con Line2D si scatter da problemas:
         # legend_handles.append(plt.Line2D([0], [0], marker=poi_marker, color='w', # Sin línea
         #                           markerfacecolor=poi_color, markeredgecolor='black',
         #                           markersize=math.sqrt(poi_size*0.8), # Ajustar tamaño
         #                           label='Punto de Pericia (POI)'))

    # Añadir la leyenda principal al gráfico
    ax.legend(handles=legend_handles, loc='upper right', title="Leyenda Velocidad/POI", fontsize=8)

    # 11. Guardar o Mostrar
    _save_or_show_animation(fig, ani, save_path, interval, "Ruta con Flechas")


# --- Bloque Principal de Ejecución ---
if __name__ == '__main__':
    try:
        # --- Directorio de Salida para Animaciones ---
        # Define el nombre de la carpeta donde se guardarán las animaciones
        OUTPUT_ANIMATION_FOLDER = Path("animaciones_cdr")
        # Crea la carpeta si no existe (aunque la función _save_or_show_animation también lo hará)
        OUTPUT_ANIMATION_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"Las animaciones se guardarán en la carpeta: '{OUTPUT_ANIMATION_FOLDER.resolve()}'")
        # ---------------------------------------------

        # Cargar datos
        df_cdr = load_data(CDR_CSV_PATH)
        df_poi = load_puntos_pericia(POI_CSV_PATH) # Puede ser None

        # Mostrar estadísticas iniciales
        print("\n--- Estadísticas Iniciales de Teléfonos (Top 10) ---")
        list_phone_stats(df_cdr, n=10)

        # --- Parámetros para el Análisis Específico ---
        # Cambia estos valores según el teléfono y periodo que quieras analizar
        telefono_sospechoso = "59171234567" # Ejemplo
        inicio_tiempo = '2023-10-07 00:00:00'
        fin_tiempo    = '2023-10-07 23:59:59'

        # Crear nombres de archivo base dinámicos (opcional pero recomendado)
        # Reemplaza caracteres no válidos para nombres de archivo
        start_time_safe = inicio_tiempo.replace(":", "-").replace(" ", "_")
        end_time_safe = fin_tiempo.replace(":", "-").replace(" ", "_")
        file_base_name = f"{telefono_sospechoso}_{start_time_safe}_to_{end_time_safe}"

        # Filtrar datos para el teléfono y periodo de interés
        print(f"\nFiltrando datos para {telefono_sospechoso} entre {inicio_tiempo} y {fin_tiempo}...")
        mask_phone_time = (
            ((df_cdr['origin'] == telefono_sospechoso) | (df_cdr['destination'] == telefono_sospechoso)) &
            (df_cdr['datetime'] >= pd.to_datetime(inicio_tiempo)) &
            (df_cdr['datetime'] <= pd.to_datetime(fin_tiempo)) &
            df_cdr['latitude'].notna() & df_cdr['longitude'].notna()
        )
        df_phone_filtered = df_cdr.loc[mask_phone_time].sort_values('datetime').reset_index(drop=True)

        if df_phone_filtered.empty:
             print(f"No se encontraron registros para {telefono_sospechoso} en el periodo especificado.")
        else:
            print(f"Se encontraron {len(df_phone_filtered)} registros para el análisis.")

            # Calcular velocidades
            speeds_kph = []
            if len(df_phone_filtered) >= 2:
                print("Calculando velocidades...")
                # Umbral de tiempo mínimo (en segundos) para considerar válida una velocidad
                tiempo_minimo_para_velocidad = 10
                _, speeds_kph = calculate_speeds(
                    df_phone_filtered,
                    min_time_seconds_threshold=tiempo_minimo_para_velocidad
                )
                # `speeds_kph` tendrá longitud len(df_phone_filtered) - 1
                print(f"Se calcularon {len(speeds_kph)} segmentos de velocidad.")
            else:
                print("No hay suficientes puntos (se necesitan >= 2) para calcular velocidades.")

            # --- Ejecutar Animación 1: Heatmap + Punto Móvil + POIs ---
            print("\n*** Recordatorio: Círculos = Cobertura Estimada, NO Triangulación precisa. ***")
            # Parámetros para la animación heatmap (puedes ajustarlos)
            heatmap_params = {
                'heatmap_cmap': 'inferno', # Colormap para el heatmap
                'circle_radius_km': 1.5,   # Radio estimado de cobertura de antena
                'heatmap_base_size': 40,   # Tamaño base de los puntos del heatmap
                'heatmap_scale_factor': 20,# Factor de escala para el tamaño (según uso)
                'interval': 1000,           # Milisegundos entre frames
                # MODIFICADO: Especificar la ruta completa incluyendo la carpeta
                'save_path': OUTPUT_ANIMATION_FOLDER / f"heatmap_{file_base_name}.gif",
                'poi_marker': '^',         # Marcador para POIs
                'poi_color': 'lime',       # Color para POIs
                'poi_size': 120            # Tamaño para POIs
            }
            # Llamada a la función de animación (sin cambios)
            animate_phone_heatmap(
                df=df_cdr, # Pasar el DF completo para calcular densidad de todas las antenas usadas
                poi_df=df_poi,
                phone_number=telefono_sospechoso,
                start_time=inicio_tiempo,
                end_time=fin_tiempo,
                **heatmap_params
            )

            # --- Ejecutar Animación 2: Ruta con Flechas Coloreadas + POIs ---
            # Verificar si tenemos velocidades calculadas correctamente
            if speeds_kph and len(speeds_kph) == len(df_phone_filtered) - 1:
                print("\nPreparando animación de ruta con flechas...")
                # Parámetros para la animación de ruta (puedes ajustarlos)
                path_params = {
                    'cmap_realistic': 'coolwarm', # Colormap para velocidades normales
                    'speed_threshold_low': 150,  # Límite superior velocidad normal (km/h)
                    'speed_threshold_high': 800, # Límite superior velocidad alta (km/h)
                    'color_nan': 'silver',       # Color para velocidad inválida (intervalo corto)
                    'color_high': 'orange',      # Color para velocidad alta
                    'color_unrealistic': 'magenta',# Color para velocidad muy alta/irreal
                    'path_linewidth': 1.5,       # Grosor de las flechas
                    'arrow_style': '-|>',        # Estilo de flecha ('->', 'simple', 'fancy', '-|>', etc.)
                    'interval': 1000,             # Milisegundos entre frames
                    # MODIFICADO: Especificar la ruta completa incluyendo la carpeta
                    'save_path': OUTPUT_ANIMATION_FOLDER / f"path_arrows_{file_base_name}.gif",
                    'poi_marker': '^',           # Marcador POI (igual que heatmap)
                    'poi_color': 'lime',         # Color POI (igual que heatmap)
                    'poi_size': 120              # Tamaño POI (igual que heatmap)
                }
                # Llamada a la función de animación (sin cambios)
                animate_phone_path_with_speed_color(
                    df_phone=df_phone_filtered, # Usar el DF ya filtrado para la ruta
                    poi_df=df_poi,
                    speeds=speeds_kph,
                    phone_number=telefono_sospechoso,
                    start_time=inicio_tiempo,
                    end_time=fin_tiempo,
                    **path_params
                )
            elif len(df_phone_filtered) >= 2:
                 # Si hay puntos pero el cálculo de velocidad falló o dio longitud incorrecta
                 print(f"\nAdvertencia: Omitiendo animación de ruta coloreada. "
                       f"No se pudieron calcular correctamente las velocidades. "
                       f"(Puntos: {len(df_phone_filtered)}, Velocidades esperadas: {len(df_phone_filtered)-1}, "
                       f"Velocidades calculadas: {len(speeds_kph)})")
            else:
                 # Si no había suficientes puntos desde el inicio
                 print("\nOmitiendo animación de ruta coloreada: No hay suficientes puntos de datos (se necesitan >= 2).")

    except FileNotFoundError as e:
        print(f"Error: Archivo de datos no encontrado. {e}")
    except ValueError as e:
        print(f"Error en parámetros o conversión de datos: {e}")
    except Exception as e:
        print(f"Error inesperado durante la ejecución: {e}")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------")

    print("\nAnálisis completado.")