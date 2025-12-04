import json
import time
import concurrent.futures
from functools import lru_cache
from typing import List, Dict, Tuple, Generator
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass
from enum import Enum

"""
Módulo de optimización de producción.
Este módulo implementa el núcleo del sistema de optimización de producción
utilizando programación dinámica y técnicas avanzadas de optimización.

ALGORITMO IMPLEMENTADO:
• Problema de la mochila 0/1 con dos restricciones (tiempo y material)
• Programación dinámica recursiva con memoización (@lru_cache)
• Cache persistente en disco (JSON)

CARACTERÍSTICAS AVANZADAS:
• Paralelismo con ThreadPoolExecutor
• Procesamiento por lotes (batching)
• Sistema de métricas y profiling
• Cola de prioridad para pedidos

CLASES PRINCIPALES:
    OptimizadorProduccion - Clase principal del optimizador
    Producto - Representación de productos (dataclass)
    GrafoDependencias - Manejo de dependencias entre productos
    ColaPrioridadPedidos - Sistema de priorización
"""
class EstadoProduccion(Enum):
    """
        Estados posibles de un proceso de producción u optimización.
        Se podría usar para etiquetar el estado de una línea de producción
        o de una optimización larga si se quisiera extender el proyecto.
        Valores:
            PENDIENTE: Todavía no se procesó.
            OPTIMIZANDO: Se está ejecutando el cálculo.
            COMPLETADO: Terminó correctamente.
            ERROR: Hubo algún problema durante la ejecución.
        """
    PENDIENTE = "pendiente"
    OPTIMIZANDO = "optimizando"
    COMPLETADO = "completado"
    ERROR = "error"
@dataclass
class Producto:
    """
    Representa un producto fabricable con sus recursos y ganancia.
    Atributos:
        id (str): Identificador único del producto.
        nombre (str): Nombre legible del producto.
        ganancia (int): Ganancia unitaria asociada al producto.
        tiempo (int): Tiempo requerido para fabricar una unidad.
        material (int): Material requerido para fabricar una unidad.
    Esta clase se usa como base para el algoritmo de optimización.
    """
    id: str
    nombre: str
    ganancia: int
    tiempo: int
    material: int

    def __hash__(self):
        """
               Devuelve un hash del producto.
               Se basa en los atributos clave del producto para poder usarlo
               en estructuras hash (por ejemplo, al generar claves de cache).
               Returns:
                   int: Valor hash del producto.
               """
        return hash((self.id, self.ganancia, self.tiempo, self.material))

class GrafoDependencias:
    """
        Grafo dirigido muy simple para modelar dependencias entre productos.
        La idea es poder representar reglas como:
        "para fabricar X, antes necesito fabricar Y".
        En este trabajo lo dejo preparado para poder extenderlo,
        e incluyo un método de ordenamiento topológico básico.
        """
    def __init__(self):
        """Inicializa la estructura de adyacencia del grafo."""
        self.adjacencia = defaultdict(list)

    def agregar_dependencia(self, producto: str, depende_de: str):
        """
                Agrega una relación de dependencia entre dos productos.
                Args:
                    producto (str): ID del producto que depende de otro.
                    depende_de (str): ID del producto del que depende.
                """
        self.adjacencia[producto].append(depende_de)

    def obtener_orden_produccion(self) -> List[str]:
        """
                Calcula un orden de producción válido según las dependencias.
                Utiliza un ordenamiento topológico muy simple:
                cuenta grados de entrada y va agregando nodos sin dependencias.
                Returns:
                    List[str]: Lista de IDs de productos en un orden posible de producción.
                """
        grados = defaultdict(int)
        for producto, dependencias in self.adjacencia.items():
            for dep in dependencias:
                grados[dep] += 1
        cola = deque([p for p in self.adjacencia if grados[p] == 0])
        orden = []
        while cola:
            actual = cola.popleft()
            orden.append(actual)
            for vecino in self.adjacencia.get(actual, []):
                grados[vecino] -= 1
                if grados[vecino] == 0:
                    cola.append(vecino)
        return orden
class ColaPrioridadPedidos:
    """
       Cola de prioridad para manejar pedidos según urgencia.

       Internamente usa un heap (heapq) donde:
       - Una prioridad más baja significa más urgente.
       - Se agrega un contador interno para evitar empates entre elementos
         con la misma prioridad.
       """

    def __init__(self):
        """Inicializa la cola de prioridad vacía."""
        self.heap = []
        self.contador = 0

    def agregar_pedido(self, prioridad: int, pedido: Dict):
        """
        Agrega un pedido a la cola.

        Args:
            prioridad (int): Valor de prioridad (menor = más urgente).
            pedido (Dict): Datos del pedido (productos, recursos, etc.).
        """
        heapq.heappush(self.heap, (prioridad, self.contador, pedido))
        self.contador += 1

    def obtener_siguiente(self) -> Dict:
        """
        Obtiene el siguiente pedido a procesar según su prioridad.

        Returns:
            Dict | None: El pedido con mayor prioridad, o None si la cola está vacía.
        """
        if self.heap:
            return heapq.heappop(self.heap)[2]
        return None

    def __len__(self) -> int:
        """
        Devuelve la cantidad de elementos en la cola.

        Returns:
            int: Número de pedidos pendientes.
        """
        return len(self.heap)


class OptimizadorProduccion:
    """
    Núcleo del sistema de optimización de producción.

    Responsabilidades principales:
        - Resolver el problema de selección óptima de productos
          (tipo mochila / knapsack con doble restricción: tiempo y material).
        - Aprovechar memoización (@lru_cache) para evitar recalcular subproblemas.
        - Mantener un cache persistente en JSON entre ejecuciones.
        - Ejecutar múltiples optimizaciones en paralelo (paralelismo).
        - Soportar procesamiento en lote de pedidos.
        - Exponer métricas para profiling y análisis de rendimiento.
    """

    def __init__(self, cache_file: str = 'cache_produccion.json'):
        """
        Inicializa el optimizador y las estructuras de soporte.

        Args:
            cache_file (str): Ruta del archivo donde se guardan resultados
                de optimizaciones previas (cache persistente).
        """
        self.cache_file = cache_file
        self.cache = self._cargar_cache()
        self.historial = []
        self.grafo_dependencias = GrafoDependencias()
        self.cola_pedidos = ColaPrioridadPedidos()

        # Estadísticas para profiling
        self.estadisticas = {
            'llamadas_optimizacion': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tiempo_total_optimizacion': 0.0,
            'pedidos_procesados': 0,
            'lineas_paralelas_ejecutadas': 0
        }

        # Tiempos individuales de ejecución para poder sacar métricas
        self.tiempos_ejecucion = []

    def _cargar_cache(self) -> Dict:
        """
        Carga el cache desde disco si existe y es válido.

        Returns:
            Dict: Estructura con optimizaciones previas, o un dict vacío.
        """
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _guardar_cache(self):
        """
        Persiste el contenido del cache en el archivo JSON configurado.
        """
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _generar_clave_cache(self, productos: List[Producto],
                             tiempo_max: int, material_max: int) -> str:
        """
        Genera una clave única para identificar una combinación productos+recursos.

        La clave incluye:
            - Productos (id, ganancia, tiempo, material) ordenados por id.
            - Tiempo máximo.
            - Material máximo.

        Args:
            productos (List[Producto]): Lista de productos.
            tiempo_max (int): Tiempo disponible.
            material_max (int): Material disponible.

        Returns:
            str: Cadena única que representa ese escenario de optimización.
        """
        productos_str = '|'.join(sorted([
            f"{p.id}:{p.ganancia}:{p.tiempo}:{p.material}"
            for p in productos
        ]))
        return f"{productos_str}_{tiempo_max}_{material_max}"

    @lru_cache(maxsize=1024)
    def _optimizar_subproblema(self, idx: int, tiempo_restante: int,
                               material_restante: int, productos_hash: str) -> Tuple[int, List]:
        """
        Resuelve recursivamente un subproblema de la mochila con memoización.

        Este método implementa el clásico enfoque de programación dinámica:
        en cada paso decide si tomar o no el producto actual, respetando
        las restricciones de tiempo y material.

        Args:
            idx (int): Índice del producto actual (se recorre hacia atrás).
            tiempo_restante (int): Unidades de tiempo que quedan disponibles.
            material_restante (int): Unidades de material que quedan disponibles.
            productos_hash (str): Representación serializada de la lista de productos.

        Returns:
            Tuple[int, List[str]]:
                - Ganancia máxima alcanzable en este subproblema.
                - Lista de IDs de productos seleccionados.
        """
        # Caso base
        if tiempo_restante <= 0 or material_restante <= 0 or idx < 0:
            return 0, []

        # Recuperar productos del hash
        productos = self._deserializar_productos(productos_hash)
        producto_actual = productos[idx]

        # Caso 1: no tomar el producto actual
        ganancia_no_tomar, seleccion_no_tomar = self._optimizar_subproblema(
            idx - 1, tiempo_restante, material_restante, productos_hash
        )

        # Caso 2: tomar el producto actual (solo si alcanzan los recursos)
        if (tiempo_restante >= producto_actual.tiempo and
                material_restante >= producto_actual.material):

            ganancia_tomar, seleccion_tomar = self._optimizar_subproblema(
                idx - 1,
                tiempo_restante - producto_actual.tiempo,
                material_restante - producto_actual.material,
                productos_hash
            )
            ganancia_tomar += producto_actual.ganancia

            # Agregar producto a la solución
            seleccion_tomar = seleccion_tomar + [producto_actual.id]

            # Elegir la mejor de las dos opciones
            if ganancia_tomar > ganancia_no_tomar:
                return ganancia_tomar, seleccion_tomar

        return ganancia_no_tomar, seleccion_no_tomar

    def _serializar_productos(self, productos: List[Producto]) -> str:
        """
        Serializa una lista de productos a una cadena.

        Se usa para generar un identificador estable que se pueda
        pasar al método memoizado `_optimizar_subproblema`.

        Args:
            productos (List[Producto]): Lista de productos.

        Returns:
            str: Representación en texto de los productos.
        """
        return ';'.join([
            f"{p.id},{p.nombre},{p.ganancia},{p.tiempo},{p.material}"
            for p in sorted(productos, key=lambda x: x.id)
        ])

    def _deserializar_productos(self, productos_str: str) -> List[Producto]:
        """
        Reconstruye objetos Producto a partir de la cadena serializada.

        Args:
            productos_str (str): Texto con información de productos.

        Returns:
            List[Producto]: Lista de instancias Producto.
        """
        productos = []
        for item in productos_str.split(';'):
            if item:
                id_p, nombre, ganancia, tiempo, material = item.split(',')
                productos.append(Producto(
                    id=id_p,
                    nombre=nombre,
                    ganancia=int(ganancia),
                    tiempo=int(tiempo),
                    material=int(material)
                ))
        return productos

    def optimizar_produccion(self, productos: List[Dict], tiempo_max: int,
                             material_max: int, use_cache: bool = True) -> Dict:
        """
        Punto de entrada principal para optimizar una combinación de productos.

        Acá se combinan:
            - Conversión de dicts a objetos Producto.
            - Uso (opcional) de cache persistente.
            - Llamada al algoritmo de programación dinámica.
            - Cálculo de métricas y armado de la respuesta final.

        Args:
            productos (List[Dict]): Lista de productos en formato diccionario.
            tiempo_max (int): Tiempo total disponible.
            material_max (int): Material total disponible.
            use_cache (bool): Si True, intenta reutilizar resultados del cache.

        Returns:
            Dict: Resultado con:
                - ganancia_total
                - tiempo_utilizado
                - material_utilizado
                - productos_seleccionados
                - eficiencias (%)
                - tiempos de ejecución
                - flags de uso de cache
        """
        inicio = time.perf_counter()
        self.estadisticas['llamadas_optimizacion'] += 1

        # Convertir a objetos Producto
        productos_obj = [Producto(**p) for p in productos]

        # Verificar cache
        if use_cache:
            cache_key = self._generar_clave_cache(productos_obj, tiempo_max, material_max)

            if cache_key in self.cache:
                self.estadisticas['cache_hits'] += 1
                print("CACHE HIT: Reutilizando optimización previa")
                resultado = self.cache[cache_key]
                resultado['cache_usado'] = True
                resultado['tiempo_ejecucion'] = 0.001  # Tiempo mínimo simbólico
                return resultado

            self.estadisticas['cache_misses'] += 1

        # Serializar productos para memoización
        productos_hash = self._serializar_productos(productos_obj)

        # Resolver con programación dinámica
        n = len(productos_obj) - 1
        ganancia_max, productos_seleccionados_ids = self._optimizar_subproblema(
            n, tiempo_max, material_max, productos_hash
        )

        # Procesar resultados
        conteo_productos = defaultdict(int)
        for pid in productos_seleccionados_ids:
            conteo_productos[pid] += 1

        tiempo_utilizado = 0
        material_utilizado = 0
        productos_seleccionados = []

        for prod in productos_obj:
            if prod.id in conteo_productos:
                cantidad = conteo_productos[prod.id]
                tiempo_utilizado += prod.tiempo * cantidad
                material_utilizado += prod.material * cantidad

                productos_seleccionados.append({
                    'id': prod.id,
                    'nombre': prod.nombre,
                    'cantidad': cantidad,
                    'ganancia_unit': prod.ganancia,
                    'ganancia_total': prod.ganancia * cantidad,
                    'tiempo_unit': prod.tiempo,
                    'material_unit': prod.material
                })

        # Construir resultado final
        resultado = {
            'ganancia_total': ganancia_max,
            'tiempo_utilizado': tiempo_utilizado,
            'material_utilizado': material_utilizado,
            'productos_seleccionados': productos_seleccionados,
            'productos_ids': productos_seleccionados_ids,
            'eficiencia_tiempo': (tiempo_utilizado / tiempo_max) * 100 if tiempo_max > 0 else 0,
            'eficiencia_material': (material_utilizado / material_max) * 100 if material_max > 0 else 0
        }

        # Tiempo de ejecución
        tiempo_ejecucion = time.perf_counter() - inicio
        resultado['tiempo_ejecucion'] = tiempo_ejecucion
        self.tiempos_ejecucion.append(tiempo_ejecucion)
        self.estadisticas['tiempo_total_optimizacion'] += tiempo_ejecucion

        # Guardar en cache si corresponde
        if use_cache:
            resultado['cache_key'] = cache_key
            resultado['cache_usado'] = False
            self.cache[cache_key] = resultado
            self._guardar_cache()

        # Guardar en historial
        self.historial.append({
            'timestamp': time.time(),
            'parametros': {
                'n_productos': len(productos),
                'tiempo_max': tiempo_max,
                'material_max': material_max
            },
            'resultado': resultado
        })

        print(f"Optimización completada en {tiempo_ejecucion:.4f} segundos")
        print(f"   Ganancia: ${ganancia_max}, Productos: {len(productos_seleccionados)}")

        return resultado

    def simular_lineas_produccion_paralelo(self,
                                           configuraciones: List[Dict]) -> List[Dict]:
        """
        Ejecuta varias optimizaciones en paralelo, una por línea de producción.

        Usa ThreadPoolExecutor para paralelizar llamadas a `optimizar_produccion`.

        Args:
            configuraciones (List[Dict]): Lista donde cada elemento tiene:
                - productos
                - tiempo_max
                - material_max

        Returns:
            List[Dict]: Lista de resultados por línea, incluyendo estado y errores.
        """
        print(f"Iniciando procesamiento paralelo de {len(configuraciones)} líneas...")
        resultados = []

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(4, len(configuraciones))
        ) as executor:
            futuros = {
                executor.submit(
                    self.optimizar_produccion,
                    config['productos'],
                    config['tiempo_max'],
                    config['material_max']
                ): i for i, config in enumerate(configuraciones)
            }

            for futuro in concurrent.futures.as_completed(futuros):
                linea_idx = futuros[futuro]
                try:
                    resultado = futuro.result(timeout=30)
                    resultados.append({
                        'linea': linea_idx + 1,
                        'resultado': resultado,
                        'estado': 'completado'
                    })
                    print(f"  Línea {linea_idx + 1}: ${resultado['ganancia_total']}")
                except concurrent.futures.TimeoutError:
                    resultados.append({
                        'linea': linea_idx + 1,
                        'error': 'Timeout',
                        'estado': 'error'
                    })
                except Exception as e:
                    resultados.append({
                        'linea': linea_idx + 1,
                        'error': str(e),
                        'estado': 'error'
                    })

        self.estadisticas['lineas_paralelas_ejecutadas'] += len(configuraciones)
        return resultados

    def procesar_lote_pedidos(self, pedidos: List[Dict],
                              batch_size: int = 10) -> List[Dict]:
        """
        Procesa un conjunto grande de pedidos en bloques (batching).

        Args:
            pedidos (List[Dict]): Lista de pedidos a optimizar.
            batch_size (int): Cantidad de pedidos por batch.

        Returns:
            List[Dict]: Resultados individuales de cada pedido.
        """
        print(f"Procesando lote de {len(pedidos)} pedidos en batches de {batch_size}...")
        resultados_totales = []

        for i in range(0, len(pedidos), batch_size):
            batch = pedidos[i:i + batch_size]
            print(f"  Procesando batch {i // batch_size + 1}/{(len(pedidos) - 1) // batch_size + 1}")

            for j, pedido in enumerate(batch, 1):
                resultado = self.optimizar_produccion(
                    pedido['productos'],
                    pedido['tiempo_max'],
                    pedido['material_max']
                )
                resultado['pedido_id'] = i + j
                resultados_totales.append(resultado)

                self.estadisticas['pedidos_procesados'] += 1

        return resultados_totales

    def generar_simulacion_produccion(self, n_simulaciones: int = 100) -> Generator[Dict, None, None]:
        """
        Genera múltiples escenarios aleatorios de producción y los optimiza.

        Esta función se pensó como herramienta de experimentación para ver
        cómo se comporta el optimizador ante variaciones aleatorias.

        Args:
            n_simulaciones (int): Número de escenarios a generar.

        Yields:
            Dict: Resultado de cada simulación (igual formato que optimizar_produccion).
        """
        import random
        productos_base = [
            {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5},
            {'id': 'P2', 'nombre': 'Mesa', 'ganancia': 150, 'tiempo': 3, 'material': 8},
            {'id': 'P3', 'nombre': 'Estante', 'ganancia': 200, 'tiempo': 4, 'material': 10},
        ]

        for i in range(n_simulaciones):
            productos = []
            for p in productos_base:
                variacion = random.uniform(0.8, 1.2)
                productos.append({
                    'id': p['id'],
                    'nombre': p['nombre'],
                    'ganancia': int(p['ganancia'] * variacion),
                    'tiempo': max(1, int(p['tiempo'] * random.uniform(0.9, 1.1))),
                    'material': max(1, int(p['material'] * random.uniform(0.9, 1.1)))
                })

            resultado = self.optimizar_produccion(
                productos,
                tiempo_max=random.randint(6, 12),
                material_max=random.randint(15, 30),
                use_cache=False  # No cacheo simulaciones
            )
            resultado['simulacion_id'] = i + 1
            yield resultado

    def obtener_metricas_rendimiento(self) -> Dict:
        """
        Calcula métricas globales de rendimiento del optimizador.

        Returns:
            Dict: Diccionario con datos como:
                - total_optimizaciones
                - cache_hits / cache_misses / cache_hit_rate
                - tiempos total, promedio, mínimo, máximo
                - pedidos_procesados
                - lineas_paralelas
                - tamano_cache
        """
        if not self.tiempos_ejecucion:
            return {}

        tiempos = self.tiempos_ejecucion
        metricas = {
            'total_optimizaciones': self.estadisticas['llamadas_optimizacion'],
            'cache_hits': self.estadisticas['cache_hits'],
            'cache_misses': self.estadisticas['cache_misses'],
            'cache_hit_rate': (self.estadisticas['cache_hits'] /
                               max(1, self.estadisticas['cache_hits'] +
                                   self.estadisticas['cache_misses'])) * 100,
            'tiempo_total': self.estadisticas['tiempo_total_optimizacion'],
            'tiempo_promedio': sum(tiempos) / len(tiempos),
            'tiempo_minimo': min(tiempos),
            'tiempo_maximo': max(tiempos),
            'pedidos_procesados': self.estadisticas['pedidos_procesados'],
            'lineas_paralelas': self.estadisticas['lineas_paralelas_ejecutadas'],
            'tamano_cache': len(self.cache)
        }

        return metricas

    def generar_reporte_profiling(self) -> str:
        """
        Genera un reporte de texto con las métricas de rendimiento.

        Returns:
            str: Cadena lista para mostrar en consola o en la interfaz.
        """
        metricas = self.obtener_metricas_rendimiento()

        reporte = "=" * 70 + "\n"
        reporte += "REPORTE DE PROFILING - OPTIMIZADOR DE PRODUCCIÓN\n"
        reporte += "=" * 70 + "\n\n"

        reporte += "ESTADÍSTICAS GENERALES:\n"
        reporte += "-" * 40 + "\n"
        reporte += f"• Optimizaciones realizadas: {metricas.get('total_optimizaciones', 0)}\n"
        reporte += f"• Pedidos procesados: {metricas.get('pedidos_procesados', 0)}\n"
        reporte += f"• Líneas paralelas ejecutadas: {metricas.get('lineas_paralelas', 0)}\n"
        reporte += f"• Tamaño de cache: {metricas.get('tamano_cache', 0)} entradas\n\n"

        reporte += "RENDIMIENTO DE CACHE:\n"
        reporte += "-" * 40 + "\n"
        reporte += f"• Cache hits: {metricas.get('cache_hits', 0)}\n"
        reporte += f"• Cache misses: {metricas.get('cache_misses', 0)}\n"
        reporte += f"• Cache hit rate: {metricas.get('cache_hit_rate', 0):.1f}%\n\n"

        reporte += "TIEMPOS DE EJECUCIÓN:\n"
        reporte += "-" * 40 + "\n"
        reporte += f"• Tiempo total: {metricas.get('tiempo_total', 0):.3f}s\n"
        reporte += f"• Tiempo promedio: {metricas.get('tiempo_promedio', 0):.4f}s\n"
        reporte += f"• Tiempo mínimo: {metricas.get('tiempo_minimo', 0):.4f}s\n"
        reporte += f"• Tiempo máximo: {metricas.get('tiempo_maximo', 0):.4f}s\n"

        return reporte

    def agregar_pedido_cola(self, pedido: Dict, prioridad: int = 5):
        """
        Agrega un pedido a la cola de prioridad interna.

        Args:
            pedido (Dict): Pedido a registrar.
            prioridad (int): Valor de prioridad (menor = más urgente).
        """
        self.cola_pedidos.agregar_pedido(prioridad, pedido)

    def procesar_cola_pedidos(self) -> List[Dict]:
        """
        Procesa todos los pedidos pendientes en la cola de prioridad.

        Returns:
            List[Dict]: Resultados de las optimizaciones de cada pedido.
        """
        resultados = []
        while len(self.cola_pedidos) > 0:
            pedido = self.cola_pedidos.obtener_siguiente()
            if pedido:
                resultado = self.optimizar_produccion(
                    pedido['productos'],
                    pedido['tiempo_max'],
                    pedido['material_max']
                )
                resultados.append(resultado)
        return resultados

    def limpiar_cache(self):
        """
        Limpia el cache en memoria y lo persiste vacío en disco.
        """
        self.cache = {}
        self._guardar_cache()
        print("Cache limpiado exitosamente")

    def obtener_historial(self, n_entradas: int = 10) -> List[Dict]:
        """
        Devuelve las últimas ejecuciones de optimización.

        Args:
            n_entradas (int): Cantidad de registros a devolver.

        Returns:
            List[Dict]: Lista de entradas de historial (las más recientes).
        """
        return self.historial[-n_entradas:] if self.historial else []