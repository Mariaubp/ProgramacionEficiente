import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from predictorIA import PredictorDemanda
import threading
import queue
import time
from typing import Dict, List
import json
"""MÓDULO DE INTERFAZ GRÁFICA - FabricaGUI
Este módulo contiene la interfaz gráfica completa del sistema de optimización
de producción, implementada con Tkinter y matplotlib.
CARACTERÍSTICAS PRINCIPALES:
• 4 pestañas organizadas (Optimización, IA, Avanzado, Estadísticas)
• Gráficos interactivos con matplotlib
• Procesamiento en hilos secundarios (threading)
• Comunicación asíncrona mediante colas (queue)
• Widgets personalizados con estilos ttk
CLASE PRINCIPAL:
    FabricaGUI - Controla toda la interfaz gráfica
DEPENDENCIAS:
    tkinter, matplotlib, predictorIA, threading, queue
"""
class FabricaGUI:
    def __init__(self, root, optimizador, predictor):
        """
                Inicializa la interfaz gráfica.
                Args:
                    root: Ventana principal Tkinter
                    optimizador: Instancia de OptimizadorProduccion
                    predictor: Instancia de PredictorDemanda
                """
        self.root = root
        self.optimizador = optimizador
        self.predictor = predictor
        self.queue = queue.Queue()

        # Configuración de la ventana
        self._configurar_ventana()

        # Estilos personalizados
        self._configurar_estilos()

        # Crear widgets
        self._crear_widgets()

        # Cargar datos de ejemplo
        self._cargar_datos_ejemplo()

        # Iniciar procesamiento de cola
        self.root.after(100, self._procesar_queue)

        # Configurar cierre seguro
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _configurar_ventana(self):
        """Configura la ventana principal"""
        self.root.title("Optimizador Inteligente de Producción")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg="#e8e8e8")

    def _configurar_estilos(self):
        """Configura estilos personalizados para ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        # Fondo por defecto para todos los widgets ttk
        style.configure('.', background="#e8e8e8")
        # Frame principal de la app
        style.configure('Main.TFrame', background="#e8e8e8")
        colores = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }
        # Configurar estilos
        style.configure('Primary.TButton',
                        foreground='white',
                        background=colores['primary'],
                        padding=10)

        style.configure('Success.TButton',
                        foreground='white',
                        background=colores['success'],
                        padding=10)

        style.configure('Danger.TButton',
                        foreground='white',
                        background=colores['danger'],
                        padding=10)

        style.configure('Warning.TButton',
                        foreground='white',
                        background=colores['warning'],
                        padding=10)

        style.configure('Info.TButton',
                        foreground='white',
                        background=colores['info'],
                        padding=10)

        style.configure('Title.TLabel',
                        font=('Arial', 16, 'bold'),
                        foreground=colores['primary'])

        style.configure('Subtitle.TLabel',
                        font=('Arial', 12),
                        foreground=colores['secondary'])

    def _crear_widgets(self):
        """Configura estilos personalizados para ttk"""
        # Contenedor principal con scroll
        main_container = ttk.Frame(self.root, style="Main.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        # Canvas para scroll
        canvas = tk.Canvas(main_container, bg="#e8e8e8", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        # Actualizar región de scroll cuando cambia el contenido
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        # Ventana interna dentro del canvas
        scroll_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        # Ajustar el ancho del frame al ancho del canvas (para evitar bloque gris a la derecha)
        def _ajustar_ancho(event):
            canvas.itemconfig(scroll_window, width=event.width)
        canvas.bind("<Configure>", _ajustar_ancho)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        title_label = ttk.Label(
            header_frame,
            text="SISTEMA INTELIGENTE DE OPTIMIZACIÓN DE PRODUCCIÓN",
            style='Title.TLabel'
        )
        title_label.pack()
        subtitle_label = ttk.Label(
            header_frame,
            text="Proyecto Final | Integrante: Constanza Gigli",
            style='Subtitle.TLabel'
        )
        subtitle_label.pack()
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Optimización de Producción
        self._crear_pestana_optimizacion(notebook)

        #  Predicción de Demanda
        self._crear_pestana_ia(notebook)

        #  Características Avanzadas
        self._crear_pestana_avanzada(notebook)

        # Estadísticas y Reportes
        self._crear_pestana_estadisticas(notebook)
        self._crear_barra_estado(scrollable_frame)

    def _crear_pestana_optimizacion(self, notebook):
        """ Crea la pestaña de optimización de producción
        Esta pestaña permite:
        1. Configurar recursos (tiempo y material)
        2. Gestionar productos (agregar, editar, eliminar)
        3. Ejecutar optimizaciones (con/sin cache)
        4. Visualizar resultados (tablas + gráficos)
        Args:
            notebook (ttk.Notebook): Widget notebook donde agregar la pestaña
        Returns:
            None
        """
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=" Optimización")

        # Panel izquierdo: Configuración
        config_frame = ttk.LabelFrame(frame, text="Configuración de Producción", padding="15")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)

        # Recursos
        ttk.Label(config_frame, text="Recursos Disponibles:",
                  font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        ttk.Label(config_frame, text="Tiempo (horas):").grid(row=1, column=0, sticky=tk.W)
        self.tiempo_var = tk.IntVar(value=8)
        ttk.Spinbox(config_frame, from_=1, to=24, textvariable=self.tiempo_var,
                    width=10).grid(row=1, column=1, padx=(10, 0))

        ttk.Label(config_frame, text="Material (unidades):").grid(row=2, column=0, sticky=tk.W)
        self.material_var = tk.IntVar(value=20)
        ttk.Spinbox(config_frame, from_=1, to=100, textvariable=self.material_var,
                    width=10).grid(row=2, column=1, padx=(10, 0), pady=(0, 15))

        # Productos
        ttk.Label(config_frame, text="Productos:",
                  font=('Arial', 11, 'bold')).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        # Treeview para productos
        columns = ('ID', 'Nombre', 'Ganancia ($)', 'Tiempo (h)', 'Material (u)')
        self.tree_productos = ttk.Treeview(config_frame, columns=columns, show='headings', height=8)

        for col in columns:
            self.tree_productos.heading(col, text=col)
            self.tree_productos.column(col, width=90, minwidth=80)

        self.tree_productos.grid(row=4, column=0, columnspan=2, pady=(0, 10))

        # Scrollbar para productos
        scrollbar_prod = ttk.Scrollbar(config_frame, orient=tk.VERTICAL, command=self.tree_productos.yview)
        scrollbar_prod.grid(row=4, column=2, sticky='ns')
        self.tree_productos.configure(yscrollcommand=scrollbar_prod.set)

        # Controles de productos
        btn_frame = ttk.Frame(config_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(0, 15))
        ttk.Button(btn_frame, text="Agregar", command=self._agregar_producto,
                   width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Editar", command=self._editar_producto,
                   width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Eliminar", command=self._eliminar_producto,
                   width=12).pack(side=tk.LEFT, padx=2)

        # Botones de optimización
        ttk.Button(config_frame, text="OPTIMIZAR PRODUCCIÓN",
                   command=self._ejecutar_optimizacion, style='Success.TButton').grid(row=6, column=0, columnspan=2,
                                                                                      pady=(10, 5))
        ttk.Button(config_frame, text="OPTIMIZAR SIN CACHE",
                   command=lambda: self._ejecutar_optimizacion(use_cache=False),
                   style='Warning.TButton').grid(row=7, column=0, columnspan=2, pady=(0, 5))
        # Panel derecho: Resultados
        resultados_frame = ttk.LabelFrame(frame, text="Resultados de Optimización", padding="15")
        resultados_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        frame.columnconfigure(1, weight=1)
        resultados_frame.rowconfigure(2, weight=1)

        # Métricas rápidas
        metrics_frame = ttk.Frame(resultados_frame)
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        self.lbl_ganancia = ttk.Label(metrics_frame, text="Ganancia Total",
                                      font=('Arial', 14, 'bold'), foreground='#27ae60')
        self.lbl_ganancia.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_tiempo = ttk.Label(metrics_frame, text="Tiempo Usado: 0/8H",
                                    font=('Arial', 12))
        self.lbl_tiempo.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_material = ttk.Label(metrics_frame, text="Material Usado: 0/20u",
                                      font=('Arial', 12))
        self.lbl_material.pack(side=tk.LEFT)

        # Tabla de resultados
        ttk.Label(resultados_frame, text="Productos Seleccionados:",
                  font=('Arial', 11, 'bold')).grid(row=1, column=0, sticky=tk.W)

        columns_res = ('Producto', 'Cantidad', 'Ganancia/u', 'Ganancia Total', 'Tiempo', 'Material')
        self.tree_resultados = ttk.Treeview(resultados_frame, columns=columns_res, show='headings', height=6)

        for col in columns_res:
            self.tree_resultados.heading(col, text=col)
            self.tree_resultados.column(col, width=100, minwidth=80)

        self.tree_resultados.grid(row=2, column=0, sticky="nsew", pady=(5, 10))

        # Scrollbar para resultados
        scrollbar_res = ttk.Scrollbar(resultados_frame, orient=tk.VERTICAL, command=self.tree_resultados.yview)
        scrollbar_res.grid(row=2, column=1, sticky='ns')
        self.tree_resultados.configure(yscrollcommand=scrollbar_res.set)

        # Gráfico de resultados
        self.fig_optimizacion = Figure(figsize=(6, 4), dpi=80)
        self.ax_optimizacion = self.fig_optimizacion.add_subplot(111)
        self.canvas_optimizacion = FigureCanvasTkAgg(self.fig_optimizacion, master=resultados_frame)
        self.canvas_optimizacion.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Configurar grid weights
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

    def _crear_pestana_ia(self, notebook):
        """Crea la pestaña de IA y predicción"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="IA - Predicción")

        # Panel izquierdo: Entrenamiento del modelo
        entrenamiento_frame = ttk.LabelFrame(frame, text="Entrenamiento del Modelo", padding="15")
        entrenamiento_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)

        ttk.Label(entrenamiento_frame, text="Tipo de Modelo:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.modelo_var = tk.StringVar(value='lineal')
        modelo_combo = ttk.Combobox(entrenamiento_frame, textvariable=self.modelo_var,
                                    values=['lineal', 'arbol', 'random_forest'],
                                    state='readonly', width=15)
        modelo_combo.grid(row=0, column=1, sticky=tk.W, pady=(0, 5))

        ttk.Label(entrenamiento_frame, text="Datos Históricos:",
                  font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        # Área para datos históricos
        self.txt_datos_historicos = scrolledtext.ScrolledText(entrenamiento_frame, height=12, width=40)
        self.txt_datos_historicos.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        ttk.Button(entrenamiento_frame, text="Cargar Datos de Ejemplo",
                   command=self._cargar_datos_ia_ejemplo).grid(row=3, column=0, columnspan=2, pady=(0, 10))

        ttk.Button(entrenamiento_frame, text="Entrenar Modelo",
                   command=self._entrenar_modelo_ia, style='Info.TButton').grid(row=4, column=0, columnspan=2)

        # Panel derecho: Predicción
        prediccion_frame = ttk.LabelFrame(frame, text="Predicción de Demanda", padding="15")
        prediccion_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        frame.columnconfigure(1, weight=1)

        ttk.Label(prediccion_frame, text="Mes a Predecir:").grid(row=0, column=0, sticky=tk.W)

        self.mes_prediccion_var = tk.IntVar(value=6)
        ttk.Spinbox(prediccion_frame, from_=1, to=24, textvariable=self.mes_prediccion_var,
                    width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Button(prediccion_frame, text="Predecir Demanda",
                   command=self._predecir_demanda).grid(row=1, column=0, columnspan=2, pady=(10, 5))

        # Resultado de predicción
        self.lbl_resultado_prediccion = ttk.Label(prediccion_frame, text="Predicción: --",
                                                  font=('Arial', 14, 'bold'), foreground='#3498db')
        self.lbl_resultado_prediccion.grid(row=2, column=0, columnspan=2, pady=(10, 5))

        # Gráfico de predicción
        self.fig_prediccion = Figure(figsize=(6, 4), dpi=80)
        self.ax_prediccion = self.fig_prediccion.add_subplot(111)
        self.canvas_prediccion = FigureCanvasTkAgg(self.fig_prediccion, master=prediccion_frame)
        self.canvas_prediccion.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=(10, 0))

        # Métricas del modelo
        ttk.Label(prediccion_frame, text="Métricas del Modelo:",
                  font=('Arial', 11, 'bold')).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        self.txt_metricas = scrolledtext.ScrolledText(prediccion_frame, height=6, width=40)
        self.txt_metricas.grid(row=5, column=0, columnspan=2, pady=(0, 5))

    def _crear_pestana_avanzada(self, notebook):
        """Crea la pestaña de características avanzadas"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Avanzado")
        # 1. Paralelismo
        paralelo_frame = ttk.LabelFrame(frame, text="Paralelismo", padding="15")
        paralelo_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(paralelo_frame, text="Simular múltiples líneas de producción",
                  wraplength=200).pack(pady=(0, 10))

        ttk.Button(paralelo_frame, text="Ejecutar en Paralelo",
                   command=self._demo_paralelismo).pack()
        # 2. Batching
        batching_frame = ttk.LabelFrame(frame, text="Batching", padding="15")
        batching_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(batching_frame, text="Procesar lotes de pedidos",
                  wraplength=200).pack(pady=(0, 10))

        ttk.Button(batching_frame, text="Procesar Lote",
                   command=self._demo_batching).pack()
        # 3. Memoización
        memo_frame = ttk.LabelFrame(frame, text="Memoización", padding="15")
        memo_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        ttk.Label(memo_frame, text="Demostrar @lru_cache en acción",
                  wraplength=200).pack(pady=(0, 10))

        ttk.Button(memo_frame, text="Probar Memoización",
                   command=self._demo_memoizacion).pack()
        # 4. Caching
        cache_frame = ttk.LabelFrame(frame, text="Caching Inteligente", padding="15")
        cache_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        ttk.Label(cache_frame, text="Cache persistente en JSON",
                  wraplength=200).pack(pady=(0, 10))

        btn_frame_cache = ttk.Frame(cache_frame)
        btn_frame_cache.pack()
        ttk.Button(btn_frame_cache, text="Mostrar Cache",
                   command=self._mostrar_cache).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame_cache, text="Limpiar Cache",
                   command=self._limpiar_cache).pack(side=tk.LEFT, padx=2)

        # Configurar grid weights
        for i in range(2):
            frame.rowconfigure(i, weight=1)
            frame.columnconfigure(i, weight=1)

    def _crear_pestana_estadisticas(self, notebook):
        """Crea la pestaña de estadísticas y reportes"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Estadísticas")

        # Panel izquierdo: Profiling
        profiling_frame = ttk.LabelFrame(frame, text="Profiling y Rendimiento", padding="15")
        profiling_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)

        ttk.Button(profiling_frame, text="Generar Reporte de Profiling",
                   command=self._generar_reporte_profiling, width=25).pack(pady=(0, 10))

        self.txt_profiling = scrolledtext.ScrolledText(profiling_frame, height=20, width=40)
        self.txt_profiling.pack(fill=tk.BOTH, expand=True)

        # Panel derecho: Testing
        testing_frame = ttk.LabelFrame(frame, text="Testing y Validación", padding="15")
        testing_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        frame.columnconfigure(1, weight=1)

        ttk.Label(testing_frame, text="Pruebas Unitarias:",
                  font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        ttk.Button(testing_frame, text="Ejecutar Tests",
                   command=self._ejecutar_tests).grid(row=1, column=0, pady=(0, 10))

        ttk.Label(testing_frame, text="Resultados:",
                  font=('Arial', 11, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(10, 5))

        self.txt_testing = scrolledtext.ScrolledText(testing_frame, height=15, width=40)
        self.txt_testing.grid(row=3, column=0, sticky="nsew")
        # Configurar grid weights
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        testing_frame.rowconfigure(3, weight=1)
        testing_frame.columnconfigure(0, weight=1)

    def _crear_barra_estado(self, parent):
        """Crea la barra de estado inferior"""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="Sistema listo")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 font=('Arial', 9))
        status_label.pack(side=tk.LEFT, padx=5, pady=2)

        # Estadísticas rápidas
        self.stats_var = tk.StringVar(value="Optimizaciones: 0 | Cache Hits: 0")
        stats_label = ttk.Label(status_frame, textvariable=self.stats_var,
                                font=('Arial', 9))
        stats_label.pack(side=tk.RIGHT, padx=5, pady=2)

    def _cargar_datos_ejemplo(self):
        """Carga datos de ejemplo en la interfaz"""
        # Limpiar tabla actual
        for item in self.tree_productos.get_children():
            self.tree_productos.delete(item)

        productos_ejemplo = [
            ('P1', 'Silla', 100, 2, 5),
            ('P2', 'Mesa', 150, 3, 8),
            ('P3', 'Estante', 200, 4, 10),
            ('P4', 'Escritorio', 300, 6, 15),
            ('P5', 'Sillón', 250, 5, 12)
        ]

        for producto in productos_ejemplo:
            self.tree_productos.insert('', tk.END, values=producto)

        self.status_var.set("Datos de ejemplo cargados")

    def _cargar_datos_ia_ejemplo(self):
        """Carga datos de ejemplo para IA"""
        datos_ejemplo = """[
    {"mes": 1, "demanda": 100},
    {"mes": 2, "demanda": 120},
    {"mes": 3, "demanda": 110},
    {"mes": 4, "demanda": 130},
    {"mes": 5, "demanda": 140},
    {"mes": 6, "demanda": 150},
    {"mes": 7, "demanda": 160},
    {"mes": 8, "demanda": 155}
]"""
        self.txt_datos_historicos.delete(1.0, tk.END)
        self.txt_datos_historicos.insert(1.0, datos_ejemplo)
        self.status_var.set("Datos de IA cargados")
    def _agregar_producto(self):
        """Abre el diálogo para agregar un producto nuevo."""
        self._abrir_dialogo_producto(modo="agregar")

    def _editar_producto(self):
        """Abre el diálogo para editar el producto seleccionado."""
        self._abrir_dialogo_producto(modo="editar")

    def _abrir_dialogo_producto(self, modo="agregar"):
        """Diálogo común para agregar/editar productos."""
        item_id = None
        valores_actuales = ("", "", 100, 2, 5)  # valores por defecto

        if modo == "editar":
            seleccion = self.tree_productos.selection()
            if not seleccion:
                messagebox.showwarning("Advertencia", "Selecciona un producto para editar")
                return

            item_id = seleccion[0]
            valores_actuales = self.tree_productos.item(item_id, "values")
            # (id, nombre, ganancia, tiempo, material)

        dialog = tk.Toplevel(self.root)
        dialog.title("Editar Producto" if modo == "editar" else "Agregar Producto")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="ID del Producto:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        id_entry = ttk.Entry(dialog, width=20)
        id_entry.grid(row=0, column=1, padx=10, pady=10)
        id_entry.insert(0, valores_actuales[0])

        ttk.Label(dialog, text="Nombre:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        nombre_entry = ttk.Entry(dialog, width=20)
        nombre_entry.grid(row=1, column=1, padx=10, pady=10)
        nombre_entry.insert(0, valores_actuales[1])

        ttk.Label(dialog, text="Ganancia ($):").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        ganancia_entry = ttk.Spinbox(dialog, from_=1, to=1000, width=18)
        ganancia_entry.grid(row=2, column=1, padx=10, pady=10)
        ganancia_entry.set(valores_actuales[2])

        ttk.Label(dialog, text="Tiempo (horas):").grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        tiempo_entry = ttk.Spinbox(dialog, from_=1, to=24, width=18)
        tiempo_entry.grid(row=3, column=1, padx=10, pady=10)
        tiempo_entry.set(valores_actuales[3])

        ttk.Label(dialog, text="Material (unidades):").grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
        material_entry = ttk.Spinbox(dialog, from_=1, to=100, width=18)
        material_entry.grid(row=4, column=1, padx=10, pady=10)
        material_entry.set(valores_actuales[4])

        def guardar():
            try:
                valores = (
                    id_entry.get(),
                    nombre_entry.get(),
                    int(ganancia_entry.get()),
                    int(tiempo_entry.get()),
                    int(material_entry.get())
                )

                if modo == "agregar":
                    self.tree_productos.insert('', tk.END, values=valores)
                    self.status_var.set("Producto agregado")
                else:
                    # actualizar la fila seleccionada
                    self.tree_productos.item(item_id, values=valores)
                    self.status_var.set("Producto actualizado")

                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Valores inválidos. Usa números enteros.")

        ttk.Button(dialog, text="Guardar", command=guardar, style='Success.TButton').grid(
            row=5, column=0, columnspan=2, pady=20
        )

    def _eliminar_producto(self):
        """Edita el producto seleccionado"""
        seleccion = self.tree_productos.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Selecciona un producto para eliminar")
            return

        if messagebox.askyesno("Confirmar", "¿Eliminar producto seleccionado?"):
            for item in seleccion:
                self.tree_productos.delete(item)
            self.status_var.set("Producto eliminado")

    def _ejecutar_optimizacion(self, use_cache: bool = True):
        """Ejecuta una optimización de producción.
            Proceso:
            1. Obtiene productos de la tabla
            2. Obtiene recursos configurados
            3. Ejecuta optimización en hilo secundario
            4. Muestra resultados cuando finaliza
            Args:
                use_cache (bool): Si True, utiliza cache para acelerar cálculos repetidos
            Returns:
            None
            """
        # Obtener productos de la tabla
        productos = []
        for item in self.tree_productos.get_children():
            valores = self.tree_productos.item(item, 'values')
            productos.append({
                'id': valores[0],
                'nombre': valores[1],
                'ganancia': int(valores[2]),
                'tiempo': int(valores[3]),
                'material': int(valores[4])
            })

        if not productos:
            messagebox.showwarning("Advertencia", "Agrega al menos un producto")
            return

        # Obtener recursos
        tiempo_max = self.tiempo_var.get()
        material_max = self.material_var.get()

        # Actualizar estado
        cache_msg = "con cache" if use_cache else "sin cache"
        self.status_var.set(f"Optimizando producción {cache_msg}...")
        # Ejecutar en hilo separado
        thread = threading.Thread(
            target=self._optimizar_thread,
            args=(productos, tiempo_max, material_max, use_cache)
        )
        thread.daemon = True
        thread.start()

    def _optimizar_thread(self, productos, tiempo_max, material_max, use_cache):
        """Hilo para optimización"""
        try:
            resultado = self.optimizador.optimizar_produccion(
                productos, tiempo_max, material_max, use_cache
            )
            self.queue.put(('optimizacion_completa', resultado))
        except Exception as e:
            self.queue.put(('error', f"Error en optimización: {str(e)}"))

    def _procesar_queue(self):
        """Procesa mensajes de la cola"""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()

                if msg_type == 'optimizacion_completa':
                    self._mostrar_resultados_optimizacion(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.status_var.set("Error en optimización")
                elif msg_type == 'estado':
                    self.status_var.set(data)

        except queue.Empty:
            pass

        # Actualizar estadísticas
        self._actualizar_estadisticas()

        # Programar siguiente verificación
        self.root.after(100, self._procesar_queue)

    def _mostrar_resultados_optimizacion(self, resultado):
        """Muestra los resultados de la optimización"""
        # Actualizar métricas
        self.lbl_ganancia.config(text=f"Ganancia Total: ${resultado['ganancia_total']}")
        self.lbl_tiempo.config(text=f"Tiempo Usado: {resultado['tiempo_utilizado']}/{self.tiempo_var.get()}h")
        self.lbl_material.config(text=f"Material Usado: {resultado['material_utilizado']}/{self.material_var.get()}u")

        # Limpiar tabla anterior
        for item in self.tree_resultados.get_children():
            self.tree_resultados.delete(item)

        # Agregar nuevos resultados
        ganancia_total = 0
        tiempo_total = 0
        material_total = 0

        for prod in resultado['productos_seleccionados']:
            self.tree_resultados.insert('', tk.END, values=(
                prod['nombre'],
                prod['cantidad'],
                f"${prod['ganancia_unit']}",
                f"${prod['ganancia_total']}",
                f"{prod['tiempo_unit'] * prod['cantidad']}h",
                f"{prod['material_unit'] * prod['cantidad']}u"
            ))

            ganancia_total += prod['ganancia_total']
            tiempo_total += prod['tiempo_unit'] * prod['cantidad']
            material_total += prod['material_unit'] * prod['cantidad']

        # Actualizar gráfico
        self._actualizar_grafico_optimizacion(resultado)

        # Actualizar estado
        cache_usado = " (con cache)" if resultado.get('cache_usado', False) else " (sin cache)"
        self.status_var.set(f"Optimización completada{cache_usado} - Ganancia: ${ganancia_total}")
        if resultado.get('cache_usado', False):
            mensaje = f"Optimización CON CACHE - Ganancia: ${ganancia_total}"
        else:
            mensaje = f"Optimización SIN cache - Ganancia: ${ganancia_total}"

        self.status_var.set(mensaje)

    def _actualizar_grafico_optimizacion(self, resultado):
        """Actualiza el gráfico de optimización"""
        self.ax_optimizacion.clear()

        if resultado['productos_seleccionados']:
            nombres = [p['nombre'] for p in resultado['productos_seleccionados']]
            cantidades = [p['cantidad'] for p in resultado['productos_seleccionados']]
            ganancias = [p['ganancia_total'] for p in resultado['productos_seleccionados']]

            x = range(len(nombres))
            width = 0.35

            # Barras para cantidades
            self.ax_optimizacion.bar(x, cantidades, width, label='Cantidad', color='skyblue')

            # Barras para ganancias (eje secundario)
            ax2 = self.ax_optimizacion.twinx()
            ax2.bar([i + width for i in x], ganancias, width, label='Ganancia ($)', color='lightgreen')

            # Configurar ejes
            self.ax_optimizacion.set_xlabel('Productos')
            self.ax_optimizacion.set_ylabel('Cantidad', color='blue')
            ax2.set_ylabel('Ganancia ($)', color='green')

            self.ax_optimizacion.set_xticks([i + width / 2 for i in x])
            self.ax_optimizacion.set_xticklabels(nombres, rotation=45)

            # Leyenda combinada
            lines1, labels1 = self.ax_optimizacion.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax_optimizacion.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            self.ax_optimizacion.set_title('Producción Óptima')

        self.fig_optimizacion.tight_layout()
        self.canvas_optimizacion.draw()

    def _entrenar_modelo_ia(self):
        """Entrena el modelo de IA con los datos proporcionados"""
        try:
            # Obtener datos del text area
            datos_texto = self.txt_datos_historicos.get(1.0, tk.END).strip()
            if not datos_texto:
                messagebox.showwarning("Advertencia", "Ingresa datos históricos primero")
                return

            datos = json.loads(datos_texto)

            # Configurar predictor
            try:
                modelo_tipo = self.modelo_var.get()
            except Exception:
                modelo_tipo = 'lineal'  # Si falla, usa este por defecto

            self.predictor = PredictorDemanda(modelo_tipo)

            # Cargar y entrenar
            self.predictor.cargar_datos_historicos(datos)
            metricas = self.predictor.entrenar_modelo()

            # Mostrar métricas
            self.txt_metricas.delete(1.0, tk.END)
            for key, value in metricas.items():
                if key != 'predicciones':
                    self.txt_metricas.insert(tk.END, f"{key}: {value:.4f}\n")

            self.status_var.set(f"Modelo {modelo_tipo} entrenado exitosamente")

        except json.JSONDecodeError:
            messagebox.showerror("Error", "Formato JSON inválido")
        except Exception as e:
            messagebox.showerror("Error", f"Error entrenando modelo: {str(e)}")

    def _predecir_demanda(self):
        """Realiza una predicción de demanda"""
        if not hasattr(self.predictor, 'entrenado') or not self.predictor.entrenado:
            messagebox.showwarning("Advertencia", "Entrena el modelo primero")
            return

        try:
            mes = self.mes_prediccion_var.get()
            prediccion = self.predictor.predecir_demanda(mes)

            self.lbl_resultado_prediccion.config(
                text=f"Predicción para mes {mes}: {prediccion:.0f} unidades"
            )

            # Actualizar gráfico de predicción
            self._actualizar_grafico_prediccion(mes, prediccion)

            self.status_var.set(f"Predicción realizada: {prediccion:.0f} unidades")

        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")

    def _actualizar_grafico_prediccion(self, mes, prediccion):
        """Actualiza el gráfico de predicción"""
        self.ax_prediccion.clear()
        # Datos de ejemplo para el gráfico
        meses = list(range(1, mes + 1))
        demandas = [100 + i * 10 for i in range(len(meses) - 1)] + [prediccion]

        # Línea de datos históricos
        self.ax_prediccion.plot(meses[:-1], demandas[:-1], 'bo-', label='Histórico', linewidth=2)

        # Punto de predicción
        self.ax_prediccion.plot(meses[-1], demandas[-1], 'ro', markersize=10, label='Predicción')

        # Línea punteada para predicción
        self.ax_prediccion.axvline(x=meses[-1], color='r', linestyle='--', alpha=0.5)

        self.ax_prediccion.set_xlabel('Mes')
        self.ax_prediccion.set_ylabel('Demanda (unidades)')
        self.ax_prediccion.set_title('Predicción de Demanda')
        self.ax_prediccion.legend()
        self.ax_prediccion.grid(True, alpha=0.3)

        self.fig_prediccion.tight_layout()
        self.canvas_prediccion.draw()

    def _demo_paralelismo(self):
        """Demuestra el paralelismo"""
        configuraciones = [
            {
                'productos': [
                    {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5},
                    {'id': 'P2', 'nombre': 'Mesa', 'ganancia': 150, 'tiempo': 3, 'material': 8}
                ],
                'tiempo_max': 6,
                'material_max': 15
            },
            {
                'productos': [
                    {'id': 'P3', 'nombre': 'Estante', 'ganancia': 200, 'tiempo': 4, 'material': 10},
                    {'id': 'P4', 'nombre': 'Escritorio', 'ganancia': 300, 'tiempo': 6, 'material': 15}
                ],
                'tiempo_max': 8,
                'material_max': 20
            },
            {
                'productos': [
                    {'id': 'P5', 'nombre': 'Sillón', 'ganancia': 250, 'tiempo': 5, 'material': 12},
                    {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5}
                ],
                'tiempo_max': 7,
                'material_max': 18
            }
        ]

        self.status_var.set("Ejecutando líneas en paralelo...")

        thread = threading.Thread(
            target=self._ejecutar_paralelismo_thread,
            args=(configuraciones,)
        )
        thread.daemon = True
        thread.start()

    def _ejecutar_paralelismo_thread(self, configuraciones):
        """Hilo para paralelismo"""
        try:
            resultados = self.optimizador.simular_lineas_produccion_paralelo(configuraciones)

            reporte = "RESULTADOS DEL PARALELISMO:\n"
            ganancia_total = 0
            for res in resultados:
                if 'resultado' in res:
                    reporte += f"Línea {res['linea']}: ${res['resultado']['ganancia_total']}\n"
                    ganancia_total += res['resultado']['ganancia_total']

            reporte += f"\nGanancia total paralela: ${ganancia_total}"

            messagebox.showinfo("Paralelismo", reporte)
            self.status_var.set(f"Paralelismo completado - {len(configuraciones)} líneas")

        except Exception as e:
            self.queue.put(('error', f"Error en paralelismo: {str(e)}"))

    def _demo_batching(self):
        """Demuestra el procesamiento por lotes"""
        pedidos = []
        for i in range(1, 6):
            pedidos.append({
                'productos': [
                    {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5},
                    {'id': 'P2', 'nombre': 'Mesa', 'ganancia': 150, 'tiempo': 3, 'material': 8}
                ],
                'tiempo_max': 6 + i,
                'material_max': 15 + i * 2
            })

        self.status_var.set("Procesando lote de pedidos...")

        thread = threading.Thread(
            target=self._ejecutar_batching_thread,
            args=(pedidos,)
        )
        thread.daemon = True
        thread.start()

    def _ejecutar_batching_thread(self, pedidos):
        """Hilo para batching"""
        try:
            resultados = self.optimizador.procesar_lote_pedidos(pedidos, batch_size=2)

            reporte = "RESULTADOS DEL BATCHING:\n"
            reporte += f"Pedidos procesados: {len(resultados)}\n"

            ganancia_total = sum(r['ganancia_total'] for r in resultados)
            reporte += f"Ganancia total: ${ganancia_total}\n"
            reporte += f"Tiempo promedio: {sum(r.get('tiempo_ejecucion', 0) for r in resultados) / len(resultados):.4f}s por pedido"

            messagebox.showinfo("Batching", reporte)
            self.status_var.set(f"Batching completado - {len(pedidos)} pedidos")

        except Exception as e:
            self.queue.put(('error', f"Error en batching: {str(e)}"))

    def _demo_memoizacion(self):
        """Demuestra la memoización con @lru_cache"""
        try:
            productos = [
                {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5},
                {'id': 'P2', 'nombre': 'Mesa', 'ganancia': 150, 'tiempo': 3, 'material': 8},
                {'id': 'P3', 'nombre': 'Estante', 'ganancia': 200, 'tiempo': 4, 'material': 10},
                {'id': 'P4', 'nombre': 'Escritorio', 'ganancia': 300, 'tiempo': 6, 'material': 15},
            ]

            self.status_var.set("Probando memoización...")

            # Limpiar cache LRU interna de la función recursiva
            self.optimizador._optimizar_subproblema.cache_clear()
            N = 80

            # Primera ejecución: LRU fría
            inicio1 = time.time()
            for _ in range(N):
                self.optimizador.optimizar_produccion(
                    productos, 12, 30, use_cache=False
                )
            tiempo1 = time.time() - inicio1

            # Segunda ejecución: misma instancia, LRU caliente
            inicio2 = time.time()
            for _ in range(N):
                self.optimizador.optimizar_produccion(
                    productos, 12, 30, use_cache=False
                )
            tiempo2 = time.time() - inicio2

            speedup = tiempo1 / tiempo2 if tiempo2 > 0 else 1.0

            reporte = "DEMOSTRACIÓN DE MEMOIZACIÓN (@lru_cache)\n"
            reporte += f"Repeticiones por caso: {N}\n\n"
            reporte += f"Primera tanda (LRU fría):    {tiempo1:.4f} s\n"
            reporte += f"Segunda tanda (LRU caliente): {tiempo2:.4f} s\n"
            reporte += f"Speedup aproximado: {speedup:.1f}x\n\n"
            reporte += "La memoización evita recalcular subproblemas repetidos."

            messagebox.showinfo("Memoización", reporte)
            self.status_var.set(f"Memoización demostrada (~{speedup:.1f}x más rápido)")

        except Exception as e:
            messagebox.showerror("Error en demo de memoización", str(e))
            self.status_var.set("Error en demo de memoización")

    def _mostrar_cache(self):
        """Muestra el contenido del cache"""
        cache_size = len(self.optimizador.cache)
        reporte = "CACHE DEL OPTIMIZADOR:\n"
        reporte += f"Entradas en cache: {cache_size}\n"
        reporte += f"Cache hits: {self.optimizador.estadisticas['cache_hits']}\n"
        reporte += f"Cache misses: {self.optimizador.estadisticas['cache_misses']}\n"

        if cache_size > 0:
            reporte += "\nÚltimas 5 entradas:\n"
            keys = list(self.optimizador.cache.keys())[-5:]
            for key in keys:
                reporte += f"• {key[:50]}...\n"
        messagebox.showinfo("Cache Inteligente", reporte)
        self.status_var.set(f"Cache mostrado - {cache_size} entradas")

    def _limpiar_cache(self):
        """Limpia el cache"""
        if messagebox.askyesno("Confirmar", "¿Limpiar todo el cache?"):
            self.optimizador.limpiar_cache()
            self.status_var.set("Cache limpiado")

    def _generar_reporte_profiling(self):
        """Genera reporte de profiling"""
        metricas = self.optimizador.obtener_metricas_rendimiento()
        reporte = "REPORTE DE PROFILING:\n"
        reporte += "=" * 40 + "\n\n"
        for key, value in metricas.items():
            if isinstance(value, float):
                reporte += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                reporte += f"{key.replace('_', ' ').title()}: {value}\n"

        self.txt_profiling.delete(1.0, tk.END)
        self.txt_profiling.insert(1.0, reporte)

        self.status_var.set("Reporte de profiling generado")

    def _ejecutar_tests(self):
        """Ejecuta tests unitarios"""
        self.txt_testing.delete(1.0, tk.END)
        self.txt_testing.insert(1.0, "Ejecutando tests unitarios...\n")
        self.status_var.set("Ejecutando tests...")

        # Ejecutar tests en segundo plano
        thread = threading.Thread(target=self._ejecutar_tests_thread)
        thread.daemon = True
        thread.start()

    def _ejecutar_tests_thread(self):
        """Hilo para ejecutar tests"""
        try:
            # Importar y ejecutar tests
            from test import TestOptimizadorProduccion, TestPredictorDemanda
            import unittest

            # Crear test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizadorProduccion)
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPredictorDemanda))

            # Ejecutar tests
            runner = unittest.TextTestRunner(verbosity=2, stream=self.TestStream(self.txt_testing))
            result = runner.run(suite)

            # Mostrar resultados
            self.queue.put(('estado', f"Tests completados: {result.testsRun} ejecutados"))

        except Exception as e:
            self.queue.put(('error', f"Error ejecutando tests: {str(e)}"))

    class TestStream:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, text):
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)

        def flush(self):
            pass

    def _actualizar_estadisticas(self):
        """Actualiza las estadísticas en la barra de estado"""
        stats = self.optimizador.estadisticas
        self.stats_var.set(
            f"Optimizaciones: {stats['llamadas_optimizacion']} | "
            f"Cache Hits: {stats['cache_hits']} | "
            f"Pedidos: {stats['pedidos_procesados']}"
        )

    def _on_closing(self):
        """Maneja el cierre de la aplicación"""
        if messagebox.askokcancel("Salir", "¿Deseas salir de la aplicación?"):
            # Guardar cache antes de salir
            if hasattr(self.optimizador, '_guardar_cache'):
                self.optimizador._guardar_cache()
            self.root.destroy()