"""
Módulo de predicción de demanda para la fábrica.
Este módulo implementa modelos de machine learning para la predicción
de demanda, siguiendo el patrón de diseño Strategy.

ARQUITECTURA:
• Interfaz abstracta ModeloIA
• Implementaciones concretas: ModeloLineal, ModeloArbolDecision
• Clase fachada PredictorDemanda

MODELOS IMPLEMENTADOS:
• Regresión lineal (LinearRegression)
• Árbol de decisión (DecisionTreeRegressor)

MÉTRICAS DE EVALUACIÓN:
• MAE (Error Absoluto Medio)
• MSE (Error Cuadrático Medio)
• RMSE (Raíz del Error Cuadrático Medio)
• R² (Coeficiente de determinación)
"""

import time
import os
import json
import pickle
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union

import numpy as np


class ModeloTipo(Enum):
    """
    Tipos de modelos de IA soportados por el predictor.

    Values:
        LINEAL: Regresión lineal clásica.
        ARBOL: Árbol de decisión para regresión.
        RANDOM_FOREST: (pendiente) Random Forest, por ahora uso un placeholder.
        SVR: (reservado) Support Vector Regression, sin implementar en este trabajo.
    """

    LINEAL = "lineal"
    ARBOL = "arbol"
    RANDOM_FOREST = "random_forest"
    SVR = "svr"


@dataclass
class DatosEntrenamiento:
    """
    Contenedor de los datos de entrenamiento ya preparados.

    Attributes:
        X (np.ndarray): Matriz de características (features).
        y (np.ndarray): Vector objetivo (demanda histórica).
        caracteristicas (List[str]): Nombres de las características usadas.
        tamanio (int): Cantidad de registros utilizados.
    """

    X: np.ndarray
    y: np.ndarray
    caracteristicas: List[str]
    tamanio: int

    def __str__(self):
        """Devuelve un resumen legible de los datos de entrenamiento."""
        return f"DatosEntrenamiento(tamaño={self.tamanio}, características={len(self.caracteristicas)})"


@dataclass
class ResultadoPrediccion:
    """
    Estructura para encapsular una predicción enriquecida.

    No la uso a fondo en la interfaz, pero me deja preparado
    para devolver más información que un simple número.

    Attributes:
        valor_predicho (float): Predicción principal.
        intervalo_confianza (Tuple[float, float]): Rango aproximado alrededor de la predicción.
        modelo_usado (str): Nombre del modelo que hizo la predicción.
        certeza (float): Nivel de confianza (0 a 100).
    """

    valor_predicho: float
    intervalo_confianza: Tuple[float, float]
    modelo_usado: str
    certeza: float  # 0-100%


class ModeloIA(ABC):
    """
    Interfaz base para cualquier modelo de IA que quiera usar.

    La idea es poder cambiar el algoritmo (lineal, árbol, random forest, etc.)
    sin tener que reescribir todo el resto del código.
    """

    @abstractmethod
    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo con los datos dados.

        Args:
            X (np.ndarray): Matriz de entrada.
            y (np.ndarray): Vector objetivo.
        """
        pass

    @abstractmethod
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones para los datos de entrada.

        Args:
            X (np.ndarray): Datos sobre los que se quiere predecir.

        Returns:
            np.ndarray: Predicciones calculadas por el modelo.
        """
        pass

    @abstractmethod
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo con un conjunto de test.

        Args:
            X_test (np.ndarray): Datos de entrada de test.
            y_test (np.ndarray): Valores reales de test.

        Returns:
            Dict: Métricas de evaluación (MAE, MSE, RMSE, R2, etc.).
        """
        pass

    @abstractmethod
    def guardar(self, ruta: str) -> None:
        """
        Persiste el modelo entrenado en disco.

        Args:
            ruta (str): Ruta del archivo donde se guarda el modelo.
        """
        pass

    @abstractmethod
    def cargar(self, ruta: str) -> None:
        """
        Carga un modelo previamente guardado.

        Args:
            ruta (str): Ruta del archivo desde donde se carga el modelo.
        """
        pass


class ModeloLineal(ModeloIA):
    """
    Implementación concreta de un modelo de regresión lineal.

    Internamente usa `LinearRegression` de scikit-learn y un `StandardScaler`
    para normalizar las características.
    """

    def __init__(self):
        """
        Inicializa el modelo lineal y el scaler.

        Si scikit-learn no está instalado, deja el modelo como None
        y muestra un mensaje de aviso.
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            self.modelo = LinearRegression()
            self.scaler = StandardScaler()
            self.entrenado = False
        except ImportError:
            print("sklearn no instalado. Instala con: pip install scikit-learn")
            self.modelo = None
            self.scaler = None
            self.entrenado = False

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo de regresión lineal.

        Primero escala las características y luego ajusta el modelo.

        Args:
            X (np.ndarray): Matriz de entrada.
            y (np.ndarray): Vector objetivo.
        """
        if self.modelo is None:
            return

        # Escalar características
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar modelo
        self.modelo.fit(X_scaled, y)
        self.entrenado = True

        # Métrica rápida para tener feedback
        score = self.modelo.score(X_scaled, y)
        print(f"Modelo lineal entrenado. R²: {score:.3f}")

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice valores a partir de los datos de entrada.

        Args:
            X (np.ndarray): Datos sobre los que se quiere predecir.

        Returns:
            np.ndarray: Valores predichos.

        Raises:
            ValueError: Si el modelo todavía no fue entrenado.
        """
        if not self.entrenado or self.modelo is None:
            raise ValueError("Modelo no entrenado")

        X_scaled = self.scaler.transform(X)
        return self.modelo.predict(X_scaled)

    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo lineal usando diferentes métricas estándar.

        Args:
            X_test (np.ndarray): Datos de test.
            y_test (np.ndarray): Valores reales de test.

        Returns:
            Dict: Métricas MAE, MSE, RMSE, R2 y las predicciones.
        """
        if not self.entrenado or self.modelo is None:
            return {"error": "Modelo no entrenado"}

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.modelo.predict(X_test_scaled)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'predicciones': y_pred.tolist()
        }

    def guardar(self, ruta: str) -> None:
        """
        Guarda el modelo lineal y el scaler en un archivo pickle.

        Args:
            ruta (str): Ruta del archivo .pkl.
        """
        if self.modelo is None:
            return

        with open(ruta, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'scaler': self.scaler,
                'entrenado': self.entrenado
            }, f)

    def cargar(self, ruta: str) -> None:
        """
        Carga un modelo lineal previamente guardado.

        Args:
            ruta (str): Ruta del archivo .pkl.
        """
        if not os.path.exists(ruta):
            return

        with open(ruta, 'rb') as f:
            datos = pickle.load(f)
            self.modelo = datos['modelo']
            self.scaler = datos['scaler']
            self.entrenado = datos['entrenado']


class ModeloArbolDecision(ModeloIA):
    """
    Implementación de un árbol de decisión para regresión.

    Usa `DecisionTreeRegressor` de scikit-learn.
    """

    def __init__(self, max_depth: int = 5):
        """
        Inicializa el árbol con una profundidad máxima dada.

        Args:
            max_depth (int): Profundidad máxima del árbol.
        """
        try:
            from sklearn.tree import DecisionTreeRegressor
            self.modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            self.entrenado = False
        except ImportError:
            print("sklearn no instalado")
            self.modelo = None
            self.entrenado = False

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el árbol de decisión.

        Args:
            X (np.ndarray): Datos de entrada.
            y (np.ndarray): Valores objetivo.
        """
        if self.modelo is None:
            return

        self.modelo.fit(X, y)
        self.entrenado = True
        print(f"Árbol de decisión entrenado. Profundidad: {self.modelo.get_depth()}")

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice valores usando el árbol entrenado.

        Args:
            X (np.ndarray): Datos de entrada.

        Returns:
            np.ndarray: Predicciones.

        Raises:
            ValueError: Si el modelo no fue entrenado.
        """
        if not self.entrenado or self.modelo is None:
            raise ValueError("Modelo no entrenado")
        return self.modelo.predict(X)

    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el árbol de decisión (mismas métricas que el lineal).

        Args:
            X_test (np.ndarray): Datos de test.
            y_test (np.ndarray): Valores reales.

        Returns:
            Dict: Métricas MAE, MSE, RMSE, R2 y predicciones.
        """
        if not self.entrenado or self.modelo is None:
            return {"error": "Modelo no entrenado"}

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.modelo.predict(X_test)

        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'predicciones': y_pred.tolist()
        }

    def guardar(self, ruta: str) -> None:
        """
        Guarda el árbol de decisión en un archivo pickle.

        Args:
            ruta (str): Ruta del archivo .pkl.
        """
        if self.modelo is None:
            return

        with open(ruta, 'wb') as f:
            pickle.dump(
                {
                    'modelo': self.modelo,
                    'entrenado': self.entrenado,
                },
                f
            )

    def cargar(self, ruta: str) -> None:
        """
        Carga un árbol de decisión previamente guardado.

        Args:
            ruta (str): Ruta del archivo .pkl.
        """
        if not os.path.exists(ruta):
            return
        with open(ruta, 'rb') as f:
            datos = pickle.load(f)
            self.modelo = datos['modelo']
            self.entrenado = datos.get('entrenado', True)


class PredictorDemanda:
    """
    Fachada principal para trabajar con modelos de predicción de demanda.

    Desde la interfaz gráfica solo interactúo con esta clase:
        - elijo el tipo de modelo,
        - cargo datos históricos,
        - entreno el modelo,
        - consulto métricas,
        - hago predicciones para nuevos meses.
    """

    def __init__(self, modelo_tipo: Union[str, ModeloTipo] = ModeloTipo.LINEAL):
        """
        Crea un nuevo predictor de demanda con el modelo indicado.

        Args:
            modelo_tipo (Union[str, ModeloTipo]): Tipo de modelo a utilizar.
                Puede pasarse como string ("lineal", "arbol", etc.) o como enum.
        """
        if isinstance(modelo_tipo, str):
            modelo_tipo = ModeloTipo(modelo_tipo.lower())

        self.modelo_tipo = modelo_tipo
        self.modelo = self._crear_modelo(modelo_tipo)
        self.entrenado = False
        self.datos_entrenamiento: DatosEntrenamiento | None = None
        self.historial_predicciones: List[Dict] = []
        self.metricas: Dict = {}

        print(f"Predictor inicializado con modelo: {modelo_tipo.value}")

    def _crear_modelo(self, modelo_tipo: ModeloTipo) -> ModeloIA:
        """
        Crea la instancia concreta del modelo de IA según el tipo elegido.

        Args:
            modelo_tipo (ModeloTipo): Tipo de modelo a utilizar.

        Returns:
            ModeloIA: Instancia del modelo correspondiente.

        Raises:
            ValueError: Si se pasa un modelo no soportado.
        """
        if modelo_tipo == ModeloTipo.LINEAL:
            return ModeloLineal()
        elif modelo_tipo == ModeloTipo.ARBOL:
            return ModeloArbolDecision()
        elif modelo_tipo == ModeloTipo.RANDOM_FOREST:
            # En este trabajo lo dejo preparado pero reutilizo el lineal
            # para no agregar más complejidad.
            return ModeloLineal()
        else:
            raise ValueError(f"Modelo no soportado: {modelo_tipo}")

    def cargar_datos_historicos(self, datos: List[Dict],
                                caracteristica: str = 'mes') -> DatosEntrenamiento:
        """
        Convierte una lista de dicts con datos históricos en arrays NumPy.

        La idea es mantener la carga simple: cada registro tiene
        un campo `caracteristica` (por defecto, 'mes') y un campo 'demanda'.

        Args:
            datos (List[Dict]): Lista de registros históricos.
            caracteristica (str): Nombre del campo a usar como feature principal.

        Returns:
            DatosEntrenamiento: Estructura con X, y y metadatos.

        Raises:
            ValueError: Si hay menos de 2 puntos de datos.
        """
        if len(datos) < 2:
            raise ValueError("Se necesitan al menos 2 puntos de datos")

        # Extraer características y target
        X = np.array([d[caracteristica] for d in datos]).reshape(-1, 1)
        y = np.array([d['demanda'] for d in datos])

        self.datos_entrenamiento = DatosEntrenamiento(
            X=X,
            y=y,
            caracteristicas=[caracteristica],
            tamanio=len(datos)
        )
        print(f"Datos cargados: {len(datos)} registros")
        return self.datos_entrenamiento

    def entrenar_modelo(self, test_size: float = 0.2) -> Dict:
        """
        Entrena el modelo configurado y calcula métricas sobre un set de test.

        El flujo es:
            - Tomar datos de `self.datos_entrenamiento`.
            - Dividir en train/test.
            - Entrenar el modelo.
            - Evaluar con el test.
            - Guardar el modelo entrenado en disco.

        Args:
            test_size (float): Proporción de datos reservados para test.

        Returns:
            Dict: Métricas calculadas en el conjunto de test.

        Raises:
            ValueError: Si no se cargaron datos históricos antes.
        """
        if self.datos_entrenamiento is None:
            raise ValueError("Primero carga datos históricos")

        from sklearn.model_selection import train_test_split

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            self.datos_entrenamiento.X,
            self.datos_entrenamiento.y,
            test_size=test_size,
            random_state=42
        )

        # Entrenar modelo
        self.modelo.entrenar(X_train, y_train)
        self.entrenado = True

        # Evaluar modelo
        self.metricas = self.modelo.evaluar(X_test, y_test)

        # Guardar modelo entrenado
        self.guardar_modelo()

        print("Modelo entrenado exitosamente")
        print(f"   R² en test: {self.metricas.get('R2', 0):.3f}")
        print(f"   RMSE: {self.metricas.get('RMSE', 0):.2f}")

        return self.metricas

    def predecir_demanda(self, valores: Union[int, float, List]) -> Union[float, List[float]]:
        """
        Predice la demanda para uno o varios valores (por ejemplo, meses).

        Si el modelo todavía no fue entrenado, recurre a una estrategia simple:
        usar el promedio histórico como predicción base.

        Args:
            valores (int | float | List): Un valor o lista de valores sobre los
                que se quiere predecir (por ejemplo, número de mes).

        Returns:
            float | List[float]: Predicción o lista de predicciones.
        """
        # Fallback: modelo no entrenado -> promedio histórico
        if not self.entrenado:
            print("Modelo no entrenado. Usando promedio histórico.")
            if self.datos_entrenamiento is not None:
                promedio = np.mean(self.datos_entrenamiento.y)
                return promedio if isinstance(valores, (int, float)) else [promedio] * len(valores)
            # Si ni siquiera hay datos, devuelvo un valor fijo
            return 100.0 if isinstance(valores, (int, float)) else [100.0] * len(valores)

        # Convertir a array 2D para scikit-learn
        if isinstance(valores, (int, float)):
            X = np.array([[valores]])
            prediccion = self.modelo.predecir(X)[0]
        else:
            X = np.array(valores).reshape(-1, 1)
            prediccion = self.modelo.predecir(X).tolist()

        # Guardar en historial
        self.historial_predicciones.append({
            'valores': valores if isinstance(valores, list) else [valores],
            'predicciones': prediccion if isinstance(prediccion, list) else [prediccion],
            'timestamp': time.time()
        })

        return prediccion

    def predecir_proximos_meses(self, n_meses: int = 6, mes_inicio: int = 1) -> Dict[int, float]:
        """
        Predice la demanda para una secuencia de meses futuros.

        Args:
            n_meses (int): Cantidad de meses a predecir.
            mes_inicio (int): Mes desde el cual empezar a predecir.

        Returns:
            Dict[int, float]: Mapa {mes: predicción}.
        """
        meses = list(range(mes_inicio, mes_inicio + n_meses))
        predicciones = self.predecir_demanda(meses)

        return {mes: pred for mes, pred in zip(meses, predicciones)}

    def guardar_modelo(self, ruta: str = 'modelo_demanda.pkl'):
        """
        Guarda el modelo actual en disco si está entrenado.

        Args:
            ruta (str): Ruta del archivo donde se va a guardar.
        """
        if self.entrenado:
            self.modelo.guardar(ruta)
            print(f"Modelo guardado en: {ruta}")

    def cargar_modelo(self, ruta: str = 'modelo_demanda.pkl'):
        """
        Intenta cargar un modelo ya entrenado desde disco.

        Args:
            ruta (str): Ruta del archivo .pkl a cargar.
        """
        if os.path.exists(ruta):
            self.modelo.cargar(ruta)
            self.entrenado = True
            print(f"Modelo cargado desde: {ruta}")
        else:
            print(f"No se encontró modelo en: {ruta}")

    def generar_reporte(self) -> str:
        """
        Genera un pequeño reporte de estado del predictor.

        Incluye:
            - Tipo de modelo.
            - Si está entrenado o no.
            - Tamaño de los datos de entrenamiento.
            - Métricas del modelo (si existen).
            - Cantidad de predicciones realizadas.

        Returns:
            str: Cadena lista para mostrarse en la interfaz o consola.
        """
        reporte = "=" * 60 + "\n"
        reporte += "REPORTE DEL PREDICTOR DE DEMANDA\n"
        reporte += "=" * 60 + "\n\n"

        reporte += f"Modelo: {self.modelo_tipo.value}\n"
        reporte += f"Entrenado: {'Sí' if self.entrenado else 'No'}\n"

        if self.datos_entrenamiento:
            reporte += f"Datos de entrenamiento: {self.datos_entrenamiento.tamanio} registros\n"

        if self.metricas:
            reporte += "\nMÉTRICAS DEL MODELO:\n"
            reporte += "-" * 40 + "\n"
            for key, value in self.metricas.items():
                if key != 'predicciones':
                    reporte += f"{key}: {value:.4f}\n"

        if self.historial_predicciones:
            reporte += f"\nPredicciones realizadas: {len(self.historial_predicciones)}\n"

        return reporte
