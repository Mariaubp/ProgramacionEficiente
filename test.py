import unittest
import tempfile
import os
# Importa tus módulos (ajusta los nombres si son diferentes)
from optimizador import OptimizadorProduccion
from predictorIA import PredictorDemanda
"""
MÓDULO DE PRUEBAS UNITARIAS
Este módulo contiene pruebas para validar el correcto funcionamiento
de las clases principales del sistema.
PRUEBAS IMPLEMENTADAS:
• TestOptimizadorProduccion - Valida el optimizador
• TestPredictorDemanda - Valida el predictor de IA
FRAMEWORK:
• unittest - Framework estándar de Python
"""
class TestOptimizadorProduccion(unittest.TestCase):
    """Tests para la clase OptimizadorProduccion"""
    def setUp(self):
        self.optimizador = OptimizadorProduccion()

    def test_optimizacion_basica(self):
        """Test de optimización básica"""
        productos = [
            {'id': 'P1', 'nombre': 'Silla', 'ganancia': 100, 'tiempo': 2, 'material': 5},
            {'id': 'P2', 'nombre': 'Mesa', 'ganancia': 150, 'tiempo': 3, 'material': 8},
        ]
        resultado = self.optimizador.optimizar_produccion(productos, 6, 15)
        self.assertIn('ganancia_total', resultado)
        self.assertIn('productos_seleccionados', resultado)
        self.assertGreaterEqual(resultado['ganancia_total'], 0)
class TestPredictorDemanda(unittest.TestCase):
    """Tests para PredictorDemanda"""
    def test_prediccion_basica(self):
        """Test básico de predicción"""
        predictor = PredictorDemanda()
        datos = [{'mes': 1, 'demanda': 100}, {'mes': 2, 'demanda': 120}]

        predictor.cargar_datos_historicos(datos)
        predictor.entrenar_modelo()

        prediccion = predictor.predecir_demanda(3)
        self.assertIsInstance(prediccion, float)
if __name__ == '__main__':
    unittest.main()