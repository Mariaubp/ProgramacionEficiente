"""
PUNTO DE ENTRADA PRINCIPAL
Este archivo inicializa todos los componentes del sistema:
1. OptimizadorProduccion - Motor de optimización
2. PredictorDemanda - Sistema de IA/ML
3. FabricaGUI - Interfaz gráfica

FUNCIONAMIENTO:
• main() - Función principal que crea y conecta todos los componentes
• Inicia el loop principal de Tkinter
"""
import tkinter as tk
from optimizador import OptimizadorProduccion
from predictorIA import PredictorDemanda
from interfazGraf import FabricaGUI

def main():
    print("=" * 70)
    print("OPTIMIZADOR INTELIGENTE DE PRODUCCIÓN")
    print("=" * 70)

    # Inicializar componentes
    optimizador = OptimizadorProduccion()
    predictor = PredictorDemanda()

    # Crear ventana principal
    root = tk.Tk()
    app = FabricaGUI(root, optimizador, predictor)

    # Iniciar loop principal
    root.mainloop()

if __name__ == "__main__":
    main()