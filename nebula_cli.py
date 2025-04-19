#!/usr/bin/env python3
"""
NEBULA CLI - Interfaz de línea de comandos para el sistema NEBULA

Este script proporciona una interfaz de línea de comandos para ejecutar
el sistema NEBULA directamente desde terminales, incluyendo entornos
de GitHub como Codespaces o GitHub Actions.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar que podemos importar desde el directorio actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Intentar importar la versión simplificada de NEBULA
try:
    from simplified_nebula import SimplifiedNEBULA
except ImportError:
    print("Error: No se pudo importar SimplifiedNEBULA. Asegúrate de estar ejecutando este script desde el directorio raíz del proyecto.")
    sys.exit(1)

def setup_environment():
    """Prepara el entorno para la ejecución de NEBULA."""
    print("Preparando entorno para NEBULA...")
    
    # Crear directorios necesarios
    os.makedirs("./nebula_data", exist_ok=True)
    os.makedirs("./nebula_results", exist_ok=True)
    
    # Verificar disponibilidad de NumPy
    try:
        import numpy
        print("NumPy disponible para operaciones matemáticas")
    except ImportError:
        print("Advertencia: NumPy no está instalado. Instalando...")
        os.system("pip install numpy")
    
    # Verificar disponibilidad de Matplotlib
    try:
        import matplotlib
        print("Matplotlib disponible para visualizaciones")
    except ImportError:
        print("Advertencia: Matplotlib no está instalado. Instalando...")
        os.system("pip install matplotlib")
    
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    
    print("Entorno preparado correctamente.")

def load_training_data():
    """
    Carga datos de entrenamiento para NEBULA.
    
    Returns:
        Lista de documentos con sus metadatos
    """
    print("Cargando datos de entrenamiento...")
    
    # Datos de entrenamiento sobre física cuántica, holografía y redes neuronales
    training_data = [
        {
            "text": """
            La física cuántica es una rama fundamental de la física que describe la naturaleza a escalas atómicas y subatómicas.
            A diferencia de la física clásica, la física cuántica introduce conceptos revolucionarios como la superposición de estados,
            donde las partículas pueden existir en múltiples estados simultáneamente hasta ser observadas. El entrelazamiento cuántico
            es otro fenómeno fascinante donde partículas entrelazadas mantienen correlaciones instantáneas independientemente de la
            distancia que las separe.
            """,
            "metadata": {"topic": "física cuántica", "importance": "alta", "category": "fundamentos"}
        },
        {
            "text": """
            La holografía es una técnica avanzada que permite registrar y reconstruir imágenes tridimensionales completas.
            Se basa en el fenómeno de interferencia de ondas luminosas coherentes, típicamente generadas por láseres. En un
            proceso holográfico, un haz de luz se divide en dos: un haz de referencia que llega directamente a la placa
            fotográfica y un haz de objeto que se refleja en el objeto a holografiar antes de llegar a la placa.
            """,
            "metadata": {"topic": "holografía", "importance": "alta", "category": "tecnología óptica"}
        },
        {
            "text": """
            Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano.
            Están compuestas por unidades de procesamiento interconectadas llamadas neuronas artificiales, organizadas en capas.
            La capa de entrada recibe los datos iniciales, las capas ocultas realizan transformaciones complejas, y la capa de
            salida produce el resultado final.
            """,
            "metadata": {"topic": "redes neuronales", "importance": "alta", "category": "inteligencia artificial"}
        },
        {
            "text": """
            La optimización de modelos de lenguaje mediante principios cuánticos y holográficos representa una frontera
            innovadora en inteligencia artificial. Los grandes modelos de lenguaje (LLMs) convencionales requieren enormes
            recursos computacionales debido a su arquitectura basada en transformers con atención completa. La aplicación
            de principios cuánticos permite representar información en espacios de alta dimensionalidad con menor número
            de parámetros.
            """,
            "metadata": {"topic": "optimización de LLMs", "importance": "alta", "category": "investigación avanzada"}
        }
    ]
    
    print(f"Cargados {len(training_data)} documentos de entrenamiento.")
    return training_data

def run_demo_queries(nebula):
    """
    Ejecuta consultas de demostración en NEBULA.
    
    Args:
        nebula: Instancia de NEBULA
        
    Returns:
        Diccionario con resultados de las consultas
    """
    print("\nEjecutando consultas de demostración...")
    
    demo_queries = [
        "¿Qué es la física cuántica y cómo se relaciona con la holografía?",
        "Explica cómo las redes neuronales pueden optimizarse usando principios cuánticos",
        "¿Cómo se pueden optimizar los grandes modelos de lenguaje usando física óptica y cuántica?"
    ]
    
    query_results = []
    
    for i, query in enumerate(demo_queries):
        print(f"\nConsulta {i+1}: '{query}'")
        
        # Procesar consulta
        start_time = time.time()
        result = nebula.process_query(query)
        processing_time = time.time() - start_time
        
        print(f"Tiempo de procesamiento: {processing_time:.4f} segundos")
        print(f"Respuesta: {result['response'][:150]}...")
        
        # Guardar resultado
        query_results.append({
            "query": query,
            "response": result['response'],
            "processing_time": processing_time,
            "retrieved_info_count": len(result['retrieved_info'])
        })
        
        # Proporcionar retroalimentación positiva para adaptación
        nebula.adapt_from_feedback(query, result['response'], 0.9)
    
    print("\nDemostración de consultas completada.")
    return query_results

def interactive_mode(nebula):
    """
    Ejecuta NEBULA en modo interactivo, permitiendo al usuario hacer consultas.
    
    Args:
        nebula: Instancia de NEBULA
    """
    print("\n=== Modo Interactivo de NEBULA ===")
    print("Escribe tus consultas y NEBULA las responderá.")
    print("Escribe 'salir', 'exit' o 'q' para terminar.")
    
    while True:
        try:
            # Solicitar consulta al usuario
            query = input("\n> ")
            
            # Verificar si el usuario quiere salir
            if query.lower() in ['salir', 'exit', 'q']:
                print("Saliendo del modo interactivo. ¡Hasta pronto!")
                break
            
            # Procesar consulta
            if query.strip():
                start_time = time.time()
                result = nebula.process_query(query)
                processing_time = time.time() - start_time
                
                print(f"\nTiempo de procesamiento: {processing_time:.4f} segundos")
                print(f"Respuesta:\n{result['response']}")
                
                # Solicitar retroalimentación
                feedback_str = input("\n¿Qué tan útil fue esta respuesta? (0-10): ")
                try:
                    feedback = float(feedback_str) / 10.0  # Normalizar a 0-1
                    if 0 <= feedback <= 1:
                        nebula.adapt_from_feedback(query, result['response'], feedback)
                        print(f"Gracias por tu retroalimentación. NEBULA ha adaptado su sistema.")
                    else:
                        print("Valor fuera de rango. La retroalimentación debe estar entre 0 y 10.")
                except ValueError:
                    print("Entrada no válida. Se requiere un número entre 0 y 10.")
            else:
                print("Por favor, ingresa una consulta válida.")
                
        except KeyboardInterrupt:
            print("\nOperación interrumpida por el usuario. Saliendo...")
            break
        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")

def process_custom_document(nebula, file_path):
    """
    Procesa un documento personalizado y lo añade a la memoria de NEBULA.
    
    Args:
        nebula: Instancia de NEBULA
        file_path: Ruta al archivo de documento
        
    Returns:
        True si el procesamiento fue exitoso, False en caso contrario
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            print(f"Error: El archivo {file_path} no existe.")
            return False
        
        # Leer el contenido del archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extraer metadatos básicos
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        
        # Determinar tipo de documento basado en extensión
        doc_type = "texto"
        if file_ext.lower() in ['.md', '.markdown']:
            doc_type = "markdown"
        elif file_ext.lower() in ['.txt', '.text']:
            doc_type = "texto plano"
        elif file_ext.lower() in ['.py', '.python']:
            doc_type = "código python"
        elif file_ext.lower() in ['.js', '.javascript']:
            doc_type = "código javascript"
        
        # Crear metadatos
        metadata = {
            "file_name": file_name,
            "file_type": doc_type,
            "file_size": len(content),
            "import_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Aprender del documento
        print(f"Procesando documento: {file_name} ({doc_type}, {len(content)} caracteres)")
        chunk_ids = nebula.learn_from_document(content, metadata)
        
        print(f"Documento procesado exitosamente. Dividido en {len(chunk_ids)} fragmentos.")
        return True
        
    except Exception as e:
        print(f"Error al procesar el documento: {str(e)}")
        return False

def main():
    """Función principal para ejecutar NEBULA desde la línea de comandos."""
    parser = argparse.ArgumentParser(description='NEBULA CLI - Sistema de IA basado en física óptica avanzada y física cuántica')
    
    # Argumentos generales
    parser.add_argument('--load', type=str, help='Ruta para cargar estado de NEBULA')
    parser.add_argument('--save', type=str, help='Ruta para guardar estado de NEBULA')
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: demo
    demo_parser = subparsers.add_parser('demo', help='Ejecutar demostración de NEBULA')
    
    # Comando: interactive
    interactive_parser = subparsers.add_parser('interactive', help='Iniciar modo interactivo')
    
    # Comando: query
    query_parser = subparsers.add_parser('query', help='Realizar una consulta específica')
    query_parser.add_argument('text', type=str, help='Texto de la consulta')
    
    # Comando: learn
    learn_parser = subparsers.add_parser('learn', help='Aprender de un documento')
    learn_parser.add_argument('file', type=str, help='Ruta al archivo de documento')
    
    # Comando: visualize
    visualize_parser = subparsers.add_parser('visualize', help='Visualizar estado del sistema')
    visualize_parser.add_argument('--output', type=str, default='nebula_state.png', 
                                help='Ruta para guardar la visualización')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Preparar entorno
    setup_environment()
    
    # Cargar o crear NEBULA
    if args.load and os.path.exists(args.load):
        print(f"Cargando NEBULA desde {args.load}...")
        nebula = SimplifiedNEBULA()
        nebula.load_state(args.load)
        print("NEBULA cargado correctamente.")
    else:
        print("Creando nueva instancia de NEBULA...")
        nebula = SimplifiedNEBULA()
        
        # Entrenar con datos por defecto
        training_data = load_training_data()
        for i, item in enumerate(training_data):
            print(f"Procesando documento {i+1}/{len(training_data)}: {item['metadata']['topic']}")
            nebula.learn_from_document(item['text'], item['metadata'])
        
        print("NEBULA inicializado y entrenado correctamente.")
    
    # Ejecutar comando específico
    if args.command == 'demo':
        # Ejecutar demostración
        query_results = run_demo_queries(nebula)
        
    elif args.command == 'interactive':
        # Iniciar modo interactivo
        interactive_mode(nebula)
        
    elif args.command == 'query':
        # Procesar consulta específica
        query = args.text
        print(f"Procesando consulta: '{query}'")
        
        start_time = time.time()
        result = nebula.process_query(query)
        processing_time = time.time() - start_time
        
        print(f"Tiempo de procesamiento: {processing_time:.4f} segundos")
        print(f"Respuesta:\n{result['response']}")
        
    elif args.command == 'learn':
        # Aprender de documento
        process_custom_document(nebula, args.file)
        
    elif args.command == 'visualize':
        # Visualizar estado del sistema
        output_path = args.output
        nebula.visualize_system_state(output_path)
        print(f"Visualización guardada en {output_path}")
        
    else:
        # Si no se especifica comando, mostrar ayuda
        if len(sys.argv) <= 1:
            parser.print_help()
        else:
            print("Comando no reconocido. Usa --help para ver los comandos disponibles.")
    
    # Guardar estado si se especificó
    if args.save:
        save_dir = args.save
        nebula.save_state(save_dir)
        print(f"Estado de NEBULA guardado en {save_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
