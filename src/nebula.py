"""
Módulo principal de integración para NEBULA

Este módulo integra los tres componentes principales de NEBULA:
- Módulo de Simulación Cuántica (QSM)
- Módulo de Redes Neuronales Bioinspirando (BNNM)
- Módulo de Memoria Holográfica (HMM)

Proporciona una interfaz unificada para utilizar las capacidades del sistema.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos de NEBULA
from src.qsm.quantum_simulator import QuantumSimulator
from src.bnnm.bioinspired_neural_network import BioinspiredNeuralNetwork, HolographicLayer
from src.hmm.holographic_memory import HolographicMemory, HolographicRAG

class NEBULA:
    """
    Sistema NEBULA que integra simulación cuántica, redes neuronales bioinspirando
    y memoria holográfica para crear un sistema de IA eficiente y avanzado.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el sistema NEBULA.
        
        Args:
            config: Configuración opcional para personalizar los componentes
        """
        # Configuración por defecto
        default_config = {
            'qsm': {
                'n_qubits': 8,
                'coherence_time': 1.0,
                'backend': 'auto'
            },
            'bnnm': {
                'layer_sizes': [64, 128, 64, 32],
                'use_contextual_activation': True
            },
            'hmm': {
                'dimensions': 512,
                'capacity': 10000,
                'storage_path': './nebula_memory'
            }
        }
        
        # Fusionar configuración proporcionada con valores por defecto
        if config is None:
            config = {}
        
        self.config = default_config.copy()
        for section in config:
            if section in self.config:
                self.config[section].update(config[section])
        
        # Inicializar componentes
        print("Inicializando NEBULA...")
        
        # Inicializar QSM
        print("Inicializando Módulo de Simulación Cuántica...")
        self.qsm = QuantumSimulator(
            n_qubits=self.config['qsm']['n_qubits'],
            coherence_time=self.config['qsm']['coherence_time'],
            backend=self.config['qsm']['backend']
        )
        
        # Inicializar BNNM
        print("Inicializando Módulo de Redes Neuronales Bioinspirando...")
        self.bnnm = BioinspiredNeuralNetwork(
            layer_sizes=self.config['bnnm']['layer_sizes'],
            use_contextual_activation=self.config['bnnm']['use_contextual_activation']
        )
        
        # Inicializar HMM
        print("Inicializando Módulo de Memoria Holográfica...")
        os.makedirs(self.config['hmm']['storage_path'], exist_ok=True)
        self.hmm = HolographicMemory(
            dimensions=self.config['hmm']['dimensions'],
            capacity=self.config['hmm']['capacity'],
            storage_path=self.config['hmm']['storage_path']
        )
        
        # Inicializar RAG
        self.rag = HolographicRAG(self.hmm)
        
        print("NEBULA inicializado correctamente.")
    
    def process_query(self, query: str, use_quantum_optimization: bool = True) -> Dict:
        """
        Procesa una consulta utilizando el pipeline completo de NEBULA.
        
        Args:
            query: Texto de la consulta
            use_quantum_optimization: Si se debe usar optimización cuántica
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        print(f"Procesando consulta: '{query}'")
        
        # Paso 1: Recuperar información relevante de la memoria holográfica
        retrieved_info = self.rag.query(query)
        
        # Paso 2: Preparar entrada para la red neural
        # Convertir consulta a vector de características
        query_features = self._text_to_features(query)
        
        # Combinar con información recuperada
        context_features = []
        for info in retrieved_info:
            # Extraer características del texto recuperado
            text_features = self._text_to_features(info['text'])
            # Ponderar por similitud
            weighted_features = text_features * info['similarity']
            context_features.append(weighted_features)
        
        # Promediar características de contexto
        if context_features:
            context_tensor = torch.tensor(np.mean(context_features, axis=0), dtype=torch.float32)
        else:
            context_tensor = torch.zeros(self.config['hmm']['dimensions'], dtype=torch.float32)
        
        # Paso 3: Procesar con red neural bioinspirando
        query_tensor = torch.tensor(query_features, dtype=torch.float32)
        processed_features = self.bnnm(query_tensor, context_tensor)
        
        # Paso 4: Optimización cuántica (opcional)
        if use_quantum_optimization:
            # Convertir características procesadas a formato adecuado para QSM
            processed_np = processed_features.detach().cpu().numpy()
            
            # Redimensionar si es necesario
            if len(processed_np) > 2**self.config['qsm']['n_qubits']:
                processed_np = processed_np[:2**self.config['qsm']['n_qubits']]
            elif len(processed_np) < 2**self.config['qsm']['n_qubits']:
                padded = np.zeros(2**self.config['qsm']['n_qubits'])
                padded[:len(processed_np)] = processed_np
                processed_np = padded
            
            # Codificar en estado cuántico
            self.qsm.apply_holographic_encoding(processed_np)
            
            # Aplicar QFT para transformar al dominio de frecuencia
            self.qsm.apply_quantum_fourier_transform()
            
            # Amplificar estados relevantes (simplificado)
            # En una implementación completa, se determinarían los estados a amplificar
            # basándose en análisis de la consulta y el contexto
            marked_states = [0, 1]  # Estados a amplificar
            self.qsm.apply_grover_iteration(marked_states)
            
            # Medir estado resultante
            measurement = self.qsm.measure_state()
            
            # Obtener vector de estado optimizado
            optimized_state = self.qsm.get_state_vector()
            
            # Convertir de nuevo a tensor para procesamiento final
            optimized_tensor = torch.tensor(np.abs(optimized_state), dtype=torch.float32)
            final_features = optimized_tensor
        else:
            # Sin optimización cuántica, usar directamente salida de BNNM
            final_features = processed_features
        
        # Paso 5: Generar respuesta
        response = self._generate_response(query, final_features, retrieved_info)
        
        # Preparar resultados
        results = {
            'query': query,
            'retrieved_info': retrieved_info,
            'response': response,
            'features': final_features.detach().cpu().numpy()
        }
        
        return results
    
    def learn_from_document(self, document: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Aprende de un documento, almacenándolo en la memoria holográfica.
        
        Args:
            document: Texto del documento
            metadata: Metadatos opcionales
            
        Returns:
            Lista de IDs de los fragmentos almacenados
        """
        print(f"Aprendiendo de documento ({len(document)} caracteres)...")
        
        # Almacenar documento en memoria holográfica
        chunk_ids = self.rag.add_document(document, metadata)
        
        print(f"Documento procesado y almacenado en {len(chunk_ids)} fragmentos.")
        
        return chunk_ids
    
    def adapt_from_feedback(self, query: str, response: str, feedback: float) -> None:
        """
        Adapta el sistema basándose en retroalimentación sobre una respuesta.
        
        Args:
            query: Consulta original
            response: Respuesta generada
            feedback: Valor de retroalimentación (-1 a 1, donde 1 es positivo)
        """
        print(f"Adaptando sistema basado en retroalimentación: {feedback}")
        
        # Convertir consulta y respuesta a características
        query_features = torch.tensor(self._text_to_features(query), dtype=torch.float32)
        response_features = torch.tensor(self._text_to_features(response), dtype=torch.float32)
        
        # Escalar retroalimentación
        scaled_feedback = max(-1.0, min(1.0, feedback))
        
        # Adaptar red neural
        # Si feedback es positivo, reforzar la asociación consulta-respuesta
        # Si feedback es negativo, debilitar la asociación
        target = response_features * (0.5 + 0.5 * scaled_feedback)
        
        # Adaptar red neural
        mse = self.bnnm.adapt(query_features, target)
        
        print(f"Adaptación completada. MSE: {mse:.6f}")
    
    def visualize_system_state(self) -> None:
        """Visualiza el estado actual de los componentes de NEBULA."""
        plt.figure(figsize=(15, 15))
        plt.suptitle("Estado del Sistema NEBULA", fontsize=16)
        
        # Visualizar estado cuántico
        plt.subplot(3, 1, 1)
        probabilities = np.abs(self.qsm.get_state_vector())**2
        states = [format(i, f'0{self.config["qsm"]["n_qubits"]}b') for i in range(len(probabilities))]
        plt.bar(states, probabilities)
        plt.title("Estado Cuántico (QSM)")
        plt.xlabel("Estado")
        plt.ylabel("Probabilidad")
        if len(states) > 16:
            plt.xticks(range(0, len(states), len(states)//8))
        
        # Visualizar red neural
        plt.subplot(3, 1, 2)
        # Obtener primer patrón holográfico
        if self.bnnm.holographic_layers:
            pattern = self.bnnm.holographic_layers[0].holographic_pattern.detach().cpu().numpy()
            plt.imshow(np.abs(pattern), cmap='viridis')
            plt.colorbar()
            plt.title("Patrón Holográfico de Primera Capa (BNNM)")
        else:
            plt.text(0.5, 0.5, "No hay capas holográficas disponibles", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Visualizar memoria holográfica
        plt.subplot(3, 1, 3)
        if self.hmm.pattern_count > 0:
            # Mostrar primeros patrones
            n_patterns = min(5, self.hmm.pattern_count)
            patterns = []
            for i, pattern_id in enumerate(list(self.hmm.id_to_index.keys())[:n_patterns]):
                index = self.hmm.id_to_index[pattern_id]
                pattern = self.hmm.memory_matrix[index]
                patterns.append(np.abs(pattern))
            
            # Apilar patrones para visualización
            stacked_patterns = np.vstack(patterns)
            plt.imshow(stacked_patterns, cmap='plasma', aspect='auto')
            plt.colorbar()
            plt.title(f"Patrones en Memoria Holográfica (HMM) - {self.hmm.pattern_count} total")
        else:
            plt.text(0.5, 0.5, "No hay patrones en memoria", 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
        plt.show()
    
    def save_state(self, directory: str) -> None:
        """
        Guarda el estado completo del sistema NEBULA.
        
        Args:
            directory: Directorio donde guardar el estado
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar configuración
        import json
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Guardar estado de BNNM
        torch.save(self.bnnm.state_dict(), os.path.join(directory, 'bnnm_state.pt'))
        
        # Guardar estado de HMM
        self.hmm.save_state(os.path.join(directory, 'hmm_state.json'))
        
        print(f"Estado del sistema guardado en {directory}")
    
    def load_state(self, directory: str) -> None:
        """
        Carga el estado completo del sistema NEBULA.
        
        Args:
            directory: Directorio desde donde cargar el estado
        """
        # Cargar configuración
        import json
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Reinicializar componentes con la configuración cargada
        self.__init__(self.config)
        
        # Cargar estado de BNNM
        self.bnnm.load_state_dict(torch.load(os.path.join(directory, 'bnnm_state.pt')))
        
        # Cargar estado de HMM
        self.hmm.load_state(os.path.join(directory, 'hmm_state.json'))
        
        print(f"Estado del sistema cargado desde {directory}")
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """
        Convierte texto a vector de características.
        
        Args:
            text: Texto a convertir
            
        Returns:
            Vector de características como array NumPy
        """
        # Implementación simplificada
        # En una implementación completa, se usarían técnicas más sofisticadas
        # como word embeddings, TF-IDF, etc.
        
        # Usar la función de HMM para consistencia
        return self.hmm._text_to_features(text)
    
    def _generate_response(self, query: str, features: torch.Tensor, 
                          retrieved_info: List[Dict]) -> str:
        """
        Genera una respuesta basada en características procesadas y contexto.
        
        Args:
            query: Consulta original
            features: Características procesadas
            retrieved_info: Información recuperada de la memoria
            
        Returns:
            Texto de respuesta generado
        """
        # En una implementación completa, aquí se utilizaría un modelo de lenguaje
        # para generar una respuesta coherente basada en las características y el contexto
        
        # Por ahora, generamos una respuesta simple basada en la información recuperada
        response = f"Respuesta a la consulta: '{query}'\n\n"
        
        if retrieved_info:
            response += "Basado en la información recuperada:\n"
            for i, info in enumerate(retrieved_info):
                response += f"- {info['text'][:100]}...\n"
        else:
            response += "No se encontró información relevante en la memoria."
        
        return response


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de NEBULA
    nebula = NEBULA()
    
    # Aprender de algunos documentos
    doc1 = """
    La física cuántica es una rama de la física que estudia el comportamiento de la materia
    y la energía a escalas muy pequeñas, como átomos y partículas subatómicas. A diferencia
    de la física clásica, la física cuántica introduce conceptos como la superposición de estados,
    el entrelazamiento cuántico y la dualidad onda-partícula.
    """
    
    doc2 = """
    La holografía es una técnica fotográfica que permite registrar y reconstruir imágenes
    tridimensionales. Se basa en el fenómeno de interferencia de ondas luminosas coherentes.
    Un holograma contiene información sobre la amplitud y la fase de la luz, lo que permite
    reconstruir la imagen completa desde cualquier ángulo.
    """
    
    doc3 = """
    Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento
    del cerebro humano. Consisten en unidades de procesamiento interconectadas (neuronas) que
    aprenden a reconocer patrones en los datos. Las redes neuronales profundas han revolucionado
    campos como la visión por computadora, el procesamiento del lenguaje natural y el reconocimiento de voz.
    """
    
    nebula.learn_from_document(doc1, {"topic": "física cuántica"})
    nebula.learn_from_document(doc2, {"topic": "holografía"})
    nebula.learn_from_document(doc3, {"topic": "inteligencia artificial"})
    
    # Procesar consultas
    query1 = "¿Cómo se relacionan la física cuántica y la holografía?"
    results1 = nebula.process_query(query1)
    print("\nRespuesta:")
    print(results1['response'])
    
    query2 = "Explica cómo las redes neuronales pueden beneficiarse de principios cuánticos"
    results2 = nebula.process_query(query2)
    print("\nRespuesta:")
    print(results2['response'])
    
    # Proporcionar retroalimentación para adaptación
    nebula.adapt_from_feedback(query1, results1['response'], 0.8)  # Retroalimentación positiva
    
    # Visualizar estado del sistema
    nebula.visualize_system_state()
    
    # Guardar estado
    nebula.save_state("./nebula_state")
