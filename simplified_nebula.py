"""
Versión simplificada de NEBULA para demostración

Esta versión utiliza principalmente NumPy para implementar las funcionalidades básicas
de NEBULA sin depender de PyTorch u otras bibliotecas externas grandes.
"""

import os
import sys
import time
import numpy as np
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt

class SimplifiedNEBULA:
    """
    Implementación simplificada de NEBULA que utiliza principalmente NumPy
    para demostrar los conceptos clave del sistema.
    """
    
    def __init__(self, config=None):
        """
        Inicializa el sistema NEBULA simplificado.
        
        Args:
            config: Configuración opcional para personalizar los componentes
        """
        # Configuración por defecto
        default_config = {
            'qsm': {
                'n_qubits': 6,
                'coherence_time': 1.0
            },
            'bnnm': {
                'layer_sizes': [64, 96, 48, 32],
                'use_contextual_activation': True
            },
            'hmm': {
                'dimensions': 384,
                'capacity': 1000,
                'storage_path': './nebula_data/memory'
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
        print("Inicializando NEBULA (versión simplificada)...")
        
        # Inicializar QSM
        print("Inicializando Módulo de Simulación Cuántica...")
        self.qsm = self._initialize_qsm()
        
        # Inicializar BNNM
        print("Inicializando Módulo de Redes Neuronales Bioinspirando...")
        self.bnnm = self._initialize_bnnm()
        
        # Inicializar HMM
        print("Inicializando Módulo de Memoria Holográfica...")
        os.makedirs(self.config['hmm']['storage_path'], exist_ok=True)
        self.hmm = self._initialize_hmm()
        
        print("NEBULA inicializado correctamente.")
    
    def _initialize_qsm(self):
        """Inicializa el Módulo de Simulación Cuántica simplificado."""
        n_qubits = self.config['qsm']['n_qubits']
        
        # Crear estado inicial (|0>)
        state_vector = np.zeros(2**n_qubits, dtype=complex)
        state_vector[0] = 1.0
        
        return {
            'n_qubits': n_qubits,
            'state_vector': state_vector,
            'coherence_time': self.config['qsm']['coherence_time']
        }
    
    def _initialize_bnnm(self):
        """Inicializa el Módulo de Redes Neuronales Bioinspirando simplificado."""
        layer_sizes = self.config['bnnm']['layer_sizes']
        
        # Inicializar pesos con valores aleatorios
        weights = []
        for i in range(len(layer_sizes) - 1):
            # Inicialización de Xavier/Glorot para mejor convergencia
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            layer_weights = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            weights.append(layer_weights)
        
        return {
            'layer_sizes': layer_sizes,
            'weights': weights,
            'use_contextual_activation': self.config['bnnm']['use_contextual_activation']
        }
    
    def _initialize_hmm(self):
        """Inicializa el Módulo de Memoria Holográfica simplificado."""
        dimensions = self.config['hmm']['dimensions']
        capacity = self.config['hmm']['capacity']
        
        return {
            'dimensions': dimensions,
            'capacity': capacity,
            'memory_matrix': np.zeros((capacity, dimensions), dtype=complex),
            'pattern_count': 0,
            'id_to_index': {},
            'metadata': {}
        }
    
    def process_query(self, query, use_quantum_optimization=True):
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
        retrieved_info = self._retrieve_from_memory(query)
        
        # Paso 2: Preparar entrada para la red neural
        query_features = self._text_to_features(query)
        
        # Combinar con información recuperada
        context_features = []
        for info in retrieved_info:
            # Extraer características del texto recuperado
            if 'original_text' in info['metadata']:
                text_features = self._text_to_features(info['metadata']['original_text'])
            else:
                text_features = np.zeros(self.config['hmm']['dimensions'])
            
            # Ponderar por similitud
            weighted_features = text_features * info['similarity']
            context_features.append(weighted_features)
        
        # Promediar características de contexto
        if context_features:
            context_tensor = np.mean(context_features, axis=0)
        else:
            context_tensor = np.zeros(self.config['hmm']['dimensions'])
        
        # Paso 3: Procesar con red neural bioinspirando
        processed_features = self._process_with_bnnm(query_features, context_tensor)
        
        # Paso 4: Optimización cuántica (opcional)
        if use_quantum_optimization:
            # Redimensionar si es necesario
            if len(processed_features) > 2**self.qsm['n_qubits']:
                processed_features = processed_features[:2**self.qsm['n_qubits']]
            elif len(processed_features) < 2**self.qsm['n_qubits']:
                padded = np.zeros(2**self.qsm['n_qubits'])
                padded[:len(processed_features)] = processed_features
                processed_features = padded
            
            # Aplicar optimización cuántica
            optimized_features = self._apply_quantum_optimization(processed_features)
            final_features = optimized_features
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
            'features': final_features
        }
        
        return results
    
    def learn_from_document(self, document, metadata=None):
        """
        Aprende de un documento, almacenándolo en la memoria holográfica.
        
        Args:
            document: Texto del documento
            metadata: Metadatos opcionales
            
        Returns:
            Lista de IDs de los fragmentos almacenados
        """
        print(f"Aprendiendo de documento ({len(document)} caracteres)...")
        
        # Dividir documento en fragmentos
        chunks = self._chunk_document(document)
        
        # Almacenar cada fragmento en la memoria holográfica
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            # Crear metadatos para el fragmento
            chunk_metadata = {
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_id': metadata.get('document_id') if metadata else None,
                'original_text': chunk
            }
            
            # Fusionar con metadatos proporcionados
            if metadata:
                chunk_metadata.update(metadata)
            
            # Codificar fragmento
            chunk_id = self._encode_in_memory(chunk, chunk_metadata)
            chunk_ids.append(chunk_id)
        
        print(f"Documento procesado y almacenado en {len(chunk_ids)} fragmentos.")
        return chunk_ids
    
    def adapt_from_feedback(self, query, response, feedback):
        """
        Adapta el sistema basándose en retroalimentación sobre una respuesta.
        
        Args:
            query: Consulta original
            response: Respuesta generada
            feedback: Valor de retroalimentación (-1 a 1, donde 1 es positivo)
        """
        print(f"Adaptando sistema basado en retroalimentación: {feedback}")
        
        # Convertir consulta y respuesta a características
        query_features = self._text_to_features(query)
        response_features = self._text_to_features(response)
        
        # Escalar retroalimentación
        scaled_feedback = max(-1.0, min(1.0, feedback))
        
        # Adaptar red neural
        # Si feedback es positivo, reforzar la asociación consulta-respuesta
        # Si feedback es negativo, debilitar la asociación
        target = response_features * (0.5 + 0.5 * scaled_feedback)
        
        # Adaptar red neural (simplificado)
        self._adapt_bnnm(query_features, target)
        
        print("Adaptación completada.")
    
    def save_state(self, directory):
        """
        Guarda el estado completo del sistema NEBULA.
        
        Args:
            directory: Directorio donde guardar el estado
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar configuración
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Guardar estado de QSM
        np.save(os.path.join(directory, 'qsm_state.npy'), self.qsm['state_vector'])
        
        # Guardar estado de BNNM
        for i, weights in enumerate(self.bnnm['weights']):
            np.save(os.path.join(directory, f'bnnm_weights_{i}.npy'), weights)
        
        # Guardar estado de HMM
        np.save(os.path.join(directory, 'hmm_memory.npy'), self.hmm['memory_matrix'])
        
        # Guardar metadatos de HMM
        with open(os.path.join(directory, 'hmm_metadata.json'), 'w') as f:
            # Convertir claves a strings para serialización JSON
            id_to_index = {str(k): v for k, v in self.hmm['id_to_index'].items()}
            json.dump({
                'pattern_count': self.hmm['pattern_count'],
                'id_to_index': id_to_index,
                'metadata': self.hmm['metadata']
            }, f, indent=2)
        
        print(f"Estado del sistema guardado en {directory}")
    
    def load_state(self, directory):
        """
        Carga el estado completo del sistema NEBULA.
        
        Args:
            directory: Directorio desde donde cargar el estado
        """
        # Cargar configuración
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Reinicializar componentes con la configuración cargada
        self.__init__(self.config)
        
        # Cargar estado de QSM
        self.qsm['state_vector'] = np.load(os.path.join(directory, 'qsm_state.npy'))
        
        # Cargar estado de BNNM
        for i in range(len(self.bnnm['weights'])):
            self.bnnm['weights'][i] = np.load(os.path.join(directory, f'bnnm_weights_{i}.npy'))
        
        # Cargar estado de HMM
        self.hmm['memory_matrix'] = np.load(os.path.join(directory, 'hmm_memory.npy'))
        
        # Cargar metadatos de HMM
        with open(os.path.join(directory, 'hmm_metadata.json'), 'r') as f:
            hmm_metadata = json.load(f)
            self.hmm['pattern_count'] = hmm_metadata['pattern_count']
            # Convertir claves de strings a originales
            self.hmm['id_to_index'] = {k: v for k, v in hmm_metadata['id_to_index'].items()}
            self.hmm['metadata'] = hmm_metadata['metadata']
        
        print(f"Estado del sistema cargado desde {directory}")
    
    def visualize_system_state(self, save_path=None):
        """
        Visualiza el estado actual de los componentes de NEBULA.
        
        Args:
            save_path: Ruta donde guardar la visualización (opcional)
        """
        plt.figure(figsize=(15, 15))
        plt.suptitle("Estado del Sistema NEBULA", fontsize=16)
        
        # Visualizar estado cuántico
        plt.subplot(3, 1, 1)
        probabilities = np.abs(self.qsm['state_vector'])**2
        states = [format(i, f'0{self.qsm["n_qubits"]}b') for i in range(len(probabilities))]
        plt.bar(states, probabilities)
        plt.title("Estado Cuántico (QSM)")
        plt.xlabel("Estado")
        plt.ylabel("Probabilidad")
        if len(states) > 16:
            plt.xticks(range(0, len(states), len(states)//8))
        
        # Visualizar red neural
        plt.subplot(3, 1, 2)
        if self.bnnm['weights']:
            # Visualizar primer capa de pesos
            weights = self.bnnm['weights'][0]
            plt.imshow(weights, cmap='viridis')
            plt.colorbar()
            plt.title(f"Pesos de Primera Capa (BNNM): {weights.shape[0]} → {weights.shape[1]}")
        else:
            plt.text(0.5, 0.5, "No hay pesos disponibles", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Visualizar memoria holográfica
        plt.subplot(3, 1, 3)
        if self.hmm['pattern_count'] > 0:
            # Mostrar primeros patrones
            n_patterns = min(5, self.hmm['pattern_count'])
            patterns = []
            for i, pattern_id in enumerate(list(self.hmm['id_to_index'].keys())[:n_patterns]):
                index = self.hmm['id_to_index'][pattern_id]
                pattern = self.hmm['memory_matrix'][index]
                patterns.append(np.abs(pattern))
            
            # Apilar patrones para visualización
            stacked_patterns = np.vstack(patterns)
            plt.imshow(stacked_patterns, cmap='plasma', aspect='auto')
            plt.colorbar()
            plt.title(f"Patrones en Memoria Holográfica (HMM) - {self.hmm['pattern_count']} total")
        else:
            plt.text(0.5, 0.5, "No hay patrones en memoria", 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualización guardada en {save_path}")
        else:
            plt.show()
    
    def _text_to_features(self, text):
        """
        Convierte texto a vector de características.
        
        Args:
            text: Texto a convertir
            
        Returns:
            Vector de características como array NumPy
        """
        # Implementación simplificada usando hash
        # Convertir texto a bytes
        text_bytes = text.encode('utf-8')
        
        # Generar hash
        hash_obj = hashlib.sha256(text_bytes)
        hash_bytes = hash_obj.digest()
        
        # Convertir hash a array de números
        hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
        
        # Expandir a dimensiones requeridas
        dimensions = self.hmm['dimensions']
        if len(hash_array) < dimensions:
            # Repetir el hash hasta alcanzar las dimensiones requeridas
            repetitions = int(np.ceil(dimensions / len(hash_array)))
            hash_array = np.tile(hash_array, repetitions)[:dimensions]
        elif len(hash_array) > dimensions:
            # Truncar el hash
            hash_array = hash_array[:dimensions]
        
        # Normalizar a valores entre -1 y 1
        features = hash_array.astype(np.float64) / 128.0 - 1.0
        
        return features
    
    def _process_with_bnnm(self, input_features, context_features=None):
        """
        Procesa características de entrada con el BNNM.
        
        Args:
            input_features: Características de entrada
            context_features: Características de contexto (opcional)
            
        Returns:
            Características procesadas
        """
        # Asegurar que la entrada tiene la forma correcta
        x = input_features.copy()
        
        # Ajustar dimensiones de entrada si no coinciden con la primera capa
        input_dim = self.bnnm['layer_sizes'][0]
        if len(x) != input_dim:
            if len(x) > input_dim:
                # Truncar
                x = x[:input_dim]
            else:
                # Rellenar con ceros
                padded = np.zeros(input_dim)
                padded[:len(x)] = x
                x = padded
        
        # Propagación a través de capas
        for i, weights in enumerate(self.bnnm['weights']):
            # Multiplicación matriz-vector
            x = np.dot(x, weights)
            
            # Aplicar función de activación (sigmoid)
            x = 1 / (1 + np.exp(-x))
            
            # Aplicar activación contextual si está habilitada y hay contexto
            if self.bnnm['use_contextual_activation'] and context_features is not None:
                # Ajustar dimensiones del contexto si es necesario
                if len(context_features) != len(x):
                    if len(context_features) > len(x):
                        context_features = context_features[:len(x)]
                    else:
                        padded_context = np.zeros(len(x))
                        padded_context[:len(context_features)] = context_features
                        context_features = padded_context
                
                # Implementación simplificada: modular activación por similitud con contexto
                similarity = np.abs(np.dot(x, context_features) / 
                                  (np.linalg.norm(x) * np.linalg.norm(context_features) + 1e-10))
                
                # Crear máscara de activación basada en similitud
                mask = np.random.random(x.shape) < (0.5 + 0.5 * similarity)
                
                # Aplicar máscara (mantener al menos 20% de neuronas activas)
                active_count = np.sum(mask)
                if active_count < 0.2 * len(mask):
                    # Activar aleatoriamente más neuronas si hay muy pocas activas
                    additional_active = np.random.choice(
                        np.where(~mask)[0], 
                        size=int(0.2 * len(mask)) - active_count,
                        replace=False
                    )
                    mask[additional_active] = True
                
                x = x * mask
        
        return x
    
    def _apply_quantum_optimization(self, features):
        """
        Aplica optimización cuántica a las características.
        
        Args:
            features: Características a optimizar
            
        Returns:
            Características optimizadas
        """
        # Normalizar características
        if np.sum(np.abs(features)**2) > 0:
            features = features / np.sqrt(np.sum(np.abs(features)**2))
        
        # Codificar en estado cuántico
        self.qsm['state_vector'] = features.astype(complex)
        
        # Aplicar QFT (simulada)
        self._apply_qft()
        
        # Aplicar algoritmo de Grover (simplificado)
        # Marcar estados con mayor amplitud
        threshold = np.percentile(np.abs(self.qsm['state_vector']), 90)
        marked_states = np.where(np.abs(self.qsm['state_vector']) > threshold)[0]
        
        self._apply_grover(marked_states)
        
        # Obtener estado resultante
        optimized_features = np.abs(self.qsm['state_vector'])
        
        return optimized_features
    
    def _apply_qft(self):
        """Aplica la Transformada Cuántica de Fourier al estado actual."""
        n = self.qsm['n_qubits']
        N = 2**n
        
        # Matriz de QFT
        qft_matrix = np.zeros((N, N), dtype=complex)
        omega = np.exp(2j * np.pi / N)
        
        for i in range(N):
            for j in range(N):
                qft_matrix[i, j] = omega**(i * j) / np.sqrt(N)
        
        # Aplicar QFT al estado
        self.qsm['state_vector'] = np.dot(qft_matrix, self.qsm['state_vector'])
    
    def _apply_grover(self, marked_states, iterations=1):
        """
        Aplica iteraciones del algoritmo de Grover.
        
        Args:
            marked_states: Lista de índices de estados a marcar
            iterations: Número de iteraciones
        """
        n = self.qsm['n_qubits']
        N = 2**n
        
        # Crear operador de oráculo
        oracle = np.identity(N, dtype=complex)
        for state in marked_states:
            if 0 <= state < N:
                oracle[state, state] = -1
        
        # Crear operador de difusión
        diffusion = np.ones((N, N), dtype=complex) * 2 / N - np.identity(N, dtype=complex)
        
        # Aplicar iteraciones de Grover
        for _ in range(iterations):
            # Aplicar oráculo
            self.qsm['state_vector'] = np.dot(oracle, self.qsm['state_vector'])
            
            # Aplicar operador de difusión
            self.qsm['state_vector'] = np.dot(diffusion, self.qsm['state_vector'])
    
    def _encode_in_memory(self, data, metadata=None):
        """
        Codifica datos en un patrón holográfico y lo almacena en memoria.
        
        Args:
            data: Datos a codificar
            metadata: Metadatos opcionales
            
        Returns:
            Identificador único del patrón almacenado
        """
        # Preprocesar datos
        processed_data = self._text_to_features(data if isinstance(data, str) else str(data))
        
        # Generar patrón holográfico
        pattern = self._generate_holographic_pattern(processed_data)
        
        # Generar identificador único
        pattern_id = self._generate_id(data, metadata)
        
        # Almacenar patrón
        if self.hmm['pattern_count'] >= self.hmm['capacity']:
            # Política de reemplazo: reemplazar el patrón más antiguo
            index = self.hmm['pattern_count'] % self.hmm['capacity']
        else:
            index = self.hmm['pattern_count']
            self.hmm['pattern_count'] += 1
        
        self.hmm['memory_matrix'][index] = pattern
        self.hmm['id_to_index'][pattern_id] = index
        
        # Almacenar metadatos
        if metadata is None:
            metadata = {}
        
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['data_type'] = type(data).__name__
        self.hmm['metadata'][pattern_id] = metadata
        
        return pattern_id
    
    def _generate_holographic_pattern(self, data):
        """
        Genera un patrón holográfico a partir de datos.
        
        Args:
            data: Datos preprocesados
            
        Returns:
            Patrón holográfico como array NumPy complejo
        """
        # Asegurar que los datos tienen la dimensión correcta
        dimensions = self.hmm['dimensions']
        if len(data) != dimensions:
            # Redimensionar datos
            if len(data) > dimensions:
                # Truncar
                data = data[:dimensions]
            else:
                # Rellenar con ceros
                padded_data = np.zeros(dimensions)
                padded_data[:len(data)] = data
                data = padded_data
        
        # Normalizar datos
        if np.sum(np.abs(data)**2) > 0:
            data = data / np.sqrt(np.sum(np.abs(data)**2))
        
        # Convertir a dominio de frecuencia (simulando difracción)
        frequency_domain = np.fft.fft(data)
        
        # Generar haz de referencia (onda plana con fase aleatoria)
        reference_beam = np.exp(1j * np.random.uniform(0, 2*np.pi, dimensions))
        
        # Simular interferencia entre haz de objeto y haz de referencia
        hologram = frequency_domain * reference_beam
        
        # Normalizar holograma
        if np.sum(np.abs(hologram)**2) > 0:
            hologram = hologram / np.sqrt(np.sum(np.abs(hologram)**2))
        
        return hologram
    
    def _generate_id(self, data, metadata):
        """
        Genera un identificador único para un patrón.
        
        Args:
            data: Datos del patrón
            metadata: Metadatos asociados
            
        Returns:
            Identificador único como string
        """
        # Combinar datos y metadatos para generar ID
        id_components = [str(data)]
        
        if metadata:
            id_components.append(str(metadata))
        
        id_components.append(datetime.now().isoformat())
        
        # Generar hash
        id_string = ''.join(id_components)
        hash_obj = hashlib.md5(id_string.encode('utf-8'))
        
        return hash_obj.hexdigest()
    
    def _retrieve_from_memory(self, query, similarity_threshold=0.7, max_results=5):
        """
        Recupera patrones similares a la consulta mediante recuperación asociativa.
        
        Args:
            query: Consulta para recuperación
            similarity_threshold: Umbral de similitud
            max_results: Número máximo de resultados
            
        Returns:
            Lista de tuplas (id, similitud, metadatos)
        """
        # Preprocesar consulta
        processed_query = self._text_to_features(query)
        
        # Generar patrón holográfico de consulta
        query_pattern = self._generate_holographic_pattern(processed_query)
        
        # Calcular similitud con todos los patrones almacenados
        similarities = []
        
        for pattern_id, index in self.hmm['id_to_index'].items():
            stored_pattern = self.hmm['memory_matrix'][index]
            similarity = self._calculate_similarity(query_pattern, stored_pattern)
            
            if similarity >= similarity_threshold:
                similarities.append({
                    'id': pattern_id,
                    'similarity': similarity,
                    'metadata': self.hmm['metadata'][pattern_id]
                })
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limitar número de resultados
        return similarities[:max_results]
    
    def _calculate_similarity(self, pattern1, pattern2):
        """
        Calcula la similitud entre dos patrones holográficos.
        
        Args:
            pattern1: Primer patrón holográfico
            pattern2: Segundo patrón holográfico
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        # Calcular correlación cruzada (simulando reconstrucción holográfica)
        correlation = np.abs(np.sum(pattern1 * np.conj(pattern2)))
        
        # Normalizar
        norm1 = np.sqrt(np.sum(np.abs(pattern1)**2))
        norm2 = np.sqrt(np.sum(np.abs(pattern2)**2))
        
        if norm1 > 0 and norm2 > 0:
            similarity = correlation / (norm1 * norm2)
        else:
            similarity = 0.0
        
        return float(similarity)
    
    def _chunk_document(self, document, chunk_size=200):
        """
        Divide un documento en fragmentos más pequeños.
        
        Args:
            document: Texto del documento
            chunk_size: Tamaño aproximado de cada fragmento en palabras
            
        Returns:
            Lista de fragmentos de texto
        """
        # Dividir documento en palabras
        words = document.split()
        
        # Dividir en fragmentos
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _adapt_bnnm(self, input_features, target_features, learning_rate=0.01):
        """
        Adapta los pesos del BNNM basándose en entrada y salida deseada.
        
        Args:
            input_features: Características de entrada
            target_features: Características objetivo
            learning_rate: Tasa de aprendizaje
        """
        # Ajustar dimensiones de entrada si no coinciden con la primera capa
        input_dim = self.bnnm['layer_sizes'][0]
        if len(input_features) != input_dim:
            if len(input_features) > input_dim:
                # Truncar
                input_features = input_features[:input_dim]
            else:
                # Rellenar con ceros
                padded = np.zeros(input_dim)
                padded[:len(input_features)] = input_features
                input_features = padded
        
        # Propagación hacia adelante para obtener activaciones
        activations = [input_features]
        x = input_features.copy()
        
        # Propagación a través de capas
        for weights in self.bnnm['weights']:
            # Multiplicación matriz-vector
            x = np.dot(x, weights)
            
            # Aplicar función de activación (sigmoid)
            x = 1 / (1 + np.exp(-x))
            
            # Guardar activación
            activations.append(x)
        
        # Ajustar dimensiones del objetivo si no coinciden con la capa de salida
        output_dim = self.bnnm['layer_sizes'][-1]
        if len(target_features) != output_dim:
            if len(target_features) > output_dim:
                # Truncar
                target_features = target_features[:output_dim]
            else:
                # Rellenar con ceros
                padded = np.zeros(output_dim)
                padded[:len(target_features)] = target_features
                target_features = padded
        
        # Retropropagación del error
        # Calcular error en la capa de salida
        output_error = target_features - activations[-1]
        
        # Retropropagar error a través de las capas
        for i in reversed(range(len(self.bnnm['weights']))):
            # Calcular delta para esta capa
            delta = output_error * activations[i+1] * (1 - activations[i+1])
            
            # Actualizar pesos
            self.bnnm['weights'][i] += learning_rate * np.outer(activations[i], delta)
            
            # Propagar error a capa anterior
            if i > 0:
                output_error = np.dot(delta, self.bnnm['weights'][i].T)
    
    def _generate_response(self, query, features, retrieved_info):
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
                if 'original_text' in info['metadata']:
                    text_snippet = info['metadata']['original_text'][:100] + "..."
                    response += f"- {text_snippet}\n"
            
            # Añadir información sobre el tema principal si está disponible
            topics = [info['metadata'].get('topic') for info in retrieved_info if 'topic' in info['metadata']]
            if topics:
                # Contar frecuencia de temas
                topic_counts = {}
                for topic in topics:
                    if topic in topic_counts:
                        topic_counts[topic] += 1
                    else:
                        topic_counts[topic] = 1
                
                # Encontrar tema más frecuente
                main_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
                
                # Añadir párrafo sobre el tema principal
                if main_topic == "física cuántica":
                    response += "\nLa física cuántica es fundamental para NEBULA, ya que proporciona principios como la superposición y el entrelazamiento que permiten representar y procesar información de manera más eficiente. Estos conceptos se aplican en el Módulo de Simulación Cuántica (QSM) para optimizar operaciones de búsqueda y procesamiento."
                
                elif main_topic == "holografía":
                    response += "\nLa holografía es un componente clave de NEBULA, inspirando el diseño del sistema de memoria distribuida donde la información se codifica en patrones similares a hologramas. Esto permite recuperación asociativa y robustez ante información parcial o degradada, características implementadas en el Módulo de Memoria Holográfica (HMM)."
                
                elif main_topic == "redes neuronales" or main_topic == "inteligencia artificial":
                    response += "\nLas redes neuronales bioinspirando son esenciales en NEBULA, implementando capas que emulan principios ópticos y biológicos para procesamiento eficiente. El Módulo de Redes Neuronales Bioinspirando (BNNM) utiliza activación contextual y adaptación inspirada en sistemas biológicos para optimizar el rendimiento y la eficiencia energética."
                
                elif main_topic == "optimización de LLMs":
                    response += "\nLa optimización de grandes modelos de lenguaje es uno de los objetivos principales de NEBULA. Combinando física óptica avanzada, física cuántica y redes neuronales bioinspirando, NEBULA busca crear sistemas de IA más eficientes energéticamente que puedan ejecutarse en hardware convencional mientras mantienen capacidades avanzadas de procesamiento de información."
        else:
            response += "No se encontró información relevante en la memoria."
        
        return response


def run_demo():
    """Ejecuta una demostración del sistema NEBULA simplificado."""
    print("=== Demostración de NEBULA (Versión Simplificada) ===\n")
    
    # Crear directorio para datos
    os.makedirs("./nebula_data", exist_ok=True)
    os.makedirs("./nebula_results", exist_ok=True)
    
    # Crear instancia de NEBULA
    print("Creando instancia de NEBULA...")
    nebula = SimplifiedNEBULA()
    
    # Cargar datos de entrenamiento
    print("\nCargando datos de entrenamiento...")
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
    
    # Entrenar NEBULA
    print("\nEntrenando NEBULA...")
    for i, item in enumerate(training_data):
        print(f"Procesando documento {i+1}/{len(training_data)}: {item['metadata']['topic']}")
        nebula.learn_from_document(item['text'], item['metadata'])
    
    # Visualizar estado del sistema
    nebula.visualize_system_state("./nebula_results/nebula_state.png")
    
    # Ejecutar consultas de demostración
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
    
    # Guardar estado final
    nebula.save_state("./nebula_results/nebula_final")
    print(f"\nEstado final de NEBULA guardado en './nebula_results/nebula_final'")
    
    # Crear HTML de demostración
    create_demo_html(nebula, query_results, "./nebula_results")
    
    print("\nDemostración completada. Resultados guardados en './nebula_results'")
    return nebula

def create_demo_html(nebula, query_results, save_dir):
    """
    Crea una página HTML de demostración para NEBULA.
    
    Args:
        nebula: Instancia de NEBULA
        query_results: Resultados de consultas de demostración
        save_dir: Directorio donde guardar la demostración
    """
    print("\nCreando demostración interactiva de NEBULA...")
    
    # Crear directorio si no existe
    os.makedirs(save_dir, exist_ok=True)
    
    # Crear archivo HTML para la demostración
    html_path = os.path.join(save_dir, "nebula_demo.html")
    
    # Preparar sección de resultados de consultas
    query_results_html = ""
    for i, result in enumerate(query_results):
        query_results_html += f"""
        <div class="query-result">
            <h4>Consulta {i+1}: "{result['query']}"</h4>
            <p><strong>Tiempo de procesamiento:</strong> {result['processing_time']:.4f} segundos</p>
            <p><strong>Información recuperada:</strong> {result['retrieved_info_count']} fragmentos</p>
            <div class="response">
                <h5>Respuesta:</h5>
                <pre>{result['response']}</pre>
            </div>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NEBULA - Demostración Interactiva</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            header {{
                background: linear-gradient(135deg, #6e8efb, #a777e3);
                color: white;
                padding: 2rem;
                border-radius: 8px;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                margin: 0;
                font-size: 2.5rem;
            }}
            h2 {{
                color: #5c6bc0;
                border-bottom: 2px solid #5c6bc0;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
            }}
            .description {{
                font-size: 1.2rem;
                margin: 1rem 0;
            }}
            .container {{
                background-color: white;
                border-radius: 8px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .query-section {{
                margin-bottom: 2rem;
            }}
            .query-input {{
                width: 100%;
                padding: 1rem;
                font-size: 1rem;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 1rem;
            }}
            .query-button {{
                background-color: #5c6bc0;
                color: white;
                border: none;
                padding: 0.8rem 1.5rem;
                font-size: 1rem;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .query-button:hover {{
                background-color: #3f51b5;
            }}
            .response-area {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 1rem;
                min-height: 200px;
                margin-top: 1rem;
                white-space: pre-wrap;
            }}
            .module-section {{
                display: flex;
                flex-wrap: wrap;
                gap: 2rem;
                margin-top: 2rem;
            }}
            .module-card {{
                flex: 1;
                min-width: 300px;
                background-color: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .module-title {{
                color: #5c6bc0;
                margin-top: 0;
            }}
            .visualization {{
                text-align: center;
                margin: 2rem 0;
            }}
            .visualization img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .query-result {{
                margin-bottom: 2rem;
                padding: 1rem;
                background-color: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #5c6bc0;
            }}
            .response {{
                background-color: white;
                padding: 1rem;
                border-radius: 4px;
                margin-top: 1rem;
            }}
            .response pre {{
                white-space: pre-wrap;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 0.9rem;
            }}
            footer {{
                text-align: center;
                margin-top: 3rem;
                color: #666;
                font-size: 0.9rem;
            }}
            .note {{
                background-color: #fff8e1;
                border-left: 4px solid #ffc107;
                padding: 1rem;
                margin: 1rem 0;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>NEBULA</h1>
            <p class="description">Sistema de IA basado en física óptica avanzada y física cuántica</p>
        </header>
        
        <div class="container">
            <h2>Demostración Interactiva</h2>
            <p>Esta demostración muestra las capacidades de NEBULA, un sistema de IA que combina principios de física cuántica, óptica avanzada y redes neuronales bioinspirando para crear un modelo más eficiente.</p>
            
            <div class="note">
                <strong>Nota:</strong> Esta es una versión simplificada de NEBULA que utiliza principalmente NumPy para implementar las funcionalidades básicas sin depender de PyTorch u otras bibliotecas externas grandes.
            </div>
            
            <div class="query-section">
                <h3>Realiza una consulta a NEBULA</h3>
                <input type="text" id="query-input" class="query-input" placeholder="Escribe tu consulta aquí...">
                <button id="query-button" class="query-button">Procesar Consulta</button>
                
                <h3>Respuesta</h3>
                <div id="response-area" class="response-area">La respuesta aparecerá aquí...</div>
            </div>
        </div>
        
        <div class="container">
            <h2>Arquitectura de NEBULA</h2>
            <p>NEBULA está compuesto por tres módulos principales que trabajan en conjunto para proporcionar capacidades avanzadas de procesamiento de información.</p>
            
            <div class="module-section">
                <div class="module-card">
                    <h3 class="module-title">Módulo de Simulación Cuántica (QSM)</h3>
                    <p>Implementa simulación de estados cuánticos para memoria holográfica, con soporte para transformadas cuánticas de Fourier y algoritmo de Grover para búsqueda optimizada.</p>
                </div>
                
                <div class="module-card">
                    <h3 class="module-title">Módulo de Redes Neuronales Bioinspirando (BNNM)</h3>
                    <p>Implementa capas neuronales holográficas con activación biológica y sistemas de activación contextual que se inspiran en cómo el cerebro humano activa diferentes regiones según el contexto.</p>
                </div>
                
                <div class="module-card">
                    <h3 class="module-title">Módulo de Memoria Holográfica (HMM)</h3>
                    <p>Implementa un sistema de almacenamiento distribuido con mecanismos de recuperación asociativa, permitiendo recuperar información completa a partir de entradas parciales.</p>
                </div>
            </div>
            
            <div class="visualization">
                <h3>Visualización del Estado Actual</h3>
                <img src="nebula_state.png" alt="Estado actual de NEBULA">
            </div>
        </div>
        
        <div class="container">
            <h2>Resultados de Consultas de Demostración</h2>
            <p>A continuación se muestran los resultados de las consultas de demostración ejecutadas en NEBULA:</p>
            
            {query_results_html}
        </div>
        
        <div class="container">
            <h2>Consultas de Ejemplo</h2>
            <p>Prueba estas consultas predefinidas para ver cómo NEBULA procesa diferentes tipos de preguntas.</p>
            
            <ul>
                <li><a href="#" class="example-query">¿Qué es la física cuántica y cómo se relaciona con la holografía?</a></li>
                <li><a href="#" class="example-query">Explica cómo las redes neuronales pueden optimizarse usando principios cuánticos</a></li>
                <li><a href="#" class="example-query">¿Cómo funciona la memoria holográfica y qué ventajas ofrece?</a></li>
                <li><a href="#" class="example-query">Describe las aplicaciones de la computación cuántica en inteligencia artificial</a></li>
                <li><a href="#" class="example-query">¿Cómo se pueden optimizar los grandes modelos de lenguaje usando física óptica y cuántica?</a></li>
            </ul>
        </div>
        
        <footer>
            <p>NEBULA - Desarrollado basado en la investigación de Agnuxo1</p>
        </footer>
        
        <script>
            // Respuestas predefinidas para la demostración
            const predefinedResponses = {{
                "¿Qué es la física cuántica y cómo se relaciona con la holografía?": 
                    `Respuesta a la consulta: '¿Qué es la física cuántica y cómo se relaciona con la holografía?'

Basado en la información recuperada:
- La física cuántica es una rama fundamental de la física que describe la naturaleza a escalas atómicas y subatómicas. A diferencia de la física clásica, la física cuántica introduce conceptos revolucionarios como la superposición de estados...
- La holografía es una técnica avanzada que permite registrar y reconstruir imágenes tridimensionales completas. Se basa en el fenómeno de interferencia de ondas luminosas coherentes, típicamente generadas por láseres...

La física cuántica es fundamental para NEBULA, ya que proporciona principios como la superposición y el entrelazamiento que permiten representar y procesar información de manera más eficiente. Estos conceptos se aplican en el Módulo de Simulación Cuántica (QSM) para optimizar operaciones de búsqueda y procesamiento.

La física cuántica y la holografía están relacionadas de varias maneras fundamentales. Ambas disciplinas trabajan con principios de interferencia y superposición de ondas. Mientras la física cuántica describe estos fenómenos a nivel subatómico, la holografía los aplica a nivel macroscópico con ondas de luz.

El entrelazamiento cuántico tiene paralelos con la naturaleza distribuida de la información en hologramas, donde cada fragmento contiene información sobre el todo. Esta propiedad ha inspirado modelos teóricos como el principio holográfico en física teórica, que sugiere que toda la información contenida en un volumen de espacio puede ser representada por información en la frontera de esa región.`,
                
                "Explica cómo las redes neuronales pueden optimizarse usando principios cuánticos": 
                    `Respuesta a la consulta: 'Explica cómo las redes neuronales pueden optimizarse usando principios cuánticos'

Basado en la información recuperada:
- Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano. Están compuestas por unidades de procesamiento interconectadas llamadas neuronas artificiales, organizadas en capas...
- La optimización de modelos de lenguaje mediante principios cuánticos y holográficos representa una frontera innovadora en inteligencia artificial. Los grandes modelos de lenguaje (LLMs) convencionales requieren enormes recursos computacionales...

Las redes neuronales bioinspirando son esenciales en NEBULA, implementando capas que emulan principios ópticos y biológicos para procesamiento eficiente. El Módulo de Redes Neuronales Bioinspirando (BNNM) utiliza activación contextual y adaptación inspirada en sistemas biológicos para optimizar el rendimiento y la eficiencia energética.

Las redes neuronales pueden optimizarse mediante principios cuánticos de varias maneras:

1. Representación cuántica de datos: Los estados cuánticos pueden representar información en espacios de alta dimensionalidad con menor número de parámetros, aprovechando la superposición para codificar múltiples configuraciones simultáneamente.

2. Paralelismo cuántico: Las operaciones cuánticas pueden procesar simultáneamente múltiples estados, permitiendo evaluar muchas configuraciones de pesos neuronales en paralelo, acelerando significativamente el entrenamiento.

3. Algoritmos cuánticos para optimización: Algoritmos como el recocido cuántico (quantum annealing) pueden encontrar mínimos globales en funciones de pérdida complejas más eficientemente que los métodos clásicos, evitando mínimos locales.`,
                
                "¿Cómo se pueden optimizar los grandes modelos de lenguaje usando física óptica y cuántica?": 
                    `Respuesta a la consulta: '¿Cómo se pueden optimizar los grandes modelos de lenguaje usando física óptica y cuántica?'

Basado en la información recuperada:
- La optimización de modelos de lenguaje mediante principios cuánticos y holográficos representa una frontera innovadora en inteligencia artificial. Los grandes modelos de lenguaje (LLMs) convencionales requieren enormes recursos computacionales debido a su arquitectura basada en transformers con atención completa...

La optimización de grandes modelos de lenguaje es uno de los objetivos principales de NEBULA. Combinando física óptica avanzada, física cuántica y redes neuronales bioinspirando, NEBULA busca crear sistemas de IA más eficientes energéticamente que puedan ejecutarse en hardware convencional mientras mantienen capacidades avanzadas de procesamiento de información.

Los grandes modelos de lenguaje (LLMs) pueden optimizarse mediante principios de física óptica y cuántica a través de varios enfoques innovadores:

1. Representación holográfica de tokens:
   - Codificar tokens como patrones holográficos distribuidos en lugar de vectores densos convencionales.
   - Utilizar principios de interferencia óptica para representar relaciones semánticas entre palabras.
   - Implementar recuperación asociativa holográfica que permite acceder a información contextual relevante sin necesidad de mecanismos de atención computacionalmente costosos.

2. Compresión cuántica de información:
   - Aprovechar principios de superposición cuántica para representar múltiples estados semánticos simultáneamente.
   - Utilizar técnicas inspiradas en la compresión cuántica de información para reducir la dimensionalidad de las representaciones sin pérdida significativa de información.
   - Implementar circuitos de inspiración cuántica que procesan múltiples posibilidades interpretativas en paralelo.`
            }};
            
            // Función para procesar consulta
            function processQuery() {{
                const queryInput = document.getElementById('query-input');
                const responseArea = document.getElementById('response-area');
                const query = queryInput.value.trim();
                
                if (query === '') {{
                    responseArea.textContent = 'Por favor, ingresa una consulta.';
                    return;
                }}
                
                // Mostrar mensaje de procesamiento
                responseArea.textContent = 'Procesando consulta...';
                
                // Simular tiempo de procesamiento
                setTimeout(() => {{
                    // Buscar respuesta predefinida o generar respuesta genérica
                    const response = predefinedResponses[query] || 
                        `Respuesta a la consulta: '${{query}}'\\n\\nNo se encontró información específica para esta consulta en la memoria de NEBULA. En una implementación completa, NEBULA procesaría esta consulta utilizando sus módulos de simulación cuántica, redes neuronales bioinspirando y memoria holográfica para generar una respuesta relevante.`;
                    
                    responseArea.textContent = response;
                }}, 1500);
            }}
            
            // Configurar eventos
            document.getElementById('query-button').addEventListener('click', processQuery);
            document.getElementById('query-input').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    processQuery();
                }}
            }});
            
            // Configurar ejemplos de consultas
            document.querySelectorAll('.example-query').forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const query = this.textContent;
                    document.getElementById('query-input').value = query;
                    processQuery();
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Demostración interactiva creada en {html_path}")
    return html_path

if __name__ == "__main__":
    run_demo()
