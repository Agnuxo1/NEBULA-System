"""
Módulo de Memoria Holográfica (HMM) para NEBULA

Este módulo implementa un sistema de almacenamiento distribuido con mecanismos de recuperación
asociativa, inspirado en principios holográficos para almacenamiento y recuperación eficiente
de información.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
import hashlib
from datetime import datetime

class HolographicMemory:
    """
    Sistema de memoria holográfica que implementa almacenamiento distribuido y
    recuperación asociativa de información basada en principios holográficos.
    """
    
    def __init__(self, dimensions: int = 1024, capacity: int = 10000, 
                 storage_path: Optional[str] = None):
        """
        Inicializa el sistema de memoria holográfica.
        
        Args:
            dimensions: Dimensionalidad de los patrones holográficos
            capacity: Capacidad máxima de almacenamiento (número de patrones)
            storage_path: Ruta para almacenamiento persistente (opcional)
        """
        self.dimensions = dimensions
        self.capacity = capacity
        self.storage_path = storage_path
        
        # Inicializar matriz de memoria holográfica
        self.memory_matrix = np.zeros((capacity, dimensions), dtype=np.complex128)
        
        # Contador de patrones almacenados
        self.pattern_count = 0
        
        # Diccionario para mapear identificadores a índices
        self.id_to_index = {}
        
        # Metadatos para cada patrón
        self.metadata = {}
        
        # Crear directorio de almacenamiento si no existe
        if storage_path is not None:
            os.makedirs(storage_path, exist_ok=True)
    
    def encode(self, data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Codifica datos en un patrón holográfico y lo almacena en memoria.
        
        Args:
            data: Datos a codificar (texto, vector, imagen, etc.)
            metadata: Metadatos opcionales asociados con los datos
            
        Returns:
            Identificador único del patrón almacenado
        """
        # Preprocesar datos según su tipo
        processed_data = self._preprocess_data(data)
        
        # Generar patrón holográfico
        pattern = self._generate_holographic_pattern(processed_data)
        
        # Generar identificador único
        pattern_id = self._generate_id(data, metadata)
        
        # Almacenar patrón
        if self.pattern_count >= self.capacity:
            # Política de reemplazo: reemplazar el patrón más antiguo
            # En una implementación más sofisticada, se podría usar una política
            # basada en frecuencia de uso, importancia, etc.
            index = self.pattern_count % self.capacity
        else:
            index = self.pattern_count
            self.pattern_count += 1
        
        self.memory_matrix[index] = pattern
        self.id_to_index[pattern_id] = index
        
        # Almacenar metadatos
        if metadata is None:
            metadata = {}
        
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['data_type'] = type(data).__name__
        self.metadata[pattern_id] = metadata
        
        # Guardar en almacenamiento persistente si está configurado
        if self.storage_path is not None:
            self._save_pattern(pattern_id, pattern, metadata)
        
        return pattern_id
    
    def retrieve(self, query: Any, similarity_threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """
        Recupera patrones similares a la consulta mediante recuperación asociativa.
        
        Args:
            query: Consulta para recuperación (puede ser un patrón parcial, ID, o datos similares)
            similarity_threshold: Umbral de similitud para incluir resultados
            
        Returns:
            Lista de tuplas (id, similitud, metadatos) ordenadas por similitud descendente
        """
        # Manejar caso especial: si la consulta es un ID existente
        if isinstance(query, str) and query in self.id_to_index:
            index = self.id_to_index[query]
            pattern = self.memory_matrix[index]
            return [(query, 1.0, self.metadata[query])]
        
        # Preprocesar consulta
        processed_query = self._preprocess_data(query)
        
        # Generar patrón holográfico de consulta
        query_pattern = self._generate_holographic_pattern(processed_query)
        
        # Calcular similitud con todos los patrones almacenados
        similarities = []
        
        for pattern_id, index in self.id_to_index.items():
            stored_pattern = self.memory_matrix[index]
            similarity = self._calculate_similarity(query_pattern, stored_pattern)
            
            if similarity >= similarity_threshold:
                similarities.append((pattern_id, similarity, self.metadata[pattern_id]))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def update(self, pattern_id: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """
        Actualiza un patrón existente con nuevos datos o metadatos.
        
        Args:
            pattern_id: Identificador del patrón a actualizar
            data: Nuevos datos para el patrón
            metadata: Nuevos metadatos (se fusionarán con los existentes)
            
        Returns:
            True si la actualización fue exitosa, False en caso contrario
        """
        if pattern_id not in self.id_to_index:
            return False
        
        # Obtener índice del patrón
        index = self.id_to_index[pattern_id]
        
        # Preprocesar nuevos datos
        processed_data = self._preprocess_data(data)
        
        # Generar nuevo patrón holográfico
        new_pattern = self._generate_holographic_pattern(processed_data)
        
        # Actualizar patrón en memoria
        self.memory_matrix[index] = new_pattern
        
        # Actualizar metadatos
        if metadata is not None:
            # Preservar timestamp original
            original_timestamp = self.metadata[pattern_id].get('timestamp')
            
            # Fusionar metadatos
            self.metadata[pattern_id].update(metadata)
            
            # Añadir timestamp de actualización
            self.metadata[pattern_id]['updated_timestamp'] = datetime.now().isoformat()
            
            # Restaurar timestamp original
            if original_timestamp:
                self.metadata[pattern_id]['original_timestamp'] = original_timestamp
        
        # Guardar en almacenamiento persistente si está configurado
        if self.storage_path is not None:
            self._save_pattern(pattern_id, new_pattern, self.metadata[pattern_id])
        
        return True
    
    def delete(self, pattern_id: str) -> bool:
        """
        Elimina un patrón de la memoria holográfica.
        
        Args:
            pattern_id: Identificador del patrón a eliminar
            
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        if pattern_id not in self.id_to_index:
            return False
        
        # Obtener índice del patrón
        index = self.id_to_index[pattern_id]
        
        # Eliminar patrón de la memoria
        self.memory_matrix[index] = np.zeros(self.dimensions, dtype=np.complex128)
        
        # Eliminar metadatos
        del self.metadata[pattern_id]
        del self.id_to_index[pattern_id]
        
        # Eliminar del almacenamiento persistente si está configurado
        if self.storage_path is not None:
            pattern_path = os.path.join(self.storage_path, f"{pattern_id}.npz")
            if os.path.exists(pattern_path):
                os.remove(pattern_path)
        
        return True
    
    def clear(self) -> None:
        """Limpia toda la memoria holográfica."""
        self.memory_matrix = np.zeros((self.capacity, self.dimensions), dtype=np.complex128)
        self.pattern_count = 0
        self.id_to_index = {}
        self.metadata = {}
        
        # Limpiar almacenamiento persistente si está configurado
        if self.storage_path is not None:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.npz'):
                    os.remove(os.path.join(self.storage_path, filename))
    
    def save_state(self, filepath: str) -> None:
        """
        Guarda el estado completo de la memoria holográfica.
        
        Args:
            filepath: Ruta del archivo donde guardar el estado
        """
        state = {
            'dimensions': self.dimensions,
            'capacity': self.capacity,
            'pattern_count': self.pattern_count,
            'memory_matrix': self.memory_matrix.tolist(),
            'id_to_index': self.id_to_index,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_state(self, filepath: str) -> None:
        """
        Carga el estado completo de la memoria holográfica.
        
        Args:
            filepath: Ruta del archivo desde donde cargar el estado
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.dimensions = state['dimensions']
        self.capacity = state['capacity']
        self.pattern_count = state['pattern_count']
        self.memory_matrix = np.array(state['memory_matrix'], dtype=np.complex128)
        self.id_to_index = state['id_to_index']
        self.metadata = state['metadata']
    
    def visualize_memory(self, n_patterns: int = 10, title: str = "Memoria Holográfica") -> None:
        """
        Visualiza los patrones almacenados en la memoria holográfica.
        
        Args:
            n_patterns: Número de patrones a visualizar
            title: Título para la visualización
        """
        n_patterns = min(n_patterns, self.pattern_count)
        
        if n_patterns == 0:
            print("No hay patrones para visualizar.")
            return
        
        plt.figure(figsize=(15, n_patterns * 2))
        plt.suptitle(title, fontsize=16)
        
        for i, pattern_id in enumerate(list(self.id_to_index.keys())[:n_patterns]):
            index = self.id_to_index[pattern_id]
            pattern = self.memory_matrix[index]
            
            plt.subplot(n_patterns, 2, 2*i + 1)
            plt.plot(np.abs(pattern))
            plt.title(f"Patrón {pattern_id} - Magnitud")
            
            plt.subplot(n_patterns, 2, 2*i + 2)
            plt.plot(np.angle(pattern))
            plt.title(f"Patrón {pattern_id} - Fase")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
        plt.show()
    
    def _preprocess_data(self, data: Any) -> np.ndarray:
        """
        Preprocesa datos para codificación holográfica según su tipo.
        
        Args:
            data: Datos a preprocesar
            
        Returns:
            Array NumPy preprocesado
        """
        if isinstance(data, np.ndarray):
            # Ya es un array NumPy
            return data
        elif isinstance(data, torch.Tensor):
            # Convertir tensor PyTorch a NumPy
            return data.detach().cpu().numpy()
        elif isinstance(data, str):
            # Texto: convertir a vector de características
            return self._text_to_features(data)
        elif isinstance(data, (list, tuple)):
            # Lista o tupla: convertir a array NumPy
            return np.array(data)
        elif isinstance(data, dict):
            # Diccionario: serializar a JSON y tratar como texto
            return self._text_to_features(json.dumps(data))
        elif isinstance(data, (int, float, bool)):
            # Escalar: convertir a array de un elemento
            return np.array([data])
        else:
            # Tipo no soportado: intentar convertir a string
            return self._text_to_features(str(data))
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """
        Convierte texto a vector de características para codificación holográfica.
        
        Args:
            text: Texto a convertir
            
        Returns:
            Vector de características como array NumPy
        """
        # Implementación simplificada: usar hash del texto
        # En una implementación más sofisticada, se usarían técnicas de NLP
        # como word embeddings, TF-IDF, etc.
        
        # Convertir texto a bytes
        text_bytes = text.encode('utf-8')
        
        # Generar hash
        hash_obj = hashlib.sha256(text_bytes)
        hash_bytes = hash_obj.digest()
        
        # Convertir hash a array de números
        hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
        
        # Expandir a dimensiones requeridas
        if len(hash_array) < self.dimensions:
            # Repetir el hash hasta alcanzar las dimensiones requeridas
            repetitions = int(np.ceil(self.dimensions / len(hash_array)))
            hash_array = np.tile(hash_array, repetitions)[:self.dimensions]
        elif len(hash_array) > self.dimensions:
            # Truncar el hash
            hash_array = hash_array[:self.dimensions]
        
        # Normalizar a valores entre -1 y 1
        features = hash_array.astype(np.float64) / 128.0 - 1.0
        
        return features
    
    def _generate_holographic_pattern(self, data: np.ndarray) -> np.ndarray:
        """
        Genera un patrón holográfico a partir de datos preprocesados.
        
        Args:
            data: Datos preprocesados
            
        Returns:
            Patrón holográfico como array NumPy complejo
        """
        # Asegurar que los datos tienen la dimensión correcta
        if len(data) != self.dimensions:
            # Redimensionar datos
            if len(data) > self.dimensions:
                # Truncar
                data = data[:self.dimensions]
            else:
                # Rellenar con ceros
                padded_data = np.zeros(self.dimensions)
                padded_data[:len(data)] = data
                data = padded_data
        
        # Normalizar datos
        if np.sum(np.abs(data)**2) > 0:
            data = data / np.sqrt(np.sum(np.abs(data)**2))
        
        # Convertir a dominio de frecuencia (simulando difracción)
        frequency_domain = np.fft.fft(data)
        
        # Generar haz de referencia (onda plana con fase aleatoria)
        reference_beam = np.exp(1j * np.random.uniform(0, 2*np.pi, self.dimensions))
        
        # Simular interferencia entre haz de objeto y haz de referencia
        hologram = frequency_domain * reference_beam
        
        # Normalizar holograma
        if np.sum(np.abs(hologram)**2) > 0:
            hologram = hologram / np.sqrt(np.sum(np.abs(hologram)**2))
        
        return hologram
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
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
    
    def _generate_id(self, data: Any, metadata: Optional[Dict]) -> str:
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
    
    def _save_pattern(self, pattern_id: str, pattern: np.ndarray, metadata: Dict) -> None:
        """
        Guarda un patrón en almacenamiento persistente.
        
        Args:
            pattern_id: Identificador del patrón
            pattern: Patrón holográfico
            metadata: Metadatos asociados
        """
        if self.storage_path is None:
            return
        
        # Crear archivo para el patrón
        pattern_path = os.path.join(self.storage_path, f"{pattern_id}.npz")
        
        # Guardar patrón y metadatos
        np.savez_compressed(
            pattern_path,
            pattern=pattern,
            metadata=json.dumps(metadata)
        )


class HolographicRAG:
    """
    Sistema de Generación Aumentada por Recuperación (RAG) que utiliza
    memoria holográfica para mejorar la generación de texto.
    """
    
    def __init__(self, holographic_memory: HolographicMemory, 
                 context_window_size: int = 5):
        """
        Inicializa el sistema RAG holográfico.
        
        Args:
            holographic_memory: Instancia de memoria holográfica
            context_window_size: Tamaño de la ventana de contexto para recuperación
        """
        self.memory = holographic_memory
        self.context_window_size = context_window_size
        self.current_context = []
    
    def add_document(self, document: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Añade un documento a la memoria holográfica, dividiéndolo en fragmentos.
        
        Args:
            document: Texto del documento
            metadata: Metadatos asociados al documento
            
        Returns:
            Lista de IDs de los fragmentos almacenados
        """
        # Dividir documento en fragmentos
        chunks = self._chunk_document(document)
        
        # Almacenar cada fragmento en la memoria holográfica
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            # Crear metadatos para el fragmento
            chunk_metadata = {
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_id': metadata.get('document_id') if metadata else None
            }
            
            # Fusionar con metadatos proporcionados
            if metadata:
                chunk_metadata.update(metadata)
            
            # Codificar fragmento
            chunk_id = self.memory.encode(chunk, chunk_metadata)
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def query(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """
        Consulta la memoria holográfica para recuperar información relevante.
        
        Args:
            query_text: Texto de la consulta
            n_results: Número máximo de resultados a devolver
            
        Returns:
            Lista de diccionarios con fragmentos recuperados y metadatos
        """
        # Actualizar contexto
        self._update_context(query_text)
        
        # Recuperar fragmentos similares
        results = self.memory.retrieve(query_text)
        
        # Limitar número de resultados
        results = results[:n_results]
        
        # Formatear resultados
        formatted_results = []
        
        for pattern_id, similarity, metadata in results:
            # Recuperar texto original del fragmento
            # En una implementación real, esto requeriría almacenar el texto
            # original en los metadatos o en un almacenamiento separado
            chunk_text = metadata.get('original_text', f"Fragmento {pattern_id}")
            
            formatted_results.append({
                'id': pattern_id,
                'text': chunk_text,
                'similarity': similarity,
                'metadata': metadata
            })
        
        return formatted_results
    
    def generate_response(self, query_text: str, 
                         context_integration: str = 'prepend') -> str:
        """
        Genera una respuesta a una consulta utilizando información recuperada.
        
        Args:
            query_text: Texto de la consulta
            context_integration: Método de integración del contexto ('prepend', 'append', 'interpolate')
            
        Returns:
            Texto de respuesta generado
        """
        # Recuperar información relevante
        retrieved_info = self.query(query_text)
        
        # Extraer textos de los fragmentos recuperados
        context_texts = [info['text'] for info in retrieved_info]
        
        # Integrar contexto y consulta
        if context_integration == 'prepend':
            # Añadir contexto antes de la consulta
            integrated_text = "\n".join(context_texts) + "\n\n" + query_text
        elif context_integration == 'append':
            # Añadir contexto después de la consulta
            integrated_text = query_text + "\n\n" + "\n".join(context_texts)
        elif context_integration == 'interpolate':
            # Intercalar consulta y contexto
            integrated_text = query_text + "\n\n"
            for i, context in enumerate(context_texts):
                integrated_text += f"Contexto {i+1}: {context}\n\n"
        else:
            integrated_text = query_text
        
        # En una implementación completa, aquí se enviaría el texto integrado
        # a un modelo de lenguaje para generar la respuesta
        # Por ahora, simulamos una respuesta simple
        response = f"Respuesta generada basada en la consulta: '{query_text}' "
        response += f"y {len(retrieved_info)} fragmentos recuperados."
        
        return response
    
    def _chunk_document(self, document: str, chunk_size: int = 200) -> List[str]:
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
    
    def _update_context(self, text: str) -> None:
        """
        Actualiza el contexto actual con nuevo texto.
        
        Args:
            text: Texto a añadir al contexto
        """
        # Añadir texto al contexto
        self.current_context.append(text)
        
        # Mantener tamaño de ventana de contexto
        if len(self.current_context) > self.context_window_size:
            self.current_context = self.current_context[-self.context_window_size:]


# Ejemplo de uso
if __name__ == "__main__":
    # Crear memoria holográfica
    memory = HolographicMemory(dimensions=512, capacity=1000, 
                              storage_path="/tmp/holographic_memory")
    
    # Almacenar algunos datos de ejemplo
    text1 = "La física cuántica es una rama de la física que estudia el comportamiento de la materia a escalas muy pequeñas."
    text2 = "La holografía es una técnica que permite registrar y reconstruir imágenes tridimensionales."
    text3 = "Las redes neuronales son modelos inspirados en el funcionamiento del cerebro humano."
    
    id1 = memory.encode(text1, {"topic": "física"})
    id2 = memory.encode(text2, {"topic": "óptica"})
    id3 = memory.encode(text3, {"topic": "inteligencia artificial"})
    
    print(f"Patrones almacenados con IDs: {id1}, {id2}, {id3}")
    
    # Recuperar información
    query = "física cuántica y holografía"
    results = memory.retrieve(query)
    
    print(f"\nResultados para consulta '{query}':")
    for pattern_id, similarity, metadata in results:
        print(f"ID: {pattern_id}, Similitud: {similarity:.4f}, Metadatos: {metadata}")
    
    # Crear sistema RAG holográfico
    rag = HolographicRAG(memory)
    
    # Añadir documento
    document = """
    NEBULA es un sistema de inteligencia artificial que combina principios de física cuántica,
    óptica avanzada y redes neuronales bioinspirando para crear un modelo más eficiente.
    Utiliza técnicas de ray tracing y holografía para representar y procesar información
    de manera distribuida y paralela. La física cuántica aporta principios de superposición
    y entrelazamiento que potencian la capacidad de procesamiento.
    """
    
    chunk_ids = rag.add_document(document, {"title": "NEBULA: Descripción General"})
    print(f"\nDocumento dividido en {len(chunk_ids)} fragmentos")
    
    # Generar respuesta
    query = "¿Cómo combina NEBULA la física cuántica y la óptica?"
    response = rag.generate_response(query)
    print(f"\nConsulta: {query}")
    print(f"Respuesta: {response}")
    
    # Visualizar memoria
    memory.visualize_memory(n_patterns=3)
