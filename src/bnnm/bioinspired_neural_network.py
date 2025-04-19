"""
Módulo de Redes Neuronales Bioinspirando (BNNM) para NEBULA

Este módulo implementa capas neuronales holográficas utilizando PyTorch con extensiones
personalizadas, inspiradas en principios biológicos y ópticos para procesamiento eficiente.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt

class HolographicLayer(nn.Module):
    """
    Capa neuronal holográfica que implementa transformaciones inspiradas en principios ópticos
    y holográficos, modelando cómo la información se propaga a través de elementos ópticos virtuales.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 adaptation_rate: float = 0.01,
                 holographic_pattern: Optional[torch.Tensor] = None,
                 activation: Optional[Callable] = None):
        """
        Inicializa una capa neuronal holográfica.
        
        Args:
            input_dim: Dimensión de entrada
            output_dim: Dimensión de salida
            adaptation_rate: Tasa de adaptación para el mecanismo de aprendizaje bioinspirando
            holographic_pattern: Patrón holográfico inicial (opcional)
            activation: Función de activación (opcional)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Inicializar patrón holográfico, ya sea aleatorio o predefinido
        if holographic_pattern is None:
            # Inicialización especial para patrones holográficos
            # Usamos distribución uniforme en círculo unitario complejo
            magnitude = torch.sqrt(torch.rand(input_dim, output_dim))
            phase = 2 * np.pi * torch.rand(input_dim, output_dim)
            real_part = magnitude * torch.cos(phase)
            imag_part = magnitude * torch.sin(phase)
            holographic_pattern = torch.complex(real_part, imag_part)
        
        self.holographic_pattern = nn.Parameter(holographic_pattern)
        self.adaptation_rate = nn.Parameter(torch.tensor([adaptation_rate]))
        
        # Memoria bacteriana para almacenar información sobre adaptaciones previas
        self.register_buffer('bacterial_memory', torch.zeros(input_dim, output_dim, dtype=torch.complex64))
        
        # Función de activación
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante utilizando principios de interferencia holográfica.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            
        Returns:
            Tensor de salida [batch_size, output_dim]
        """
        # Convertir entrada a complejo si es necesario
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        
        # Simulación de propagación de luz a través del patrón holográfico
        # Esto es análogo a una multiplicación matriz-vector en redes neuronales estándar,
        # pero modelado como un proceso de interferencia holográfica
        output = torch.matmul(x, self.holographic_pattern)
        
        # Aplicar función de activación si está definida
        if self.activation is not None:
            # Para funciones de activación estándar, usamos solo la parte real
            output_real = self.activation(output.real)
            # Mantenemos la fase original
            output_phase = torch.angle(output)
            output_magnitude = torch.abs(output_real)
            
            # Reconstruir número complejo
            output = torch.polar(output_magnitude, output_phase)
        
        return output
    
    def adapt(self, feedback: torch.Tensor) -> None:
        """
        Implementa el mecanismo de adaptación bacteriana que ajusta el patrón holográfico
        basándose en retroalimentación ambiental.
        
        Args:
            feedback: Tensor con información de retroalimentación para adaptación
        """
        # Actualizar memoria bacteriana con nueva información
        self.bacterial_memory = 0.9 * self.bacterial_memory + 0.1 * feedback
        
        # Calcular ajuste basado en memoria bacteriana
        adjustment = self.adaptation_rate * self.bacterial_memory
        
        # Aplicar ajuste al patrón holográfico
        with torch.no_grad():
            self.holographic_pattern.add_(adjustment)
    
    def visualize_pattern(self, title: str = "Patrón Holográfico") -> None:
        """
        Visualiza el patrón holográfico actual.
        
        Args:
            title: Título para el gráfico
        """
        # Convertir a CPU y NumPy para visualización
        pattern = self.holographic_pattern.detach().cpu().numpy()
        
        # Visualizar magnitud
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(pattern), cmap='viridis')
        plt.colorbar()
        plt.title(f"{title} - Magnitud")
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(pattern), cmap='hsv')
        plt.colorbar()
        plt.title(f"{title} - Fase")
        
        plt.tight_layout()
        plt.show()


class BiologicalActivation(nn.Module):
    """
    Función de activación inspirada en procesos biológicos, que combina
    características de activaciones sigmoidales y comportamiento adaptativo.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, adaptation_rate: float = 0.01):
        """
        Inicializa la función de activación biológica.
        
        Args:
            alpha: Parámetro de forma para la curva de activación
            beta: Parámetro de escala para la curva de activación
            adaptation_rate: Tasa de adaptación para ajuste dinámico
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))
        self.adaptation_rate = adaptation_rate
        self.register_buffer('threshold', torch.tensor([0.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la función de activación biológica.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor activado
        """
        # Función de activación adaptativa
        activation = torch.sigmoid(self.beta * (x - self.threshold))
        
        # Actualizar umbral basado en activación media (simulando adaptación homeostática)
        if self.training:
            with torch.no_grad():
                mean_activation = torch.mean(activation)
                self.threshold += self.adaptation_rate * (mean_activation - 0.5)
        
        return activation ** self.alpha


class ContextualActivation(nn.Module):
    """
    Sistema de activación contextual que controla qué sectores y clústeres neuronales
    se activan en cada momento, basándose en el contexto de la tarea actual.
    """
    
    def __init__(self, total_neurons: int, n_clusters: int = 10, sparsity: float = 0.1):
        """
        Inicializa el sistema de activación contextual.
        
        Args:
            total_neurons: Número total de neuronas en la capa
            n_clusters: Número de clústeres neuronales
            sparsity: Nivel de dispersión (fracción de neuronas activas)
        """
        super().__init__()
        self.total_neurons = total_neurons
        self.n_clusters = n_clusters
        self.sparsity = sparsity
        
        # Inicializar asignaciones de clústeres (cada neurona pertenece a un clúster)
        self.register_buffer('cluster_assignments', 
                            torch.randint(0, n_clusters, (total_neurons,)))
        
        # Inicializar pesos de activación de clústeres
        self.cluster_weights = nn.Parameter(torch.ones(n_clusters))
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aplica activación contextual basada en el contexto de entrada.
        
        Args:
            x: Tensor de entrada [batch_size, total_neurons]
            context: Tensor de contexto opcional [batch_size, context_dim]
            
        Returns:
            Tensor con activación selectiva [batch_size, total_neurons]
        """
        batch_size = x.shape[0]
        
        # Determinar pesos de clústeres basados en contexto
        if context is not None:
            # Proyectar contexto a pesos de clústeres
            context_projection = nn.Linear(context.shape[1], self.n_clusters).to(x.device)
            cluster_activations = torch.sigmoid(context_projection(context))
        else:
            # Usar pesos predeterminados
            cluster_activations = F.softmax(self.cluster_weights, dim=0).expand(batch_size, -1)
        
        # Crear máscara de activación
        mask = torch.zeros(batch_size, self.total_neurons, device=x.device)
        
        for i in range(self.n_clusters):
            # Encontrar neuronas en este clúster
            cluster_neurons = (self.cluster_assignments == i).nonzero().squeeze()
            if cluster_neurons.dim() == 0:  # Si solo hay una neurona
                cluster_neurons = cluster_neurons.unsqueeze(0)
            
            # Número de neuronas a activar en este clúster
            n_active = max(1, int(len(cluster_neurons) * self.sparsity))
            
            # Seleccionar aleatoriamente neuronas para activar
            if len(cluster_neurons) > 0:
                for b in range(batch_size):
                    # Probabilidad de activación basada en peso de clúster
                    activation_prob = cluster_activations[b, i]
                    
                    # Activar neuronas con probabilidad proporcional al peso del clúster
                    active_count = torch.binomial(torch.tensor([len(cluster_neurons)]), 
                                                 torch.tensor([activation_prob * self.sparsity]))
                    active_count = max(1, int(active_count.item()))
                    
                    # Seleccionar neuronas activas
                    perm = torch.randperm(len(cluster_neurons))
                    active_neurons = cluster_neurons[perm[:active_count]]
                    
                    # Establecer máscara
                    mask[b, active_neurons] = 1.0
        
        # Aplicar máscara
        return x * mask


class BioinspiredNeuralNetwork(nn.Module):
    """
    Red neuronal bioinspirando que integra capas holográficas, activación biológica
    y activación contextual para un procesamiento eficiente y adaptativo.
    """
    
    def __init__(self, layer_sizes: List[int], use_contextual_activation: bool = True):
        """
        Inicializa la red neuronal bioinspirando.
        
        Args:
            layer_sizes: Lista con tamaños de cada capa (incluyendo entrada y salida)
            use_contextual_activation: Si se debe usar activación contextual
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.use_contextual_activation = use_contextual_activation
        
        # Crear capas holográficas
        self.holographic_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.holographic_layers.append(
                HolographicLayer(
                    input_dim=layer_sizes[i],
                    output_dim=layer_sizes[i + 1],
                    activation=BiologicalActivation()
                )
            )
        
        # Crear sistema de activación contextual si está habilitado
        if use_contextual_activation:
            self.contextual_activations = nn.ModuleList()
            for size in layer_sizes[1:]:  # Para cada capa excepto la de entrada
                self.contextual_activations.append(
                    ContextualActivation(
                        total_neurons=size,
                        n_clusters=min(10, size // 5),  # Número razonable de clústeres
                        sparsity=0.2
                    )
                )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Propagación hacia adelante a través de la red neuronal bioinspirando.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            context: Tensor de contexto opcional [batch_size, context_dim]
            
        Returns:
            Tensor de salida [batch_size, output_dim]
        """
        # Convertir a tensor de PyTorch si es necesario
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Asegurar que la forma es correcta
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Añadir dimensión de batch
        
        # Propagación a través de capas
        for i, layer in enumerate(self.holographic_layers):
            # Aplicar capa holográfica
            x = layer(x)
            
            # Aplicar activación contextual si está habilitada
            if self.use_contextual_activation and i < len(self.contextual_activations):
                x = self.contextual_activations[i](x, context)
        
        return x
    
    def adapt(self, input_data: torch.Tensor, target: torch.Tensor, 
             context: Optional[torch.Tensor] = None) -> float:
        """
        Adapta la red utilizando mecanismo bioinspirando en lugar de retropropagación estándar.
        
        Args:
            input_data: Datos de entrada para adaptación
            target: Valores objetivo para adaptación
            context: Contexto opcional para adaptación
            
        Returns:
            Error cuadrático medio después de la adaptación
        """
        # Propagación hacia adelante
        output = self.forward(input_data, context)
        
        # Calcular error
        error = target - output
        mse = torch.mean(torch.abs(error) ** 2).item()
        
        # Propagar error hacia atrás para adaptación (no usa gradientes automáticos)
        feedback = error
        
        # Adaptar capas en orden inverso
        for i in range(len(self.holographic_layers) - 1, -1, -1):
            layer = self.holographic_layers[i]
            
            # Calcular retroalimentación para esta capa
            layer_feedback = feedback.clone()
            
            # Adaptar capa
            layer.adapt(layer_feedback)
            
            # Propagar retroalimentación a capa anterior (simplificado)
            if i > 0:
                feedback = torch.matmul(feedback, layer.holographic_pattern.t())
        
        return mse
    
    def visualize_network(self, title: str = "Red Neuronal Bioinspirando") -> None:
        """
        Visualiza la estructura de la red y sus patrones holográficos.
        
        Args:
            title: Título para la visualización
        """
        plt.figure(figsize=(15, 10))
        plt.suptitle(title, fontsize=16)
        
        n_layers = len(self.holographic_layers)
        
        for i, layer in enumerate(self.holographic_layers):
            plt.subplot(n_layers, 1, i + 1)
            
            # Obtener patrón holográfico
            pattern = layer.holographic_pattern.detach().cpu().numpy()
            
            # Visualizar magnitud
            plt.imshow(np.abs(pattern), cmap='viridis')
            plt.colorbar()
            plt.title(f"Capa {i+1}: {layer.input_dim} → {layer.output_dim}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el título principal
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear red neuronal bioinspirando
    network = BioinspiredNeuralNetwork(layer_sizes=[10, 20, 15, 5])
    
    # Datos de ejemplo
    input_data = torch.randn(32, 10)  # 32 muestras, 10 características
    target = torch.randn(32, 5)       # 32 muestras, 5 salidas
    
    # Propagación hacia adelante
    output = network(input_data)
    print(f"Forma de salida: {output.shape}")
    
    # Adaptación
    for epoch in range(10):
        mse = network.adapt(input_data, target)
        print(f"Época {epoch+1}, MSE: {mse:.6f}")
    
    # Visualizar red
    network.visualize_network()
