import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.colors as mcolors

# Configurar el estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Crear figura
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')

# Colores
qsm_color = '#6e8efb'  # Azul
bnnm_color = '#a777e3'  # Púrpura
hmm_color = '#5c6bc0'   # Índigo
input_color = '#4CAF50' # Verde
output_color = '#FF5722' # Naranja

# Definir posiciones
qsm_pos = (0.25, 0.7, 0.2, 0.2)  # x, y, width, height
bnnm_pos = (0.5, 0.4, 0.2, 0.2)
hmm_pos = (0.75, 0.7, 0.2, 0.2)
input_pos = (0.1, 0.4, 0.15, 0.15)
output_pos = (0.9, 0.4, 0.15, 0.15)

# Dibujar módulos principales
qsm_rect = Rectangle((qsm_pos[0], qsm_pos[1]), qsm_pos[2], qsm_pos[3], 
                    facecolor=qsm_color, alpha=0.7, edgecolor='black', 
                    linewidth=2, zorder=2)
bnnm_rect = Rectangle((bnnm_pos[0], bnnm_pos[1]), bnnm_pos[2], bnnm_pos[3], 
                     facecolor=bnnm_color, alpha=0.7, edgecolor='black', 
                     linewidth=2, zorder=2)
hmm_rect = Rectangle((hmm_pos[0], hmm_pos[1]), hmm_pos[2], hmm_pos[3], 
                    facecolor=hmm_color, alpha=0.7, edgecolor='black', 
                    linewidth=2, zorder=2)
input_rect = Rectangle((input_pos[0], input_pos[1]), input_pos[2], input_pos[3], 
                      facecolor=input_color, alpha=0.7, edgecolor='black', 
                      linewidth=2, zorder=2)
output_rect = Rectangle((output_pos[0], output_pos[1]), output_pos[2], output_pos[3], 
                       facecolor=output_color, alpha=0.7, edgecolor='black', 
                       linewidth=2, zorder=2)

# Añadir módulos al gráfico
ax.add_patch(qsm_rect)
ax.add_patch(bnnm_rect)
ax.add_patch(hmm_rect)
ax.add_patch(input_rect)
ax.add_patch(output_rect)

# Añadir etiquetas
ax.text(qsm_pos[0] + qsm_pos[2]/2, qsm_pos[1] + qsm_pos[3]/2, 'QSM\nMódulo de\nSimulación Cuántica', 
        ha='center', va='center', fontweight='bold', color='white')
ax.text(bnnm_pos[0] + bnnm_pos[2]/2, bnnm_pos[1] + bnnm_pos[3]/2, 'BNNM\nRedes Neuronales\nBioinspirando', 
        ha='center', va='center', fontweight='bold', color='white')
ax.text(hmm_pos[0] + hmm_pos[2]/2, hmm_pos[1] + hmm_pos[3]/2, 'HMM\nMódulo de\nMemoria Holográfica', 
        ha='center', va='center', fontweight='bold', color='white')
ax.text(input_pos[0] + input_pos[2]/2, input_pos[1] + input_pos[3]/2, 'Entrada\nConsulta/Texto', 
        ha='center', va='center', fontweight='bold', color='white')
ax.text(output_pos[0] + output_pos[2]/2, output_pos[1] + output_pos[3]/2, 'Salida\nRespuesta', 
        ha='center', va='center', fontweight='bold', color='white')

# Dibujar flechas de conexión
arrow_style = dict(arrowstyle='->', linewidth=2, color='gray')

# Entrada a BNNM
input_to_bnnm = FancyArrowPatch((input_pos[0] + input_pos[2], input_pos[1] + input_pos[3]/2),
                               (bnnm_pos[0], bnnm_pos[1] + bnnm_pos[3]/2),
                               connectionstyle="arc3,rad=0.0", **arrow_style)
ax.add_patch(input_to_bnnm)

# BNNM a QSM
bnnm_to_qsm = FancyArrowPatch((bnnm_pos[0] + bnnm_pos[2]/2, bnnm_pos[1] + bnnm_pos[3]),
                             (qsm_pos[0] + qsm_pos[2]/2, qsm_pos[1]),
                             connectionstyle="arc3,rad=0.0", **arrow_style)
ax.add_patch(bnnm_to_qsm)

# BNNM a HMM
bnnm_to_hmm = FancyArrowPatch((bnnm_pos[0] + bnnm_pos[2]/2, bnnm_pos[1] + bnnm_pos[3]),
                             (hmm_pos[0] + hmm_pos[2]/2, hmm_pos[1]),
                             connectionstyle="arc3,rad=0.0", **arrow_style)
ax.add_patch(bnnm_to_hmm)

# QSM a BNNM
qsm_to_bnnm = FancyArrowPatch((qsm_pos[0] + qsm_pos[2]/2, qsm_pos[1]),
                             (bnnm_pos[0] + bnnm_pos[2]/2, bnnm_pos[1] + bnnm_pos[3]),
                             connectionstyle="arc3,rad=-0.3", **arrow_style)
ax.add_patch(qsm_to_bnnm)

# HMM a BNNM
hmm_to_bnnm = FancyArrowPatch((hmm_pos[0] + hmm_pos[2]/2, hmm_pos[1]),
                             (bnnm_pos[0] + bnnm_pos[2]/2, bnnm_pos[1] + bnnm_pos[3]),
                             connectionstyle="arc3,rad=0.3", **arrow_style)
ax.add_patch(hmm_to_bnnm)

# BNNM a Salida
bnnm_to_output = FancyArrowPatch((bnnm_pos[0] + bnnm_pos[2], bnnm_pos[1] + bnnm_pos[3]/2),
                                (output_pos[0], output_pos[1] + output_pos[3]/2),
                                connectionstyle="arc3,rad=0.0", **arrow_style)
ax.add_patch(bnnm_to_output)

# Añadir título
ax.text(0.5, 0.95, 'Arquitectura del Sistema NEBULA', 
        ha='center', va='center', fontsize=18, fontweight='bold', transform=ax.transAxes)

# Añadir subtítulo
ax.text(0.5, 0.9, 'Integración de Simulación Cuántica, Redes Neuronales Bioinspirando y Memoria Holográfica', 
        ha='center', va='center', fontsize=14, color='gray', transform=ax.transAxes)

# Añadir leyenda de flujo de datos
ax.text(0.5, 0.05, 'Flujo de datos: La entrada se procesa a través del BNNM, que interactúa con QSM y HMM\npara optimizar el procesamiento y recuperar información relevante antes de generar la salida.', 
        ha='center', va='center', fontsize=12, color='gray', transform=ax.transAxes)

# Configurar ejes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Guardar la figura
plt.tight_layout()
plt.savefig('/home/ubuntu/nebula_project/images/nebula_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Imagen de arquitectura de NEBULA creada con éxito.")
