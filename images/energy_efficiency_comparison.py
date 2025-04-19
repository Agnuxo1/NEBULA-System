import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# Configurar el estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Datos para la comparativa de eficiencia energética
models = ['GPT-3 (175B)', 'LLaMA 2 (70B)', 'BERT (340M)', 'NEBULA']
energy_consumption = [100, 65, 30, 15]  # Valores normalizados (%)
carbon_footprint = [100, 60, 25, 12]    # Valores normalizados (%)
operations_per_watt = [1, 1.8, 4.2, 8.5]  # Valores relativos a GPT-3

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('white')

# Colores
colors = ['#4285F4', '#34A853', '#FBBC05', '#6e8efb']
nebula_color = '#6e8efb'

# Gráfico 1: Consumo energético y huella de carbono
x = np.arange(len(models))
width = 0.35

# Barras para consumo energético
bars1 = ax1.bar(x - width/2, energy_consumption, width, label='Consumo energético', 
               color=[colors[i] if i != 3 else nebula_color for i in range(len(models))])

# Barras para huella de carbono
bars2 = ax1.bar(x + width/2, carbon_footprint, width, label='Huella de carbono', 
               color=[colors[i] if i != 3 else nebula_color for i in range(len(models))], 
               alpha=0.7)

# Añadir etiquetas y título
ax1.set_ylabel('Porcentaje relativo a GPT-3 (%)')
ax1.set_title('Consumo Energético y Huella de Carbono')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()

# Añadir etiquetas de porcentaje en las barras
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

add_labels(bars1)
add_labels(bars2)

# Destacar NEBULA
for i, model in enumerate(models):
    if model == 'NEBULA':
        bars1[i].set_edgecolor('black')
        bars1[i].set_linewidth(2)
        bars2[i].set_edgecolor('black')
        bars2[i].set_linewidth(2)

# Gráfico 2: Operaciones por vatio
ax2.bar(models, operations_per_watt, color=[colors[i] if i != 3 else nebula_color for i in range(len(models))])
ax2.set_ylabel('Operaciones por vatio (relativo a GPT-3)')
ax2.set_title('Eficiencia Energética')

# Añadir etiquetas de valor en las barras
for i, v in enumerate(operations_per_watt):
    ax2.text(i, v + 0.1, f'{v}x', ha='center', fontweight='bold')

# Destacar NEBULA
for i, model in enumerate(models):
    if model == 'NEBULA':
        ax2.get_children()[i].set_edgecolor('black')
        ax2.get_children()[i].set_linewidth(2)

# Añadir anotación para NEBULA
ax2.annotate('NEBULA logra 8.5x más\noperaciones por vatio\nque GPT-3',
            xy=(3, operations_per_watt[3]),
            xytext=(2.5, operations_per_watt[3] + 2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, fontweight='bold')

# Añadir título general
fig.suptitle('Comparativa de Eficiencia Energética', fontsize=16, fontweight='bold')

# Añadir nota explicativa
fig.text(0.5, 0.01, 
         'La arquitectura de NEBULA logra una reducción significativa en consumo energético y huella de carbono\n'
         'gracias a su diseño basado en principios de física cuántica y óptica avanzada.',
         ha='center', fontsize=10, color='gray')

# Ajustar diseño
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/ubuntu/nebula_project/images/energy_efficiency_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Imagen de comparativa de eficiencia energética creada con éxito.")
