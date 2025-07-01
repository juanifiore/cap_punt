#%%
import matplotlib.pyplot as plt

acc_cap = [0.77, 0.78, 0.83, 0.86, 0.93, 0.95, 0.99, 0.98, 1]
f1_macro_cap = [0.36, 0.38, 0.42, 0.41, 0.65, 0.69, 0.96, 0.93, 1]
acc_ini = [0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 1, 1]
f1_macro_ini = [0.49, 0.49, 0.49, 0.49, 0.49, 0.83, 0.49, 1, 1]
aca_fin = [0.89, 0.89, 0.89, 0.89, 0.91, 0.93, 0.94, 0.97, 0.98]
f1_macro_fin = [0.23, 0.23, 0.23, 0.23, 0.38, 0.47, 0.59, 0.64, 0.70]

'''
acc_cap = [0.79, 0.78, 0.85, 0.91, 0.95, 1, 0.99, 1, 1]
f1_macro_cap = [0.38, 0.42, 0.40, 0.60, 0.67, 1,0.98, 1, 1]
acc_ini = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 1, 1]
f1_macro_ini = [0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 1, 1]
aca_fin = [0.89, 0.89, 0.89, 0.92, 0.96, 0.96, 1, 0.98, 1]
f1_macro_fin = [0.23, 0.23, 0.27, 0.44, 0.65, 0.60, 1, 0.72, 1]
'''

# Usamos strings para el eje x para que sean equiespaciados como categorías
x_labels = ['2', '4', '8', '16', '32', '64', '128', '256', '768']
x_pos = list(range(len(x_labels)))

# Gráfico
plt.figure(figsize=(10, 6))

# CAP
plt.plot(x_pos, acc_cap, linestyle='--', color='tab:blue', label='Accuracy CAP')
plt.plot(x_pos, f1_macro_cap, linestyle='-', marker='o', color='tab:blue', label='F1 Macro CAP')

# INI
plt.plot(x_pos, acc_ini, linestyle='--', color='tab:green', label='Accuracy INI')
plt.plot(x_pos, f1_macro_ini, linestyle='-', marker='o', color='tab:green', label='F1 Macro INI')

# FIN
plt.plot(x_pos, aca_fin, linestyle='--', color='tab:orange', label='Accuracy FIN')
plt.plot(x_pos, f1_macro_fin, linestyle='-', marker='o', color='tab:orange', label='F1 Macro FIN')

plt.xticks(x_pos, x_labels)
plt.xlabel('Embedding size')
plt.ylabel('Valor de métrica')
plt.title('RNN GRU Unidireccional - Resultados CAP, INI y FIN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
