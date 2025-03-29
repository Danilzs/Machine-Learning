import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

dados = np.loadtxt("atividade_enzimatica.csv", delimiter=",")
Eixo_x_temp, Eixo_x_ph = dados[:, 0:1], dados[:, 1:2] 
Eixo_y = dados[:, 2:3] 
total_amostras = len(Eixo_x_temp)

X_matriz = np.column_stack((np.ones((total_amostras, 1)), Eixo_x_temp, Eixo_x_ph))

valores_lambda = [0, 0.25, 0.5, 0.75, 1]

fig = plt.figure(figsize=(10, 6))
axis = fig.add_subplot(111, projection='3d')
axis.scatter(Eixo_x_temp, Eixo_x_ph, Eixo_y, color='red', label="Pontos")

for valor in valores_lambda:
    identidade = np.eye(X_matriz.shape[1]) * valor
    coeficientes = np.linalg.solve(X_matriz.T @ X_matriz + identidade, X_matriz.T @ Eixo_y)
    Y_pred = X_matriz @ coeficientes

    axis.plot_trisurf(Eixo_x_temp.flatten(), Eixo_x_ph.flatten(), Y_pred.flatten(), label=f'MQO Reg. (λ={valor})', alpha=0.5)

repeticoes = 500
rss_mqo = np.zeros((repeticoes, len(valores_lambda)))
rss_media = np.zeros(repeticoes)

for i in range(repeticoes):
    ordem_aleatoria = np.random.permutation(total_amostras)
    tamanho_treino = int(0.8 * total_amostras)


    treinamento_x, teste_x = X_matriz[ordem_aleatoria[:tamanho_treino]], X_matriz[ordem_aleatoria[tamanho_treino:]]
    treinamento_y, teste_y = Eixo_y[ordem_aleatoria[:tamanho_treino]], Eixo_y[ordem_aleatoria[tamanho_treino:]]


    for j, valor in enumerate(valores_lambda):
        identidade = np.eye(treinamento_x.shape[1]) * valor
        coef_reg = np.linalg.solve(treinamento_x.T @ treinamento_x + valor * np.eye(treinamento_x.shape[1]), treinamento_x.T @ treinamento_y)
        rss_mqo[i, j] = np.sum((teste_x @ coef_reg - teste_y) ** 2)

    # Cálculo do erro quadrático para a média
    rss_media[i] = np.sum(np.square(np.mean(treinamento_y) - teste_y))

estatisticas_mqo = {
    "Média": np.mean(rss_mqo, axis=0),
    "Desvio Padrão": np.std(rss_mqo, axis=0),
    "Mínimo": np.min(rss_mqo, axis=0),
    "Máximo": np.max(rss_mqo, axis=0),
}


estatisticas_media = {
    "Média": np.mean(rss_media),
    "Desvio Padrão": np.std(rss_media),
    "Mínimo": np.min(rss_media),
    "Máximo": np.max(rss_media),
}

print("\n📊 Estatísticas do modelo baseado na média:")
tabela_media = [[chave, f"{valor:.4f}"] for chave, valor in estatisticas_media.items()]
print(tabulate(tabela_media, headers=["Métrica", "Valor"], tablefmt="grid"))

print("\n📈 Estatísticas do RSS para cada λ:")
cabecalhos = ["Métrica"] + [f"λ={lmb}" for lmb in valores_lambda]
tabela_rss = [[chave] + [f"{v:.2f}" for v in valores] for chave, valores in estatisticas_mqo.items()]
print(tabulate(tabela_rss, headers=cabecalhos, tablefmt="grid"))

axis.set_xlabel("Temperatura")
axis.set_ylabel("pH")
axis.set_zlabel("Atividade Enzimática")
axis.set_title("Regressão Regularizada e Estatísticas de Erro")
axis.legend(loc="upper left", fontsize=9, frameon=True)
plt.show()
