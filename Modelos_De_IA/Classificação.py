import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CLASSES = [1, 2, 3, 4, 5]
CLASS_NAMES = ["Neutro", "Sorrindo", "Sobrancelhas levantadas", "Surpreso", "Rabugento"]
COLORS = ["gray", "y", "green", "pink", "red"]
CMAP = ListedColormap(COLORS)
PLOT_GRAPHS = True
FIGURE_INDEX = 1


def plot_config(n_figure, title, data_classes):
    """Configures and plots data points for each class."""
    fig, ax = plt.subplots(num=n_figure)  
    for i, class_data in enumerate(data_classes):
        ax.scatter(class_data[0], class_data[1], label=CLASS_NAMES[i], c=COLORS[i], ec="k")
    ax.set(xlabel="Corrugador do Supercílio", ylabel="Zigomático Maior", title=title,
           xlim=(-100, 4170), ylim=(-100, 4170))
    ax.legend()
    return ax

def plot_frontier_mqo(ax, w):
    """Plots the decision boundary for MQO."""
    x_axis = np.linspace(-200, 5000, 1000)
    for i in range(len(CLASSES)):
        x2 = -w[0, i] / w[2, i] - w[1, i] / w[2, i] * x_axis
        ax.plot(x_axis, x2, "--k")
    xx, yy = np.meshgrid(x_axis, x_axis)
    x_plot = np.hstack((np.ones((xx.size, 1)), xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))
    y_pred = np.argmax(x_plot @ w, axis=1).reshape(xx.shape)
    ax.contourf(xx, yy, y_pred, alpha=0.3, cmap=CMAP, levels=np.arange(-0.5, 5.5, 1))
    return ax

def plot_frontier_gauss(ax, means, covariances, covariances_inv, method='mahalanobis', lambdas=None):
    """Plots decision boundaries for Gaussian Bayesian models."""

    x_min, x_max = X_gauss[0].min() - 500, X_gauss[0].max() + 500
    y_min, y_max = X_gauss[1].min() - 500, X_gauss[1].max() + 500
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    x_grid = np.c_[xx.ravel(), yy.ravel()]

    z = []
    for x in x_grid:
        if method == 'mahalanobis':
            distances = [mahalanobis_distance(x, means[i], covariances_inv) for i in range(len(CLASSES))]
            z.append(np.argmin(distances))
        elif method == 'friedman':
            distances = [friedman_discriminant(x, means[i], covariances[i], covariances_inv[i]) for i in range(len(CLASSES))]
            z.append(np.argmax(distances))
        elif method == 'fdp':
             probabilities = [fdp(x, means[i], covariances[i], covariances_inv[i]) for i in range(len(CLASSES))]
             z.append(np.argmax(probabilities))

    z = np.array(z).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.3, cmap=CMAP, levels=np.arange(-0.5, 5.5, 1))
    ax.set(xlim=(-100, 4170), ylim=(-100, 4170))
    return ax

def calculate_accuracy(x, y, model_func, *args):
    """Calculates the accuracy of a model."""
    correct = 0
    for i in range(x.shape[1]): 
        prediction = model_func(x[:, i], *args)
        if prediction == np.argmax(y[i]):
            correct += 1
    return (correct / x.shape[1]) * 100

def predict_mqo(x, w):
    return np.argmax(x @ w)

def mahalanobis_distance(x, mu, sigma_inv):
    return (x - mu).T @ sigma_inv @ (x - mu)

def friedman_discriminant(x, mean, covariance, covariance_inv):
    return -0.5 * np.log(np.linalg.det(covariance)) - 0.5 * (x - mean).T @ covariance_inv @ (x - mean)

def fdp(x, mu, sigma, sigma_inv):
    p = len(mu)
    coef = -0.5 * (p * np.log(2 * np.pi) + np.log(np.linalg.det(sigma)))
    quad_form = -0.5 * (x - mu).T @ sigma_inv @ (x - mu)
    return coef + quad_form

def predict_mahalanobis(x, means, cov_inv):
    distances = [mahalanobis_distance(x, means[i], cov_inv) for i in range(len(CLASSES))]
    return np.argmin(distances)

def predict_friedman(x, means, covariances, covariances_inv):
    distances = [friedman_discriminant(x, means[i], covariances[i], covariances_inv[i]) for i in range(len(CLASSES))]
    return np.argmax(distances)

def predict_gauss_fdp(x, means, covariances, covariances_inv):
    probabilities = [fdp(x, means[i], covariances[i], covariances_inv[i]) for i in range(len(CLASSES))]
    return np.argmax(probabilities)

emgs = np.loadtxt("EMGsDataset.csv", delimiter=',')
emgs_mqo = emgs.T
data_classes = [emgs[:2, emgs[-1] == cls] for cls in CLASSES]
N, p = emgs.T.shape[0], emgs.shape[1] - 1

X_gauss = np.hstack(data_classes)

y_gauss = np.vstack([np.tile(np.array(
    [1 if i == j else -1 for j in range(len(CLASSES))]), (10000, 1)) for i in range(len(CLASSES))])

X_mqo = np.hstack((
    np.ones((N, 1)),
    np.vstack([emgs_mqo[emgs_mqo[:, -1] == cls, :2] for cls in CLASSES])
))

y_mqo = np.vstack([np.tile(np.array(
    [1 if i == j else -1 for j in range(len(CLASSES))]), (10000, 1)) for i in range(len(CLASSES))])

w_mqo = np.linalg.pinv(X_mqo.T @ X_mqo) @ X_mqo.T @ y_mqo

if PLOT_GRAPHS:
    ax = plot_config(FIGURE_INDEX, "MQO", data_classes)
    x_novo = np.hstack((1, emgs[np.random.randint(0, len(emgs)), 0:2]))
    plot_frontier_mqo(ax, w_mqo)
    FIGURE_INDEX += 1

    covariances = np.array([np.cov(cls_data) for cls_data in data_classes])
    covariances_inv = np.array([np.linalg.pinv(cov) for cov in covariances])
    means = np.array([cls_data.mean(axis=1) for cls_data in data_classes])

    ax = plot_config(FIGURE_INDEX, "Gaussiano Bayesiano (Tradicional)", data_classes)
    plot_frontier_gauss(ax, means, covariances, covariances_inv, method='fdp')
    FIGURE_INDEX += 1

    covariance = np.cov(X_gauss)
    covariance_inv = np.linalg.pinv(covariance)
    ax = plot_config(FIGURE_INDEX, "Gaussiano Bayesiano (Cov. Igual)", data_classes)
    plot_frontier_gauss(ax, means, covariance, covariance_inv)
    FIGURE_INDEX += 1

    aggregated_covariance = np.sum(
        [covariances[i] * data_classes[i].shape[1] / emgs.shape[1] for i in range(len(CLASSES))], axis=0)
    aggregated_covariance_inv = np.linalg.pinv(aggregated_covariance)
    ax = plot_config(FIGURE_INDEX, "Gaussiano Bayesiano (Cov. Agregada)", data_classes)
    plot_frontier_gauss(ax, means, aggregated_covariance, aggregated_covariance_inv)
    FIGURE_INDEX += 1


    lambdas = [0.25, 0.50, 0.75]
    for lamb in lambdas:
        covariances_friedman = [
            ((1 - lamb) * data_classes[i].shape[1] * np.cov(data_classes[i]) + N * lamb * aggregated_covariance) /
            ((1 - lamb) * data_classes[i].shape[1] + N * lamb) for i in range(len(CLASSES))]
        covariances_friedman_inv = [np.linalg.pinv(cov) for cov in covariances_friedman]
        ax = plot_config(FIGURE_INDEX, f"Gaussiano Bayesiano Friedman ({lamb})", data_classes)
        plot_frontier_gauss(ax, means, covariances_friedman, covariances_friedman_inv, method='friedman')
        FIGURE_INDEX += 1

    covariances_naive = [np.diag([np.std(emgs[0]), np.std(emgs[1])]) for _ in CLASSES]
    covariances_naive_inv = [np.linalg.pinv(cov) for cov in covariances_naive]
    ax = plot_config(FIGURE_INDEX, "Gaussiano Bayesiano Naive", data_classes)
    plot_frontier_gauss(ax, means, covariances_naive, covariances_naive_inv, method='fdp')
    FIGURE_INDEX += 1


ROUNDS = 500
PARTITION = 0.8

results_mqo = []
results_gauss = []
results_gauss_equal_cov = []
results_gauss_agg_cov = []
results_naive_bayes = []
results_gauss_friedman = []

for round_num in range(ROUNDS):
    print(f"Round: {round_num + 1}")


    indices = np.random.permutation(N)


    x_mqo_shuffled = X_mqo[indices]
    y_mqo_shuffled = y_mqo[indices]
    x_train_mqo = x_mqo_shuffled[:int(N * PARTITION)]
    y_train_mqo = y_mqo_shuffled[:int(N * PARTITION)]
    x_test_mqo = x_mqo_shuffled[int(N * PARTITION):]
    y_test_mqo = y_mqo_shuffled[int(N * PARTITION):]
    w_mqo_mc = np.linalg.pinv(x_train_mqo.T @ x_train_mqo) @ x_train_mqo.T @ y_train_mqo
    results_mqo.append(calculate_accuracy(x_test_mqo.T, y_test_mqo, predict_mqo, w_mqo_mc))


    x_gauss_shuffled = X_gauss[:, indices]
    y_gauss_shuffled = y_gauss[indices]
    data_shuffled = emgs[:, indices]
    x_train_gauss = x_gauss_shuffled[:, :int(N * PARTITION)]
    y_train_gauss = y_gauss_shuffled[:int(N * PARTITION)]
    data_train = data_shuffled[:, :int(N * PARTITION)]
    x_test_gauss = x_gauss_shuffled[:, int(N * PARTITION):]
    y_test_gauss = y_gauss_shuffled[int(N * PARTITION):]
    data_test = data_shuffled[:, int(N * PARTITION):]

    data_classes_train = [data_train[:2, data_train[-1] == cls] for cls in CLASSES]
    means_gauss = np.array([cls_data.mean(axis=1) for cls_data in data_classes_train])

    try:
        covariances_gauss = np.array([np.cov(cls_data) for cls_data in data_classes_train])
        covariances_gauss_inv = np.array([np.linalg.pinv(cov) for cov in covariances_gauss])
        results_gauss.append(calculate_accuracy(x_test_gauss, y_test_gauss, predict_gauss_fdp, means_gauss, covariances_gauss, covariances_gauss_inv))
    except Exception as e:
        print("Error calculating Gaussian Bayes (Traditional):", e)

    covariance_equal = np.cov(x_train_gauss)
    covariance_equal_inv = np.linalg.pinv(covariance_equal)
    results_gauss_equal_cov.append(calculate_accuracy(x_test_gauss, y_test_gauss, predict_mahalanobis, means_gauss, covariance_equal_inv))

    covariances_agg = np.array([np.cov(data_train[:2, data_train[-1] == cls]) for cls in CLASSES])
    aggregated_covariance = np.sum(
        [covariances_agg[i] * data_classes_train[i].shape[1] / data_train.shape[1] for i in range(len(CLASSES))], axis=0)
    aggregated_covariance_inv = np.linalg.pinv(aggregated_covariance)
    results_gauss_agg_cov.append(calculate_accuracy(x_test_gauss, y_test_gauss, predict_mahalanobis, means_gauss, aggregated_covariance_inv))

    lambdas = [0.25, 0.50, 0.75, 1]
    friedman_results = []
    for lamb in lambdas:
        covariances_friedman = [
            ((1 - lamb) * data_classes_train[i].shape[1] * np.cov(data_classes_train[i]) + data_train.shape[1] * lamb * aggregated_covariance) /
            ((1 - lamb) * data_classes_train[i].shape[1] + data_train.shape[1] * lamb) for i in range(len(CLASSES))]
        covariances_friedman_inv = [np.linalg.pinv(cov) for cov in covariances_friedman]
        friedman_results.append(calculate_accuracy(x_test_gauss, y_test_gauss, predict_friedman, means_gauss, covariances_friedman, covariances_friedman_inv))
    results_gauss_friedman.append(friedman_results)

    covariances_naive = [np.diag([np.std(emgs[0]), np.std(emgs[1])]) for _ in CLASSES]
    covariances_naive_inv = [np.linalg.pinv(cov) for cov in covariances_naive]
    results_naive_bayes.append(calculate_accuracy(x_test_gauss, y_test_gauss, predict_gauss_fdp, means_gauss, covariances_naive, covariances_naive_inv))

results_mqo = np.array(results_mqo)
results_gauss = np.array(results_gauss)
results_gauss_equal_cov = np.array(results_gauss_equal_cov)
results_gauss_agg_cov = np.array(results_gauss_agg_cov)
results_naive_bayes = np.array(results_naive_bayes)
results_gauss_friedman = np.array(results_gauss_friedman)

metrics_mqo = {
    'mean': results_mqo.mean(),
    'std': results_mqo.std(),
    'max': results_mqo.max(),
    'min': results_mqo.min()
}

metrics_gauss = {
    'mean': results_gauss.mean(),
    'std': results_gauss.std(),
    'max': results_gauss.max(),
    'min': results_gauss.min()
}

metrics_gauss_equal_cov = {
    'mean': results_gauss_equal_cov.mean(),
    'std': results_gauss_equal_cov.std(),
    'max': results_gauss_equal_cov.max(),
    'min': results_gauss_equal_cov.min()
}

metrics_gauss_agg_cov = {
    'mean': results_gauss_agg_cov.mean(),
    'std': results_gauss_agg_cov.std(),
    'max': results_gauss_agg_cov.max(),
    'min': results_gauss_agg_cov.min()
}

metrics_naive_bayes = {
    'mean': results_naive_bayes.mean(),
    'std': results_naive_bayes.std(),
    'max': results_naive_bayes.max(),
    'min': results_naive_bayes.min()
}

metrics_gauss_friedman_025 = {
    'mean': results_gauss_friedman[:, 0].mean(),
    'std': results_gauss_friedman[:, 0].std(),
    'max': results_gauss_friedman[:, 0].max(),
    'min': results_gauss_friedman[:, 0].min()
}

metrics_gauss_friedman_050 = {
    'mean': results_gauss_friedman[:, 1].mean(),
    'std': results_gauss_friedman[:, 1].std(),
    'max': results_gauss_friedman[:, 1].max(),
    'min': results_gauss_friedman[:, 1].min()
}

metrics_gauss_friedman_075 = {
    'mean': results_gauss_friedman[:, 2].mean(),
    'std': results_gauss_friedman[:, 2].std(),
    'max': results_gauss_friedman[:, 2].max(),
    'min': results_gauss_friedman[:, 2].min()
}

print("MQO Tradicional:", metrics_mqo)
print("Gauss (Tradicional):", metrics_gauss)
print("Gauss (Cov. Igual):", metrics_gauss_equal_cov)
print("Gauss (Cov. Agregada):", metrics_gauss_agg_cov)
print("Gauss Friedman (0.25):", metrics_gauss_friedman_025)
print("Gauss Friedman (0.50):", metrics_gauss_friedman_050)
print("Gauss Friedman (0.75):", metrics_gauss_friedman_075)
print("Gauss Naive:", metrics_naive_bayes)

models = [
    "MQO", "Gauss", "Gauss (Eq Cov)",
    "Gauss (Agg Cov)", "Gauss Fried (0.25)",
    "Gauss Fried (0.50)", "Gauss Fried (0.75)",
    "Gauss Naive"
]

all_metrics = {
    'Mean': [
        metrics_mqo['mean'], metrics_gauss['mean'],
        metrics_gauss_equal_cov['mean'], metrics_gauss_agg_cov['mean'],
        metrics_gauss_friedman_025['mean'], metrics_gauss_friedman_050['mean'],
        metrics_gauss_friedman_075['mean'], metrics_naive_bayes['mean']
    ],
    'Std Dev': [
        metrics_mqo['std'], metrics_gauss['std'],
        metrics_gauss_equal_cov['std'], metrics_gauss_agg_cov['std'],
        metrics_gauss_friedman_025['std'], metrics_gauss_friedman_050['std'],
        metrics_gauss_friedman_075['std'], metrics_naive_bayes['std']
    ],
    'Max': [
        metrics_mqo['max'], metrics_gauss['max'],
        metrics_gauss_equal_cov['max'], metrics_gauss_agg_cov['max'],
        metrics_gauss_friedman_025['max'], metrics_gauss_friedman_050['max'],
        metrics_gauss_friedman_075['max'], metrics_naive_bayes['max']
    ],
    'Min': [
        metrics_mqo['min'], metrics_gauss['min'],
        metrics_gauss_equal_cov['min'], metrics_gauss_agg_cov['min'],
        metrics_gauss_friedman_025['min'], metrics_gauss_friedman_050['min'],
        metrics_gauss_friedman_075['min'], metrics_naive_bayes['min']
    ],
}

x = np.arange(len(models))
width = 0.15
fig, ax = plt.subplots(layout='constrained')
for i, (metric_name, values) in enumerate(all_metrics.items()):
    offset = (i - len(all_metrics) / 2) * width
    rects = ax.bar(x + offset, values, width, label=metric_name)
    ax.bar_label(rects, padding=3)
ax.set(
    title='Model Metrics Comparison',
    xticks=x,
    xticklabels=models,
    ylim=(0, 110)
)
ax.legend(loc='upper left', ncols=2)

if PLOT_GRAPHS:
    plt.show()