import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from app import outputs, y_test

languages = ["de", "en", "es", "fr", "nl", "pt"]

# Visualize using PCA
model_outputs = outputs.cpu().numpy()
targets = y_test.numpy()
outputs_PCA = PCA(n_components=2).fit_transform(model_outputs)

fig, ax = plt.subplots()
scatter = ax.scatter(*outputs_PCA.T, c=targets, cmap="tab10", alpha=0.3)
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
for i, text in enumerate(legend1.get_texts()):
    text.set_text(languages[i])
plt.show()
