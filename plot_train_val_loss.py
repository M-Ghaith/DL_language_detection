import matplotlib.pyplot as plt

import app

# Plot the training and validation losses
plt.plot(app.losses['train'], label='Train Loss')
plt.plot(app.losses['validation'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
