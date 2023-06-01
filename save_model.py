import torch

import app

# Save the model
torch.jit.save(torch.jit.script(app.model), "model.pt")
