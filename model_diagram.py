import model
import torch
from torchviz import make_dot

# Generate a model diagram
model = model.CNN_LSTM()
x = torch.randn(1, 1, 13, 248)
y = model(x)

make_dot(y, params=dict(model.named_parameters())).render("rnn_torchviz", format="png")
