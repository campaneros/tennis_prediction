import torch
from scripts.pretrain_tennis_rules import TennisRulesNet, compute_tennis_features, compute_labels, custom_loss_pretrain
torch.cuda.set_device(0)

model = TennisRulesNet(input_size=31, hidden_sizes=[128, 64], dropout=0.4)
model = model.cuda()

x = torch.randn(64, 31, device="cuda")   # input finto compatibile
y = model(x)
loss = y['match'].float().mean()
loss.backward()

torch.cuda.synchronize()
print("mini-train ok")

