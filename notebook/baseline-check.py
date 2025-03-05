# %%
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


# %%
from dgl.data import CoraGraphDataset

dataset = CoraGraphDataset()
g = dataset[0]


# %%
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.dropout = dropout

    def forward(self, g, x):
        # First GCN layer
        x = self.conv1(g, x)
        x = F.relu(x)
        # Dropout helps regularize
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(g, x)
        return x


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)

features = g.ndata['feat'].to(device)
labels = g.ndata['label'].to(device)
train_mask = g.ndata['train_mask'].to(device)
val_mask = g.ndata['val_mask'].to(device)
test_mask = g.ndata['test_mask'].to(device)

in_feats = features.shape[1]
num_classes = dataset.num_classes
hidden_feats = 64

model = GCN(in_feats, hidden_feats, num_classes, dropout=0.5).to(device)

# Typical Adam optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    logits = model(g, features)  # [N, num_classes]
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        pred = logits.argmax(dim=1)
        
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss.item():.4f} | "
        f"Train Acc {train_acc.item():.4f} | "
        f"Val Acc {val_acc.item():.4f} | "
        f"Test Acc {test_acc.item():.4f}"
    )



