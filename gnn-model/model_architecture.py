from import_library import *

class HybridGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        #Input projection
        self.proj = nn.Linear(in_dim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)

        #GraphSAGE branch
        self.sage1 = SAGEConv(hidden, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.bn_s = nn.BatchNorm1d(hidden)

        #GAT branch
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads,
                             dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads,
                             dropout=dropout, add_self_loops=False)
        self.bn_g = nn.BatchNorm1d(hidden)

        #Residual MLP on raw features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        #Fusion head
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index):
        h0 = F.relu(self.bn0(self.proj(x)))

        #SAGE
        s = F.relu(self.sage1(h0, edge_index))
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = self.bn_s(self.sage2(s, edge_index))

        #GAT
        g = F.elu(self.gat1(h0, edge_index))
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = self.bn_g(self.gat2(g, edge_index))

        #MLP residual
        m = self.mlp(x)

        return self.head(torch.cat([s, g, m], dim=1))
