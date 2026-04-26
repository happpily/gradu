"""
models.py — 神经网络模型定义
============================
本模块实现两种基于图神经网络的无线功率分配模型：

1. CGCNetLayer / CGCNet
   - 基于最大聚合的自定义图卷积网络（Graph Convolutional Network）
   - 消息传递：拼接邻居特征与边特征 → MLP → 最大池化聚合
   - 保留作为对比基线

2. GATLayer / GATNet  ← 本论文主要提出的方案
   - 图注意力网络（Graph Attention Network, GAT）
   - 核心创新：通过多头注意力机制，自适应地对不同干扰源加权
     · 注意力权重由发送节点特征、接收节点特征、信道特征共同决定
     · 多头机制使模型从多个视角捕获干扰关系
   - 残差连接 + Layer Norm：防止梯度消失，稳定深层训练
   - 输出层：MLP + Sigmoid → 映射至 [0, p_max] 功率范围

图结构说明：
    - 节点 i 对应第 i 个发射-接收链路对
    - 节点特征 x_i = [log(1+g_ii), α_i, log10(σ²)]
        · g_ii: 直传信道增益
        · α_i:  基于距离的链路权重（越近权重越大）
        · σ²:   噪声功率
    - 有向边 (j→i) 表示发射机 j 对接收机 i 的干扰
    - 边特征 e_{j→i} = [log(1+g_{j,i}), log(1+g_{i,j})]
        · g_{j,i}: 干扰信道增益（j→i 方向）
        · g_{i,j}: 反向信道增益（用于捕获信道互易性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, MessagePassing

from module.config import Config


# ═══════════════════════════════════════════════════════════════════════════
# CGCNet：自定义图卷积网络（对比基线）
# CGCNet: Custom Graph Convolutional Network (Comparison Baseline)
# ═══════════════════════════════════════════════════════════════════════════

class CGCNetLayer(MessagePassing):
    """
    自定义图卷积层，采用最大聚合策略。
    Custom graph convolution layer using max aggregation.

    消息传递流程：
        1. 消息生成：m_{j→i} = MLP1([h_j ∥ e_{j,i}])
           （将邻居特征与边特征拼接后通过 MLP 生成消息）
        2. 聚合：agg_i = MAX_{j∈N(i)} m_{j→i}
           （对所有邻居消息取逐元素最大值）
        3. 更新：h_i' = MLP2([h_i ∥ agg_i]) + h_i
           （拼接原特征与聚合结果，通过 MLP 更新，加残差）
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden1: int, hidden2: int):
        super().__init__(aggr="max")  # 最大聚合 (Max aggregation)

        # 消息 MLP：生成邻居传向当前节点的消息
        # Message MLP: generate messages from neighbors
        self.mlp1 = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        # 更新 MLP：将聚合消息与节点特征融合，生成新特征
        # Update MLP: fuse aggregated messages with node features
        self.mlp2 = nn.Sequential(
            nn.Linear(node_dim + hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, node_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # 执行消息传递并聚合 (Run message passing & aggregation)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.mlp2(combined)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j: 邻居节点特征 [E, node_dim]
        # edge_attr: 边特征  [E, edge_dim]
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp1(msg_input)


class CGCNet(nn.Module):
    """
    CGCNet：自定义图卷积功率分配网络。
    CGCNet: Custom graph convolutional network for power allocation.

    结构：
        输入节点特征 (3维)
        → num_layers 次共享图卷积层（带残差连接）
        → 拼接初始特征
        → 线性输出层
        → Sigmoid × p_max → 功率分配向量
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.p_max = config.p_max

        node_dim = 3  # [log(1+g_ii), α_i, log10(σ²)]
        edge_dim = 2  # [log(1+g_{j,i}), log(1+g_{i,j})]

        # 所有层共享权重的图卷积层
        # Shared-weight graph convolution layer (applied num_layers times)
        self.shared_layer = CGCNetLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden1=config.hidden_dim1,
            hidden2=config.hidden_dim2,
        )

        # 输出层：将深层特征与初始特征拼接后映射到标量功率
        # Output layer: map concatenated features to scalar power
        self.output_layer = nn.Linear(node_dim * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播：给定网络拓扑图，输出每条链路的发射功率。

        Args:
            data: PyG Data 对象，包含 x, edge_index, edge_attr 等字段

        Returns:
            power: [N] 张量，每条链路的功率分配值 ∈ [0, p_max]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_init = x  # 保存初始特征用于跳跃连接 (Save for skip connection)

        # 多次应用共享图卷积层（带残差连接）
        # Apply shared layer multiple times with residual connections
        for _ in range(self.config.num_layers):
            residual = self.shared_layer(x, edge_index, edge_attr)
            x = x + residual  # 残差连接 (Residual connection)

        # 拼接初始特征（跳跃连接），丰富输出表示
        # Concatenate initial features (skip connection) for richer output
        x_out = torch.cat([x, x_init], dim=-1)

        # 线性映射 + Sigmoid 激活，输出功率分配
        # Linear mapping + Sigmoid to output power allocation
        power_raw = self.output_layer(x_out).squeeze(-1)
        power = torch.sigmoid(power_raw) * self.p_max
        return power


# ═══════════════════════════════════════════════════════════════════════════
# GATNet：图注意力网络（本论文主要方案）
# GATNet: Graph Attention Network (Main Proposed Scheme)
# ═══════════════════════════════════════════════════════════════════════════

class GATNet(nn.Module):
    """
    GATNet：基于多头图注意力网络的无线功率分配模型。
    GATNet: Multi-head Graph Attention Network for wireless power allocation.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    核心思想 (Core Idea)：
        在 WLAN/蜂窝网络中，不同干扰源对目标接收机的影响差异极大，
        取决于信道增益、距离和当前功率分配。传统 GCN 对所有邻居等权
        聚合，无法捕捉这种差异性。

        GAT 通过可学习的注意力机制，为每条干扰边 (j→i) 计算权重：
            α_{j,i} = softmax( LeakyReLU( aᵀ · [W·h_i ∥ W·h_j ∥ W_e·e_{j,i}] ) )
        权重 α_{j,i} 越大，表示干扰源 j 对接收机 i 的影响越显著，
        模型会重点聚合该干扰信息并在功率分配中加以规避。

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    网络结构 (Architecture)：

        输入节点特征 x ∈ ℝ^{N×3}
              ↓
        输入映射层（Linear: 3 → H×D）
              ↓
        ┌─────────────────────────────┐
        │  GAT 层 × gat_layers         │
        │  GATConv(H×D → H×D, heads=H) │  ← 边特征 e_{j,i} ∈ ℝ^2
        │  + LayerNorm + ELU + Dropout │
        │  + 残差连接                   │
        └─────────────────────────────┘
              ↓
        拼接初始特征 → [H×D + 3]
              ↓
        输出 MLP：Linear → ELU → Linear → 1
              ↓
        Sigmoid × p_max → p_i ∈ [0, p_max]

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    参数说明 (Parameters from Config)：
        gat_heads    : 注意力头数 H（默认 4）
        gat_head_dim : 每头隐藏维度 D（默认 32），总维度 = H×D = 128
        gat_layers   : GAT 层数（默认 4）
        gat_dropout  : 注意力 Dropout 率（默认 0.10）
        gat_mlp_hidden: 输出 MLP 中间层维度（默认 64）
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.p_max = config.p_max

        # 节点特征维度 / 边特征维度
        node_in_dim = 3   # [log(1+g_ii), α_i, log10(σ²)]
        edge_dim = 2      # [log(1+g_{j,i}), log(1+g_{i,j})]

        # 多头注意力的总隐藏维度
        # Total hidden dim = heads × head_dim
        H = config.gat_heads
        D = config.gat_head_dim
        hidden_dim = H * D  # 例如 4 × 32 = 128

        # ── 输入映射：将 3 维节点特征投影到 GAT 隐藏空间 ──────────────────
        # Input projection: map 3-dim node features to GAT hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ELU(),
        )

        # ── 多层 GAT 消息传递层 ──────────────────────────────────────────
        # Multi-layer GAT message passing blocks
        self.gat_layers_list = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(config.gat_layers):
            # GATConv: PyG 官方实现，支持边特征与多头注意力
            # GATConv: PyG implementation supporting edge features & multi-head attention
            gat_conv = GATConv(
                in_channels=hidden_dim,
                out_channels=D,           # 每头输出维度
                heads=H,                  # 注意力头数
                concat=True,              # 拼接各头输出 → out_dim = H × D
                dropout=config.gat_dropout,
                edge_dim=edge_dim,        # 支持边特征作为注意力输入
                add_self_loops=False,     # 图中已隐含自环信息（直传信道）
                bias=True,
            )
            self.gat_layers_list.append(gat_conv)
            # Layer Normalization：稳定深层网络训练
            # Layer Normalization: stabilize deep network training
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # ── 输出 MLP：将图特征映射为功率分配 ─────────────────────────────
        # Output MLP: map graph features to power allocation
        # 输入维度 = GAT 输出维度 + 初始节点特征维度（跳跃连接）
        # Input dim = GAT output + initial node features (skip connection)
        mlp_in_dim = hidden_dim + node_in_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, config.gat_mlp_hidden),
            nn.ELU(),
            nn.Dropout(config.gat_dropout * 0.5),  # 输出层轻量 Dropout
            nn.Linear(config.gat_mlp_hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播：给定网络拓扑图数据，输出每条链路的最优发射功率。

        流程：
            1. 输入映射：3维节点特征 → GAT 隐藏空间
            2. 多层 GAT：迭代聚合干扰邻居信息，注意力权重反映干扰强度
            3. 残差 + LayerNorm：防止梯度消失
            4. 跳跃连接：拼接初始特征保留原始信道信息
            5. MLP 输出 + Sigmoid：映射到 [0, p_max]

        Args:
            data: PyG Data 对象，必须包含：
                  · x          [N, 3]   节点特征
                  · edge_index [2, E]   有向边索引
                  · edge_attr  [E, 2]   边特征

        Returns:
            power: [N] 张量，各链路发射功率 ∈ [0, p_max]
        """
        x_init = data.x                  # [N, 3] — 保存初始特征
        edge_index = data.edge_index     # [2, E]
        edge_attr = data.edge_attr       # [E, 2]

        # Step 1: 输入映射 (Input projection)
        h = self.input_proj(x_init)  # [N, hidden_dim]

        # Step 2: 多层 GAT 消息传递（每层带残差连接和 Layer Norm）
        # Multi-layer GAT message passing with residual + LayerNorm
        for gat_conv, layer_norm in zip(self.gat_layers_list, self.layer_norms):
            # GAT 卷积：利用注意力权重聚合邻居干扰信息
            # GAT conv: aggregate neighbor interference with attention weights
            h_new = gat_conv(h, edge_index, edge_attr=edge_attr)  # [N, hidden_dim]

            # 残差连接 + Layer Normalization + 激活函数
            # Residual connection + LayerNorm + activation
            h = F.elu(layer_norm(h + h_new))  # [N, hidden_dim]

        # Step 3: 跳跃连接——拼接初始节点特征
        # Skip connection: concatenate initial node features
        h_out = torch.cat([h, x_init], dim=-1)  # [N, hidden_dim + 3]

        # Step 4: MLP 输出 + Sigmoid 激活，约束功率在 [0, p_max]
        # MLP output + Sigmoid to constrain power in [0, p_max]
        power_raw = self.output_mlp(h_out).squeeze(-1)  # [N]
        power = torch.sigmoid(power_raw) * self.p_max   # [N] ∈ [0, p_max]

        return power
