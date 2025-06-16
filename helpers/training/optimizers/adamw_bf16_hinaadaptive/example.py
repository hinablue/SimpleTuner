#!/usr/bin/env python3
"""
AdamWBF16_HinaAdaptive 優化器使用範例

這個範例展示了如何使用 AdamWBF16_HinaAdaptive 優化器來訓練一個簡單的模型。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from adamw_bf16_hinaadaptive import AdamWBF16_HinaAdaptive


# 定義一個簡單的 Transformer 風格模型
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # 前饋網路
        ffn_output = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]

        for block in self.transformer_blocks:
            x = block(x)

        return self.output_proj(x)


def create_dummy_data(batch_size=32, seq_len=128, vocab_size=10000, num_batches=100):
    """創建虛擬數據用於測試"""
    data = []
    targets = []

    for _ in range(num_batches):
        # 隨機生成序列
        seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        # 目標是向右移動一位的序列
        target = torch.cat([seq[:, 1:], torch.randint(0, vocab_size, (batch_size, 1))], dim=1)

        data.append(seq)
        targets.append(target)

    return data, targets


def train_with_adamw_bf16_hinaadaptive():
    """使用 AdamWBF16_HinaAdaptive 優化器訓練模型"""

    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 建立模型並轉換為 bfloat16
    model = SimpleModel(vocab_size=1000, d_model=256, nhead=8, num_layers=4)
    model = model.to(device).to(torch.bfloat16)

    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")

    # 創建優化器
    optimizer = AdamWBF16_HinaAdaptive(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,

        # 啟用自適應功能
        use_dynamic_adaptation=True,
        adaptation_strength=1.0,
        relationship_discovery_interval=50,
        importance_decay=0.95,
        compatibility_threshold=0.3,

        # 啟用進階優化技術
        use_spd=True,
        spd_lambda=0.06,
        use_cautious=True,
        use_orthogonal_grad=False,  # 對於這個簡單例子可能不需要
        use_adopt_stability=True,
        use_agr=True,
        use_tam=True,
        tam_beta=0.999,

        # 動態權重衰減
        dynamic_weight_decay=True,
        wd_transition_steps=200,
        wd_decay_factor=0.8,
        wd_min_ratio=0.1
    )

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 創建虛擬數據
    data, targets = create_dummy_data(batch_size=16, seq_len=64, vocab_size=1000, num_batches=50)

    print("開始訓練...")

    # 訓練循環
    for epoch in range(5):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_seq, target_seq) in enumerate(zip(data, targets)):
            # 將數據移動到裝置並轉換為 bfloat16
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # 前向傳播
            optimizer.zero_grad()
            outputs = model(input_seq)

            # 計算損失
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_seq.view(-1))

            # 反向傳播
            loss.backward()

            # 優化器步驟
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # 每 10 個批次輸出一次進度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} 完成，平均損失: {avg_loss:.4f}")

        # 輸出優化器統計信息
        if epoch % 2 == 0:  # 每兩個 epoch 輸出一次
            print("\n=== 優化器統計信息 ===")

            # 基本信息
            info = optimizer.get_optimization_info()
            print(f"優化器類型: {info['optimizer_type']}")
            print(f"訓練步數: {info.get('training_stats', {}).get('global_step', 'N/A')}")
            print(f"發現的參數關係: {info.get('training_stats', {}).get('total_relationships', 'N/A')}")

            # 重要性分析
            importance = optimizer.get_importance_analysis()
            if 'importance_statistics' in importance:
                stats = importance['importance_statistics']
                print(f"參數重要性統計:")
                print(f"  平均值: {stats['mean']:.4f}")
                print(f"  最大值: {stats['max']:.4f}")
                print(f"  最小值: {stats['min']:.4f}")
                print(f"  高重要性參數: {importance['high_importance_params']}")

            # 記憶體統計
            memory_stats = optimizer.get_buffer_pool_stats()
            if 'estimated_memory_mb' in memory_stats:
                print(f"緩衝區池記憶體使用: {memory_stats['estimated_memory_mb']:.2f} MB")
                print(f"緩衝區類型數量: {memory_stats['total_buffer_types']}")

            print("=" * 30)

    print("\n訓練完成！")

    # 最終統計
    final_info = optimizer.get_optimization_info()
    print(f"\n最終統計:")
    print(f"總訓練步數: {final_info.get('training_stats', {}).get('global_step', 'N/A')}")
    print(f"發現的總參數關係: {final_info.get('training_stats', {}).get('total_relationships', 'N/A')}")

    # 參數關係摘要
    relationships = optimizer.get_relationship_summary()
    if relationships.get('total_relationships', 0) > 0:
        print(f"\n參數關係摘要:")
        print(f"總關係數: {relationships['total_relationships']}")
        for i, rel in enumerate(relationships['relationships'][:3]):  # 只顯示前3個
            print(f"  關係 {i+1}: {rel['param_shape']} <-> {rel['partner_shape']}")
            print(f"    相容性: {rel['compatibility']:.3f}, 交互類型: {rel['interaction_type']}")

    # 清理記憶體
    optimizer.clear_buffer_pool()
    print("\n緩衝區池已清理")


def compare_optimizers():
    """比較不同優化器的性能"""
    print("\n=== 優化器比較 ===")

    # 這裡可以添加與其他優化器的比較代碼
    # 由於這只是一個範例，我們只是展示如何設置不同的配置

    configs = [
        {
            "name": "基本配置",
            "use_dynamic_adaptation": False,
            "use_spd": False,
            "use_tam": False
        },
        {
            "name": "進階配置",
            "use_dynamic_adaptation": True,
            "adaptation_strength": 1.2,
            "use_spd": True,
            "use_tam": True
        },
        {
            "name": "記憶體優化配置",
            "use_dynamic_adaptation": True,
            "relationship_discovery_interval": 200,  # 減少計算頻率
            "use_orthogonal_grad": False,  # 關閉計算密集的功能
            "use_spd": True
        }
    ]

    for config in configs:
        print(f"\n{config['name']}:")
        for key, value in config.items():
            if key != "name":
                print(f"  {key}: {value}")


if __name__ == "__main__":
    print("AdamWBF16_HinaAdaptive 優化器使用範例")
    print("=" * 50)

    # 檢查是否支援 bfloat16
    if not torch.cuda.is_available():
        print("警告: 沒有檢測到 CUDA，性能可能不佳")

    # 主要訓練範例
    train_with_adamw_bf16_hinaadaptive()

    # 顯示配置比較
    compare_optimizers()

    print("\n範例完成！")
    print("要在實際項目中使用，請根據您的模型和數據調整超參數。")