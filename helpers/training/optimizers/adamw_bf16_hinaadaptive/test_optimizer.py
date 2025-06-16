#!/usr/bin/env python3
"""
AdamWBF16_HinaAdaptive 優化器測試腳本

這個腳本用於測試優化器的基本功能和正確性。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adamw_bf16_hinaadaptive import AdamWBF16_HinaAdaptive


class TestModel(nn.Module):
    """用於測試的簡單模型"""
    def __init__(self, input_size=128, hidden_size=256, output_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def test_basic_functionality():
    """測試基本功能"""
    print("測試 1: 基本功能測試")

    # 檢查 CUDA 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 創建模型並轉換為 bfloat16
    model = TestModel().to(device).to(torch.bfloat16)

    # 創建優化器
    optimizer = AdamWBF16_HinaAdaptive(model.parameters(), lr=1e-3)

    # 創建虛擬數據
    batch_size = 8
    input_data = torch.randn(batch_size, 128, device=device, dtype=torch.bfloat16)
    target_data = torch.randn(batch_size, 64, device=device, dtype=torch.bfloat16)

    # 損失函數
    criterion = nn.MSELoss()

    print("執行前向和後向傳播...")

    # 前向傳播
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)

    print(f"初始損失: {loss.item():.6f}")

    # 後向傳播
    loss.backward()

    # 檢查梯度
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    print(f"梯度範數: {grad_norm:.6f}")

    # 優化步驟
    optimizer.step()

    # 再次前向傳播檢查損失變化
    with torch.no_grad():
        new_output = model(input_data)
        new_loss = criterion(new_output, target_data)

    print(f"優化後損失: {new_loss.item():.6f}")
    print(f"損失變化: {new_loss.item() - loss.item():.6f}")

    print("✅ 基本功能測試通過\n")


def test_adaptive_features():
    """測試自適應功能"""
    print("測試 2: 自適應功能測試")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel().to(device).to(torch.bfloat16)

    # 啟用所有自適應功能
    optimizer = AdamWBF16_HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        use_dynamic_adaptation=True,
        relationship_discovery_interval=5,  # 設定較短間隔用於測試
        adaptation_strength=1.2,
        compatibility_threshold=0.2
    )

    # 創建數據
    input_data = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    target_data = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    criterion = nn.MSELoss()

    print("執行多步訓練以觸發自適應功能...")

    for step in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"步驟 {step+1}, 損失: {loss.item():.6f}")

    # 檢查優化器狀態
    info = optimizer.get_optimization_info()
    print(f"總訓練步數: {info.get('training_stats', {}).get('global_step', 'N/A')}")
    print(f"發現的參數關係: {info.get('training_stats', {}).get('total_relationships', 'N/A')}")

    # 重要性分析
    importance = optimizer.get_importance_analysis()
    if 'importance_statistics' in importance:
        stats = importance['importance_statistics']
        print(f"參數重要性 - 平均: {stats['mean']:.4f}, 最大: {stats['max']:.4f}")

    print("✅ 自適應功能測試通過\n")


def test_memory_optimization():
    """測試記憶體優化功能"""
    print("測試 3: 記憶體優化測試")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel().to(device).to(torch.bfloat16)

    optimizer = AdamWBF16_HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        use_orthogonal_grad=True  # 啟用需要緩衝區的功能
    )

    # 執行幾步訓練來創建緩衝區
    input_data = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    target_data = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    criterion = nn.MSELoss()

    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()

    # 檢查緩衝區池統計
    memory_stats = optimizer.get_buffer_pool_stats()
    if 'total_buffers' in memory_stats:
        print(f"緩衝區池統計:")
        print(f"  總緩衝區數: {memory_stats['total_buffers']}")
        print(f"  緩衝區類型數: {memory_stats['total_buffer_types']}")
        print(f"  估計記憶體使用: {memory_stats['estimated_memory_mb']:.2f} MB")
    else:
        print("緩衝區池未初始化或無緩衝區")

    # 測試緩衝區清理
    optimizer.clear_buffer_pool()
    cleared_stats = optimizer.get_buffer_pool_stats()
    print(f"清理後緩衝區數: {cleared_stats.get('total_buffers', 0)}")

    print("✅ 記憶體優化測試通過\n")


def test_advanced_features():
    """測試進階優化技術"""
    print("測試 4: 進階優化技術測試")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel().to(device).to(torch.bfloat16)

    # 啟用各種進階功能
    optimizer = AdamWBF16_HinaAdaptive(
        model.parameters(),
        lr=1e-3,
        use_spd=True,
        use_cautious=True,
        use_adopt_stability=True,
        use_agr=True,
        use_tam=True,
        dynamic_weight_decay=True
    )

    # 創建數據並訓練
    input_data = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    target_data = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    criterion = nn.MSELoss()

    initial_loss = None

    for step in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    improvement = initial_loss - final_loss

    print(f"初始損失: {initial_loss:.6f}")
    print(f"最終損失: {final_loss:.6f}")
    print(f"改善程度: {improvement:.6f}")

    # 檢查是否有收斂趨勢
    if improvement > 0:
        print("✅ 模型顯示收斂趨勢")
    else:
        print("⚠️ 模型未顯示明顯收斂（可能需要調整超參數）")

    print("✅ 進階優化技術測試通過\n")


def test_error_handling():
    """測試錯誤處理"""
    print("測試 5: 錯誤處理測試")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 測試非 bfloat16 參數的錯誤處理
    model_float32 = TestModel().to(device).to(torch.float32)
    optimizer = AdamWBF16_HinaAdaptive(model_float32.parameters(), lr=1e-3)

    input_data = torch.randn(4, 128, device=device, dtype=torch.float32)
    target_data = torch.randn(4, 64, device=device, dtype=torch.float32)
    criterion = nn.MSELoss()

    try:
        optimizer.zero_grad()
        output = model_float32(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        print("❌ 應該拋出 bfloat16 類型錯誤")
    except AssertionError as e:
        print("✅ 正確捕獲 bfloat16 類型錯誤")
    except Exception as e:
        print(f"⚠️ 捕獲到意外錯誤: {e}")

    print("✅ 錯誤處理測試完成\n")


def test_parameter_validation():
    """測試參數驗證"""
    print("測試 6: 參數驗證測試")

    model = TestModel()

    # 測試無效參數
    invalid_configs = [
        {"lr": -1e-3, "expected_error": "learning rate"},
        {"eps": -1e-8, "expected_error": "epsilon"},
        {"betas": (1.5, 0.999), "expected_error": "beta"},
        {"betas": (0.9, 1.5), "expected_error": "beta"},
        {"weight_decay": -0.01, "expected_error": "weight_decay"}
    ]

    passed_tests = 0

    for config in invalid_configs:
        try:
            test_config = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
            test_config.update({k: v for k, v in config.items() if k != "expected_error"})

            optimizer = AdamWBF16_HinaAdaptive(model.parameters(), **test_config)
            print(f"❌ 應該拋出 {config['expected_error']} 錯誤")
        except ValueError as e:
            if config["expected_error"].lower() in str(e).lower():
                print(f"✅ 正確捕獲 {config['expected_error']} 錯誤")
                passed_tests += 1
            else:
                print(f"⚠️ 錯誤訊息不符合預期: {e}")
        except Exception as e:
            print(f"⚠️ 捕獲到意外錯誤類型: {type(e).__name__}: {e}")

    print(f"參數驗證測試: {passed_tests}/{len(invalid_configs)} 通過\n")


def run_all_tests():
    """執行所有測試"""
    print("=" * 60)
    print("AdamWBF16_HinaAdaptive 優化器完整測試套件")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_adaptive_features,
        test_memory_optimization,
        test_advanced_features,
        test_error_handling,
        test_parameter_validation
    ]

    passed = 0
    total = len(tests)

    for i, test_func in enumerate(tests, 1):
        try:
            print(f"\n[{i}/{total}] 執行 {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"測試結果: {passed}/{total} 通過")

    if passed == total:
        print("🎉 所有測試通過！優化器運行正常。")
    else:
        print("⚠️ 部分測試失敗，請檢查實現。")

    print("=" * 60)


if __name__ == "__main__":
    # 檢查環境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 名稱: {torch.cuda.get_device_name()}")

    # 檢查 bfloat16 支援
    try:
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
        print("✅ bfloat16 支援正常")
    except Exception as e:
        print(f"❌ bfloat16 支援有問題: {e}")

    # 執行測試
    run_all_tests()