"""簡化的聯邦學習伺服器，整合區塊鏈功能，使用 PyTorch。"""

import os
import sys
from pathlib import Path

# 添加專案根目錄到路徑以啟用相對引入
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl.server import BlockchainFlowerServer

# 定義用於 MNIST 的 CNN 模型
class MnistCNN(nn.Module):
    """為 MNIST 設計的簡單 CNN 模型"""

    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def main():
    """主函數執行聯邦學習服務器。"""
    # 基本設定
    contract_address = os.getenv("CONTRACT_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    server_address = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    use_krum = os.getenv("USE_KRUM", "true").lower() == "true"
    byzantine_threshold = int(os.getenv("BYZANTINE_THRESHOLD", "1"))
    
    # 創建初始模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initial_model = MnistCNN().to(device)
    
    # 創建區塊鏈聯邦學習服務器
    server = BlockchainFlowerServer(
        initial_model=initial_model,
        contract_address=contract_address,
        use_krum=use_krum,
        byzantine_threshold=byzantine_threshold
    )
    
    # 啟動聯邦學習過程
    server.run_federated_learning(
        num_rounds=num_rounds,
        num_clients=5,
        min_clients=2,
        server_address=server_address
    )
    
    print(f"聯邦學習完成，模型已儲存到區塊鏈和 IPFS。")

if __name__ == "__main__":
    main()