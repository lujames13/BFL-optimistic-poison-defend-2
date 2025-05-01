"""BlockchainFlowerServer 的 PyTorch 版本擴展，增加對 MNIST 的支援。

這個擴展需要加入到現有的 fl/server.py 文件中，在 BlockchainFlowerServer 類底部。
"""

@classmethod
def create_mnist_server(cls, contract_address, **kwargs):
    """創建專用於 MNIST 資料集的伺服器實例。
    
    Args:
        contract_address: 區塊鏈智能合約地址
        **kwargs: 其他傳遞給 BlockchainFlowerServer 構造函數的參數
        
    Returns:
        配置好的 BlockchainFlowerServer 實例
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor
    
    # 定義 MNIST 模型
    class MnistCNN(nn.Module):
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
    
    # 設置設備（GPU 如果可用，否則 CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 創建模型
    model = MnistCNN().to(device)
    
    # 可選：載入測試數據用於伺服器端評估
    x_test, y_test = None, None
    if kwargs.pop("load_test_data", False):
        # 載入 MNIST 測試數據
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        testset = MNIST("./data", train=False, download=True, transform=transform)
        
        # 轉換為 NumPy 數組
        from torch.utils.data import DataLoader
        test_loader = DataLoader(testset, batch_size=len(testset))
        
        for images, labels in test_loader:
            x_test = images.numpy()
            y_test = labels.numpy()
            break
    
    # 創建伺服器實例
    return cls(
        initial_model=model,
        contract_address=contract_address,
        x_test=x_test,
        y_test=y_test,
        **kwargs
    )

def get_model_parameters(self, model):
    """從 PyTorch 模型中獲取參數。
    
    Args:
        model: PyTorch 模型
        
    Returns:
        模型參數列表
    """
    return [val.cpu().detach().numpy() for val in model.parameters()]

def set_model_parameters(self, model, parameters):
    """將參數設置到 PyTorch 模型中。
    
    Args:
        model: PyTorch 模型
        parameters: 模型參數列表
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def evaluate_model_pytorch(self, parameters):
    """使用 PyTorch 評估模型。
    
    Args:
        parameters: 模型參數
        
    Returns:
        (loss, metrics) 元組
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    
    # 檢查是否有測試數據
    if self.x_test is None or self.y_test is None:
        return 0.0, {"accuracy": 0.0}
    
    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 設置模型參數
    self.set_model_parameters(self.model, parameters)
    self.model.to(device)
    self.model.eval()
    
    # 將 NumPy 數組轉換為 PyTorch 張量
    x_tensor = torch.tensor(self.x_test).float().to(device)
    y_tensor = torch.tensor(self.y_test).long().to(device)
    
    # 創建資料集和資料載入器
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64)
    
    # 評估模型
    correct = 0
    total = 0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = self.model(batch_x)
            loss += criterion(outputs, batch_y).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(dataloader)
    
    return float(avg_loss), {"accuracy": float(accuracy)}

def run_mnist_experiment(self, num_rounds=10, num_clients=5, byzantine_clients=None, 
                         use_krum=True, byzantine_threshold=1):
    """運行 MNIST 實驗，支持模擬拜占庭客戶端。
    
    Args:
        num_rounds: 聯邦學習輪次
        num_clients: 客戶端總數
        byzantine_clients: 拜占庭（惡意）客戶端 ID 列表
        use_krum: 是否使用 Krum 防禦機制
        byzantine_threshold: Krum 防禦機制能容忍的拜占庭客戶端數量
        
    Returns:
        實驗結果，包含最終模型和指標
    """
    import torch
    import numpy as np
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor
    from torch.utils.data import DataLoader
    
    # 更新伺服器設置
    self.use_krum = use_krum
    self.byzantine_threshold = byzantine_threshold
    
    # 創建一個新任務
    task_id = self.create_task(total_rounds=num_rounds)
    print(f"已創建任務 ID: {task_id}")
    
    # 加載測試數據用於評估
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    testset = MNIST("./data", train=False, download=True, transform=transform)
    
    # 轉換為 NumPy 數組
    test_loader = DataLoader(testset, batch_size=len(testset))
    for images, labels in test_loader:
        x_test = images.numpy()
        y_test = labels.numpy()
        break
    
    # 記錄實驗指標
    metrics_history = []
    
    # 模擬聯邦學習過程
    for round_num in range(1, num_rounds + 1):
        print(f"開始第 {round_num} 輪...")
        
        # 開始新一輪
        round_id = self.start_round(task_id)
        
        # 選擇所有客戶端參與
        self.select_clients(round_id, list(range(1, num_clients + 1)))
        
        # 等待客戶端訓練和提交更新
        print(f"等待客戶端提交更新...")
        
        # 應用防禦機制（如果啟用）
        if self.use_krum:
            selected_client_id = self.apply_defense(round_id)
            print(f"Krum 選擇了客戶端 {selected_client_id}")
        
        # 下載最新的全局模型並評估
        global_model_hash = self.blockchain.get_round_info(round_id)["globalModelHash"]
        global_model_weights = self.ipfs.download_model(global_model_hash)
        
        self.set_model_parameters(self.model, global_model_weights)
        
        # 評估模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        correct = 0
        total = 0
        loss = 0.0
        test_loader = DataLoader(testset, batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(test_loader)
        
        print(f"輪次 {round_num} 評估: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # 記錄指標
        metrics_history.append({
            "round": round_num,
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        })
        
        # 完成輪次
        self.complete_round(round_id, global_model_hash)
    
    # 完成任務
    final_model_hash = self.ipfs.upload_model(
        self.get_model_parameters(self.model),
        model_id=f"final_model_task_{task_id}",
        metadata={"task_id": task_id, "type": "final"}
    )["Hash"]
    
    self.complete_task(task_id, final_model_hash)
    print(f"實驗完成，最終模型上傳至 IPFS: {final_model_hash}")
    
    return {
        "model": self.model,
        "task_id": task_id,
        "final_model_hash": final_model_hash,
        "metrics_history": metrics_history
    }