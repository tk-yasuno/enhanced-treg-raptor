# Enhanced Treg RAPTOR - セットアップガイド

## クイックセットアップ（5分）

### ステップ 1: リポジトリのクローン

```bash
git clone <repository-url>
cd enhanced-treg-raptor
```

### ステップ 2: Python環境のセットアップ

#### Option A: 仮想環境（推奨）

```bash
# 仮想環境作成
python -m venv venv

# 仮想環境のアクティベート
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

#### Option B: Conda環境

```bash
# Conda環境作成
conda create -n treg-raptor python=3.11 -y
conda activate treg-raptor

# PyTorchインストール（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# その他の依存パッケージ
pip install -r requirements.txt
```

### ステップ 3: 動作確認

```bash
# テスト実行
python test_enhanced_treg_16x.py
```

**期待される出力**:
```
================================================================================
Enhanced Treg Differentiation - 16x Scale Integration Test
================================================================================

TEST 1: Level Determination Accuracy
Passed: 9/10 (90.0%)

TEST 2: Enhanced Label Generation  
Passed: 4/4 (100.0%)

Overall: 4/4 tests passed ✅
```

---

## 詳細セットアップ

### システム要件

#### 必須要件
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11以上
- **RAM**: 16GB以上
- **ディスク**: 10GB以上の空き容量

#### GPU要件（推奨）
- **GPU**: NVIDIA RTX 3060以上（VRAM 12GB+）
- **CUDA**: 12.1以上
- **cuDNN**: 8.9以上

**CPUのみでも動作可能**（ただし処理時間が増加）

### 依存パッケージの詳細

#### コアライブラリ

```bash
# PyTorch（GPU版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers（Hugging Face）
pip install transformers>=4.35.0

# Sentence Transformers（Embedding）
pip install sentence-transformers>=2.2.0
```

#### データ処理・分析

```bash
# 数値計算
pip install numpy>=1.24.0

# 機械学習
pip install scikit-learn>=1.3.0

# 生物情報学
pip install biopython>=1.81
```

#### 可視化

```bash
# グラフ作成
pip install matplotlib>=3.7.0

# ネットワーク図
pip install networkx>=3.1
```

#### API通信

```bash
# HTTP通信
pip install requests>=2.31.0
```

### GPU設定の確認

```python
# GPU確認スクリプト
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**期待される出力（GPU環境）**:
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4060 Ti
GPU memory: 16.0 GB
```

---

## 初回実行

### ステップ 1: サンプル実行

```bash
# サンプルスクリプト実行（小規模）
python build_treg_raptor_tree_sample.py
```

**実行内容**:
- 少量の文献で動作確認
- 約1-2分で完了
- GPU/CPU動作を確認

### ステップ 2: 本番実行

```bash
# フルスケール実行
python build_treg_raptor_16x.py
```

**実行時間**:
- GPU（RTX 4060 Ti）: 約35-40秒
- CPU（16コア）: 約2-3分

**生成されるファイル**:
```
results/
├── enhanced_treg_raptor_80x_YYYYMMDD_HHMMSS.json
├── treg_documents_80x_YYYYMMDD_HHMMSS.json
└── treg_80x_build_YYYYMMDD_HHMMSS.log
```

### ステップ 3: 結果の確認

```bash
# 統計確認
python check_clustering_stats.py results/enhanced_treg_raptor_80x_*.json

# 可視化
python visualize_treg_raptor_tree.py results/enhanced_treg_raptor_80x_*.json
```

**生成される画像**:
```
results/visualizations/
├── tree_structure_YYYYMMDD_HHMMSS.png
├── level_distribution_YYYYMMDD_HHMMSS.png
└── cluster_analysis_YYYYMMDD_HHMMSS.png
```

---

## トラブルシューティング

### 問題 1: CUDA not available

**症状**:
```
CUDA available: False
```

**解決策**:
```bash
# CUDAバージョン確認
nvidia-smi

# PyTorchを再インストール（CUDA 12.1）
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 問題 2: ModuleNotFoundError

**症状**:
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**解決策**:
```bash
# 依存パッケージを再インストール
pip install -r requirements.txt --force-reinstall
```

### 問題 3: PubMed API rate limit

**症状**:
```
HTTP Error 429: Too Many Requests
```

**解決策**:
```python
# build_treg_raptor_16x.py の delay を増やす
time.sleep(0.4)  # → 0.6に変更
```

### 問題 4: Out of Memory（GPU）

**症状**:
```
RuntimeError: CUDA out of memory
```

**解決策**:
```python
# true_raptor_builder.py のバッチサイズを削減
batch_size = 8  # → 4に変更

# または小さいOPTモデルを使用
model_name = "facebook/opt-1.3b"  # 6.7b → 1.3b
```

### 問題 5: Level 0が多すぎる

**症状**:
```
Level 0: 800 docs (40.0%)
```

**解決策**:
```python
# build_treg_raptor_16x.py を編集
level_0_max = 100     # 200 → 100
level_0_limit = 300   # 500 → 300
```

---

## カスタマイズ

### 文献収集の調整

```python
# build_treg_raptor_16x.py

# 収集数を変更
self.scale = 80  # 80x → 100x（より多くの文献）

# Level別の上限設定
level_0_max = 200    # Level 0の収集上限
level_0_limit = 500  # Level 0の判定後上限
```

### クラスタリング戦略の変更

```python
# true_raptor_builder.py

# バランス戦略（デフォルト）
self.metric_weights = {
    'silhouette': 0.5,
    'dbi': 0.5,
}

# Silhouette重視
self.metric_weights = {
    'silhouette': 0.7,
    'dbi': 0.3,
}

# DBI重視
self.metric_weights = {
    'silhouette': 0.3,
    'dbi': 0.7,
}
```

### クラスター数範囲の変更

```python
# true_raptor_builder.py

self.min_clusters = 2  # 最小クラスター数
self.max_clusters = 5  # 最大クラスター数

# より細かいクラスタリング
self.max_clusters = 7  # 5 → 7

# より粗いクラスタリング
self.max_clusters = 3  # 5 → 3
```

---

## パフォーマンス最適化

### GPU利用の最大化

```python
# true_raptor_builder.py

# バッチサイズを増やす（VRAM十分な場合）
batch_size = 16  # 8 → 16

# より大きいOPTモデルを使用
model_name = "facebook/opt-6.7b"  # より高品質な要約
```

### 並列処理の最適化

```python
# build_treg_raptor_16x.py

# ワーカー数を増やす（CPUコア数に応じて）
self.max_workers = 4  # 3 → 4
```

### キャッシュの活用

```bash
# PubMedキャッシュを保持（再実行時に高速化）
# .gitignore から pubmed_cache/ を削除

# または手動でキャッシュをクリア
rm -rf pubmed_cache/*
```

---

## 開発環境のセットアップ

### VS Code設定

#### 推奨拡張機能

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "ms-toolsai.vscode-jupyter-cell-tags",
    "GitHub.copilot"
  ]
}
```

#### settings.json

```json
{
  "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

### Git設定

```bash
# ユーザー情報設定
git config user.name "Your Name"
git config user.email "your.email@example.com"

# ブランチ作成
git checkout -b develop

# コミット
git add .
git commit -m "feat: Add new feature"

# プッシュ
git push origin develop
```

---

## よくある質問（FAQ）

### Q1: CPUのみで実行できますか？

**A**: はい、可能です。ただし処理時間が大幅に増加します（約5-10倍）。

### Q2: 必要なディスク容量は？

**A**: 
- プログラム: 約500MB
- 結果ファイル: 約100MB/実行
- PubMedキャッシュ: 約200MB
- **合計**: 約1GB（キャッシュ込み）

### Q3: インターネット接続は必要ですか？

**A**: 
- 初回実行: 必要（PubMed API、モデルダウンロード）
- 2回目以降: キャッシュがあれば不要

### Q4: 商用利用は可能ですか？

**A**: MITライセンスのため、商用利用可能です。

### Q5: 結果の再現性は保証されますか？

**A**: はい。`random.seed(42)`で乱数を固定しているため、同じ入力で同じ結果が得られます。

---

## サポート

### 問題報告

GitHub Issuesで報告してください:
- バグ報告
- 機能リクエスト
- ドキュメント改善

### コミュニティ

- [GitHub Discussions](リンク)
- [Slack Channel](リンク)

---

## 次のステップ

1. ✅ セットアップ完了
2. 📖 [README.md](README.md)で詳細を学習
3. 🔬 [STRUCTURE.md](STRUCTURE.md)でプロジェクト構造を理解
4. 🚀 実際のデータで実行
5. 📊 結果を分析
6. 🎨 可視化をカスタマイズ

---

**Last Updated**: 2025-11-02  
**Version**: 3.0.0  
**Setup Time**: ~5分
