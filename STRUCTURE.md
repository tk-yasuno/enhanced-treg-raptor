# Enhanced Treg RAPTOR - プロジェクト構成

## ディレクトリ構造

```
enhanced-treg-raptor/
│
├── .git/                              # Git リポジトリ
├── .gitignore                         # Git除外設定
│
├── README.md                          # 完全ドキュメント（29KB）
├── README_PROJECT.md                  # プロジェクト概要
├── STRUCTURE.md                       # このファイル
│
├── requirements.txt                   # Python依存パッケージ
│
├── コアスクリプト/
│   ├── build_treg_raptor_16x.py      # メインビルドスクリプト ⭐
│   ├── true_raptor_builder.py        # RAPTORツリー実装
│   └── enhanced_treg_vocab.py        # 7層316用語の語彙定義
│
├── 分析・可視化/
│   ├── check_clustering_stats.py     # クラスタリング統計分析
│   ├── visualize_treg_raptor_tree.py # ツリー可視化
│   └── test_enhanced_treg_16x.py     # 統合テスト
│
├── サンプル/
│   └── build_treg_raptor_tree_sample.py  # サンプル実装
│
├── data/                              # データディレクトリ
│   └── enhanced_treg_test_results/
│       └── test_results_*.json       # テスト結果
│
└── results/                           # 実行結果（.gitignore）
    ├── enhanced_treg_raptor_80x_*.json   # ツリーデータ
    ├── treg_documents_80x_*.json         # 文書メタデータ
    ├── treg_80x_build_*.log              # ビルドログ
    └── visualizations/
        ├── tree_structure_*.png          # ツリー構造図
        ├── level_distribution_*.png      # レベル分布図
        └── cluster_analysis_*.png        # クラスタ分析図
```

## 主要ファイル詳細

### 1. build_treg_raptor_16x.py（メインスクリプト）

**機能**:
- PubMed文献収集（E-utilities API）
- レベル判定と2段階フィルタリング
- Embedding品質検証
- RAPTORツリー構築
- 結果保存とログ出力

**主要クラス**:
```python
class EnhancedTregRaptorBuilder:
    def __init__(self, scale: int = 80)
    def collect_treg_documents_from_pubmed(self) -> List[Dict]
    def analyze_document_distribution(self, documents)
    def build_raptor_tree(self, documents) -> Dict
```

**設定パラメータ**:
```python
level_0_max = 200      # PubMed収集上限
level_0_limit = 500    # 判定後上限
max_workers = 3        # 並列収集スレッド数
```

### 2. true_raptor_builder.py（RAPTOR実装）

**機能**:
- Top-downクラスタリング
- Embedding生成（sentence-transformers）
- クラスタリング品質評価（Silhouette + DBI）
- LLM要約（OPT models）

**主要クラス**:
```python
class TrueRAPTORTree:
    def __init__(self)
    def verify_embeddings(self, documents, sample_size=10)
    def encode_text(self, text: str) -> np.ndarray
    def optimal_clusters(self, embeddings, max_k=5) -> int
    def build_raptor_tree(self, doc_texts, doc_ids)
```

**クラスタリング設定**:
```python
self.metric_weights = {
    'silhouette': 0.5,  # クラスタ内凝集度
    'dbi': 0.5,         # クラスタ間分離度
}
self.min_clusters = 2
self.max_clusters = 5  # k=2~5に制限
```

### 3. enhanced_treg_vocab.py（語彙体系）

**機能**:
- 7層階層の用語定義（316用語）
- 文脈依存レベル判定
- ラベル生成

**主要関数**:
```python
def determine_treg_level(content: str) -> int
def generate_enhanced_treg_label(content, level, cluster_id, cluster_size) -> str
```

**語彙構造**:
```python
ENHANCED_LEVEL_VOCABULARIES = {
    0: HSC_VOCABULARY,           # 68用語
    1: CLP_VOCABULARY,           # 22用語
    2: CD4_T_VOCABULARY,         # 31用語
    3: CD25_CD127LOW_VOCABULARY, # 27用語
    4: NTREG_ITREG_VOCABULARY,   # 49用語
    5: FOXP3_TREG_VOCABULARY,    # 75用語
    6: FUNCTIONAL_TREG_VOCABULARY # 44用語
}
```

### 4. visualize_treg_raptor_tree.py（可視化）

**機能**:
- ツリー構造の可視化
- レベル分布グラフ
- クラスタリング品質分析

**生成される画像**:
- `tree_structure_*.png`: NetworkX階層グラフ
- `level_distribution_*.png`: 棒グラフ
- `cluster_analysis_*.png`: メトリクス分析

**メモリ最適化**:
```python
# 500ノード超の場合は内部ノードのみ表示
if total_nodes > 500:
    internal_only = True
```

### 5. check_clustering_stats.py（統計分析）

**機能**:
- クラスタリング品質メトリクスの集計
- レベル分布の分析
- 統計レポート生成

**出力内容**:
```
- 平均Silhouette
- 平均DBI
- 平均クラスタ数
- スコア分布（min, max, std）
- ツリー構造統計
- レベル別分布
```

## ワークフロー

### 標準的な実行フロー

```bash
# 1. ツリー構築
python build_treg_raptor_16x.py
# → results/enhanced_treg_raptor_80x_YYYYMMDD_HHMMSS.json

# 2. 統計確認
python check_clustering_stats.py results/enhanced_treg_raptor_80x_*.json

# 3. 可視化
python visualize_treg_raptor_tree.py results/enhanced_treg_raptor_80x_*.json
# → results/visualizations/*.png
```

### カスタマイズフロー

#### Level 0をさらに削減したい場合

```python
# build_treg_raptor_16x.py を編集
level_0_max = 100    # 200 → 100
level_0_limit = 300  # 500 → 300
```

#### クラスター数をさらに制限したい場合

```python
# true_raptor_builder.py を編集
self.max_clusters = 3  # 5 → 3
```

#### Silhouette重視に変更したい場合

```python
# true_raptor_builder.py を編集
self.metric_weights = {
    'silhouette': 0.7,  # 0.5 → 0.7
    'dbi': 0.3,         # 0.5 → 0.3
}
```

## データフロー

```
PubMed API
    ↓
文献収集（Level別、並列）
    ↓
Level判定（enhanced_treg_vocab）
    ↓
2段階フィルタリング
  1. 収集段階: Level 0 ≤ 200
  2. 判定後: Level 0 ≤ 500
    ↓
Embedding生成（sentence-transformers）
    ↓
品質検証（verify_embeddings）
    ↓
クラスタリング（K-means）
  - k=2~5の範囲で評価
  - Silhouette 0.5 + DBI 0.5
    ↓
RAPTOR Tree構築（Top-down）
    ↓
結果保存（JSON + ログ）
    ↓
統計分析・可視化
```

## 技術スタック詳細

### 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| torch | 2.5.1+ | GPU推論基盤 |
| transformers | 4.35.0+ | OPTモデル（要約） |
| sentence-transformers | 2.2.0+ | Embedding生成 |
| numpy | 1.24.0+ | 数値計算 |
| scikit-learn | 1.3.0+ | クラスタリング・評価 |
| matplotlib | 3.7.0+ | 可視化 |
| networkx | 3.1+ | グラフ構造 |
| biopython | 1.81+ | PubMed統合 |
| requests | 2.31.0+ | API通信 |

### GPU要件

**推奨構成**:
- GPU: NVIDIA RTX 4060 Ti以上
- VRAM: 16GB以上
- CUDA: 12.1+

**メモリ使用量**:
- Embedding: ~10MB
- OPT-1.3B: ~3GB
- OPT-2.7B: ~6GB
- OPT-6.7B: ~14GB

## 設定ファイル

### .gitignore

キャッシュと大容量ファイルを除外:
```
__pycache__/
pubmed_cache/
results/*.json
results/*.log
results/visualizations/*.png
```

### requirements.txt

すべての依存パッケージを管理:
```
torch>=2.5.1
transformers>=4.35.0
sentence-transformers>=2.2.0
...
```

## 結果ファイル形式

### enhanced_treg_raptor_80x_*.json

**構造**:
```json
{
  "nodes": {
    "node_id": {
      "level": 0-6,
      "content": "text content",
      "summary": "LLM summary",
      "is_leaf": true/false,
      "cluster_id": 123,
      "embedding": [0.1, 0.2, ...],
      "source_documents": ["doc_1", "doc_2"]
    }
  },
  "metadata": {
    "total_nodes": 2245,
    "max_depth": 3,
    "build_time": 35.6,
    "clustering_stats": {...}
  }
}
```

### treg_documents_80x_*.json

**構造**:
```json
[
  {
    "id": "doc_0",
    "pmid": "12345678",
    "title": "Article title",
    "text": "Title. Abstract.",
    "determined_level": 5,
    "label": "Foxp3+Treg\nFoxp3+ stable\n(n=42)"
  }
]
```

## トラブルシューティング

### よくある問題

1. **Level 0が多すぎる**
   - `level_0_limit`を削減（500 → 300）
   - PubMedクエリに`NOT Treg`追加

2. **クラスター数が多すぎる**
   - `max_clusters`を削減（5 → 3）
   - DBI重みを増加（0.5 → 0.6）

3. **メモリ不足**
   - バッチサイズを削減（8 → 4）
   - 小さいOPTモデルを使用（6.7B → 1.3B）

4. **PubMed APIエラー**
   - レート制限遵守（0.4秒/リクエスト）
   - キャッシュディレクトリを確認

## バージョン管理

### ブランチ戦略

```
master      # 安定版（v3.0.0）
├── develop # 開発版
├── feature/xxx  # 新機能
└── hotfix/xxx   # バグ修正
```

### タグ付け

```bash
git tag -a v3.0.0 -m "Level 0 reduction version"
git tag -a v2.0.0 -m "Clustering optimization"
git tag -a v1.0.0 -m "Initial baseline"
```

## ライセンス

MIT License

---

**Last Updated**: 2025-11-02  
**Version**: 3.0.0  
**Status**: Production Ready ✅
