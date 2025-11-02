#!/usr/bin/env python3
"""クラスタリング統計確認スクリプト"""
import json
from pathlib import Path

# 最新の結果ファイルを取得
results_dir = Path("results")
json_files = list(results_dir.glob("enhanced_treg_raptor_80x_*.json"))
if not json_files:
    print("結果ファイルが見つかりません")
    exit(1)

latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
print(f"📁 結果ファイル: {latest_file.name}\n")

# JSON読み込み
with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# クラスタリング統計
stats = data.get('clustering_stats', {})
print("=" * 70)
print("📊 クラスタリング品質統計 (Silhouette 0.5 + DBI 0.5, k=2~5)")
print("=" * 70)

if 'avg_silhouette' in stats:
    print(f"  ✓ 平均Silhouette: {stats['avg_silhouette']:.4f}")
    print(f"    └─ 範囲: -1 (最悪) ~ 1 (最良)")
    print(f"    └─ 高いほどクラスタ内凝集度が高く、クラスタ間分離が良い\n")
    
    print(f"  ✓ 平均DBI (Davies-Bouldin Index): {stats['avg_dbi']:.4f}")
    print(f"    └─ 範囲: 0 (最良) ~ ∞ (最悪)")
    print(f"    └─ 低いほどクラスタが密集していて分離している\n")
    
    print(f"  ✓ 平均クラスタ数: {stats['avg_k']:.1f}")
    print(f"  ✓ 評価回数: {len(stats.get('silhouette_scores', []))}")
    
    # スコア分布
    sil_scores = stats.get('silhouette_scores', [])
    dbi_scores = stats.get('dbi_scores', [])
    k_values = stats.get('selected_k_values', [])
    
    if sil_scores:
        import numpy as np
        print(f"\n  📈 スコア分布:")
        print(f"    Silhouette: min={min(sil_scores):.3f}, max={max(sil_scores):.3f}, std={np.std(sil_scores):.3f}")
        print(f"    DBI: min={min(dbi_scores):.3f}, max={max(dbi_scores):.3f}, std={np.std(dbi_scores):.3f}")
        print(f"    クラスタ数: min={min(k_values)}, max={max(k_values)}, std={np.std(k_values):.1f}")
else:
    print("  ⚠️ 統計情報が見つかりません")

# ツリー情報
print("\n" + "=" * 70)
print("🌳 RAPTOR ツリー構造")
print("=" * 70)
print(f"  総ノード数: {data.get('total_nodes', 'N/A')}")
print(f"  リーフノード数: {data.get('leaf_count', 'N/A')} (元文書)")
print(f"  内部ノード数: {data.get('total_nodes', 0) - data.get('leaf_count', 0)} (クラスタ要約)")
print(f"  ツリー深さ: {data.get('max_depth', 'N/A')}")
print(f"  構築時間: {data.get('build_time_seconds', 'N/A'):.1f}秒")

# レベル分布
level_dist = data.get('level_distribution', {})
if level_dist:
    print("\n  📊 Tregレベル分布:")
    for level, count in sorted(level_dist.items(), key=lambda x: int(x[0])):
        pct = (count / data.get('total_documents', 1)) * 100
        print(f"    Level {level}: {count:4d} docs ({pct:5.1f}%)")

print("\n" + "=" * 70)
