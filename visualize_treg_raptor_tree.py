"""
Enhanced Treg RAPTOR Tree Visualization
制御性T細胞分化に特化したRAPTORツリーの可視化
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
from datetime import datetime
from enhanced_treg_vocab import ENHANCED_LEVEL_COLOR_MAPPING, determine_treg_level

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

def load_raptor_tree(json_path: str) -> dict:
    """RAPTORツリーのJSONファイルを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_tree_structure(tree_data: dict) -> dict:
    """ツリー構造の統計情報を分析"""
    tree = tree_data.get('tree_nodes', {})
    
    stats = {
        'total_nodes': len(tree),
        'leaf_nodes': sum(1 for node in tree.values() if node.get('is_leaf', False)),
        'internal_nodes': sum(1 for node in tree.values() if not node.get('is_leaf', False)),
        'max_depth': max((node.get('level', 0) for node in tree.values()), default=0),
        'levels': {},
        'clusters': {}
    }
    
    # レベルごとのノード数
    for node in tree.values():
        level = node.get('level', 0)
        stats['levels'][level] = stats['levels'].get(level, 0) + 1
        
        # クラスタ情報
        cluster_id = node.get('cluster_id')
        if cluster_id is not None:
            stats['clusters'][cluster_id] = stats['clusters'].get(cluster_id, 0) + 1
    
    return stats

def create_tree_graph(tree_data: dict) -> nx.DiGraph:
    """ツリーデータからネットワークグラフを作成"""
    G = nx.DiGraph()
    tree = tree_data.get('tree_nodes', {})
    
    # すべてのノードを追加（内部ノードのみ）
    for node_id, node in tree.items():
        G.add_node(node_id, **node)
    
    # エッジ追加（親→子）
    edge_count = 0
    for node_id, node in tree.items():
        children = node.get('children', [])
        for child_id in children:
            # 子ノードがtree_nodesに存在する場合のみエッジを追加
            if child_id in tree:
                G.add_edge(node_id, child_id)
                edge_count += 1
    
    print(f"  Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G

def determine_node_treg_level(node: dict) -> int:
    """ノードの内容からTregレベルを判定"""
    content = node.get('content', '')
    summary = node.get('summary', '')
    text = content + ' ' + summary
    
    result = determine_treg_level(text)
    # determine_treg_levelは辞書を返すか整数を返すか確認
    if isinstance(result, dict):
        return result['level']
    return result

def visualize_tree_hierarchical(tree_data: dict, output_path: str = None, internal_only: bool = True):
    """階層的レイアウトでツリーを可視化
    
    Args:
        tree_data: ツリーデータ
        output_path: 出力パス
        internal_only: Trueの場合は内部ノードのみ表示（メモリ節約）
    """
    # 大規模ツリーの場合は内部ノードのみを表示
    tree = tree_data.get('tree_nodes', {})
    if internal_only and len(tree) > 500:
        print(f"  ⚠️  Large tree detected ({len(tree)} nodes). Showing internal nodes only.")
        filtered_tree = {
            'tree_nodes': {
                k: v for k, v in tree.items() 
                if not v.get('is_leaf', False)
            }
        }
        G = create_tree_graph(filtered_tree)
    else:
        G = create_tree_graph(tree_data)
    
    stats = analyze_tree_structure(tree_data)
    
    # 図のサイズを動的に調整（メモリ制限を考慮）
    num_nodes = len(G.nodes())
    # 大規模ツリーの場合はサイズを制限
    if num_nodes > 1000:
        fig_width = 30  # 最大30インチ
        fig_height = 20  # 最大20インチ
        node_size = 10  # ノードサイズ縮小
        font_size = 4
    elif num_nodes > 500:
        fig_width = 25
        fig_height = 18
        node_size = 20
        font_size = 5
    else:
        fig_width = max(20, min(num_nodes * 2, 30))
        fig_height = max(12, min(stats['max_depth'] * 4 + 4, 20))
        node_size = 50
        font_size = 6
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)  # DPI制限
    
    # ルートノードを探す
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    if not root_nodes:
        print("⚠️ ルートノードが見つかりません")
        # すべてのノードをルートとして扱う
        root_nodes = list(G.nodes())
    
    print(f"  Root nodes: {len(root_nodes)}, Node IDs: {root_nodes[:3]}...")
    print(f"  Total edges: {len(G.edges())}")
    if len(G.edges()) > 0:
        print(f"  Sample edges: {list(G.edges())[:3]}")
    
    # 階層的レイアウト - レベルごとに配置
    pos = {}
    level_nodes = {}
    
    for node_id in G.nodes():
        level = tree[node_id].get('level', 0)
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node_id)
    
    # 各レベルを縦に配置
    max_level = max(level_nodes.keys()) if level_nodes else 0
    for level, nodes in sorted(level_nodes.items()):
        # 上から下へ（レベル0が一番上）
        y = (max_level - level) * 3
        num_nodes_in_level = len(nodes)
        for i, node_id in enumerate(nodes):
            # 横に均等配置
            x = (i - num_nodes_in_level / 2) * 4
            pos[node_id] = (x, y)
    
    # ノードの色とサイズを決定
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node_id in G.nodes():
        node = tree[node_id]
        
        # Tregレベルを判定して色を決定
        treg_level = determine_node_treg_level(node)
        color_info = ENHANCED_LEVEL_COLOR_MAPPING.get(treg_level, {"color": "#CCCCCC"})
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        node_colors.append(color)
        
        # サイズ（クラスタサイズに基づく、大規模ツリーでは縮小）
        cluster_size = node.get('cluster_size', 1)
        if num_nodes > 500:
            size = node_size  # 固定サイズ
        else:
            size = 500 + cluster_size * 100
        node_sizes.append(size)
        
        # ラベル（大規模ツリーでは簡略化）
        is_leaf = node.get('is_leaf', False)
        if num_nodes > 500:
            label = f"L{treg_level}"  # シンプル
        else:
            node_type = 'Leaf' if is_leaf else 'Internal'
            label = f"L{treg_level}\n{node_type}\n({cluster_size})"
        node_labels[node_id] = label
    
    # エッジを先に描画（ノードの下に配置）
    if len(G.edges()) > 0:
        edge_width = 1.0 if num_nodes > 500 else 3.5
        arrow_size = 5 if num_nodes > 500 else 25
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', 
                              arrows=True, arrowsize=arrow_size, arrowstyle='->', 
                              width=edge_width, alpha=0.5 if num_nodes > 500 else 1.0, 
                              min_source_margin=5, min_target_margin=5)
    
    # ノードを描画（エッジの上に重ねて描画）
    edge_width_node = 0.5 if num_nodes > 500 else 2.5
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, 
                          edgecolors='black', linewidths=edge_width_node)
    
    # ラベルを描画（大規模ツリーでは縮小）
    if num_nodes <= 500:  # ノード数が多い場合はラベル非表示
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, 
                               font_size=font_size, font_weight='bold')
    
    # タイトルと統計情報
    title = f"Enhanced Treg RAPTOR Tree Visualization\n"
    title += f"Total Nodes: {stats['total_nodes']} | "
    title += f"Leaves: {stats['leaf_nodes']} | "
    title += f"Internal: {stats['internal_nodes']} | "
    title += f"Max Depth: {stats['max_depth']}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 凡例（Tregレベル）
    legend_patches = []
    level_names = [
        "HSC (造血幹細胞)",
        "CLP (リンパ球前駆細胞)",
        "CD4+T (ナイーブT細胞)",
        "CD4+CD25+CD127low",
        "nTreg (胸腺由来)",
        "Foxp3+Treg (発現確認)",
        "Functional Treg (機能確認)",
        "iTreg (末梢誘導)"
    ]
    
    for level in range(7):
        color_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        label = f"L{level}: {level_names[level]}"
        patch = mpatches.Patch(color=color, label=label)
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, loc='upper left', 
             bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    
    ax.axis('off')
    plt.tight_layout()
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Visualization saved: {output_path}")
    
    plt.show()

def visualize_level_distribution(tree_data: dict, output_path: str = None):
    """Tregレベルの分布を可視化"""
    tree = tree_data.get('tree_nodes', {})
    
    # レベルごとのノード数を集計
    level_counts = {i: 0 for i in range(8)}
    
    for node in tree.values():
        treg_level = determine_node_treg_level(node)
        level_counts[treg_level] += 1
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 棒グラフ
    levels = list(level_counts.keys())
    counts = list(level_counts.values())
    colors = []
    for l in levels:
        color_info = ENHANCED_LEVEL_COLOR_MAPPING[l]
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        colors.append(color)
    
    bars = ax1.bar(levels, counts, color=colors, edgecolor='black', linewidth=2)
    ax1.set_xlabel('Treg Differentiation Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax1.set_title('Node Distribution by Treg Level', fontsize=14, fontweight='bold')
    ax1.set_xticks(levels)
    ax1.grid(axis='y', alpha=0.3)
    
    # 棒の上に数値を表示
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
    
    # 円グラフ
    non_zero_levels = [l for l in levels if level_counts[l] > 0]
    non_zero_counts = [level_counts[l] for l in non_zero_levels]
    non_zero_colors = []
    for l in non_zero_levels:
        color_info = ENHANCED_LEVEL_COLOR_MAPPING[l]
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        non_zero_colors.append(color)
    
    level_names = [
        "L0: HSC",
        "L1: CLP",
        "L2: CD4+T",
        "L3: CD25+CD127low",
        "L4: nTreg",
        "L5: Foxp3+",
        "L6: Functional",
        "L7: iTreg"
    ]
    
    labels = [level_names[l] for l in non_zero_levels]
    
    wedges, texts, autotexts = ax2.pie(non_zero_counts, labels=labels, colors=non_zero_colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontweight': 'bold'})
    
    ax2.set_title('Percentage Distribution by Treg Level', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Distribution chart saved: {output_path}")
    
    plt.show()

def visualize_cluster_analysis(tree_data: dict, output_path: str = None):
    """クラスタ分析の可視化"""
    tree = tree_data.get('tree_nodes', {})
    stats = analyze_tree_structure(tree_data)
    
    # クラスタ情報を収集
    clusters = {}
    for node_id, node in tree.items():
        cluster_id = node.get('cluster_id')
        if cluster_id is not None:
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'nodes': [],
                    'levels': [],
                    'sizes': []
                }
            clusters[cluster_id]['nodes'].append(node_id)
            clusters[cluster_id]['levels'].append(determine_node_treg_level(node))
            clusters[cluster_id]['sizes'].append(node.get('cluster_size', 1))
    
    if not clusters:
        print("⚠️ クラスタ情報が見つかりません")
        return
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # クラスタサイズ分布
    cluster_ids = list(clusters.keys())
    cluster_sizes = [len(clusters[c]['nodes']) for c in cluster_ids]
    
    ax1.bar(range(len(cluster_ids)), cluster_sizes, color='skyblue', 
           edgecolor='black', linewidth=2)
    ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(cluster_ids)))
    ax1.set_xticklabels(cluster_ids)
    ax1.grid(axis='y', alpha=0.3)
    
    # クラスタごとのTregレベル分布
    level_by_cluster = {}
    for cluster_id, data in clusters.items():
        level_counts = {i: 0 for i in range(8)}
        for level in data['levels']:
            level_counts[level] += 1
        level_by_cluster[cluster_id] = level_counts
    
    # スタック棒グラフ
    bottom = [0] * len(cluster_ids)
    for level in range(7):
        heights = [level_by_cluster[c][level] for c in cluster_ids]
        color_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        ax2.bar(range(len(cluster_ids)), heights, bottom=bottom,
               color=color, label=f'Level {level}', edgecolor='black', linewidth=1)
        bottom = [b + h for b, h in zip(bottom, heights)]
    
    ax2.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_title('Treg Level Distribution per Cluster', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(cluster_ids)))
    ax2.set_xticklabels(cluster_ids)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Cluster analysis saved: {output_path}")
    
    plt.show()

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("Enhanced Treg RAPTOR Tree Visualization")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 最新のRAPTORツリーファイルを検索
    results_dir = Path(__file__).parent / 'results'
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    json_files = list(results_dir.glob('enhanced_treg_raptor_*.json'))
    if not json_files:
        print(f"❌ No RAPTOR tree JSON files found in {results_dir}")
        return
    
    # 最新のファイルを選択
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📂 Loading RAPTOR tree: {latest_file.name}")
    
    # ツリーデータ読み込み
    tree_data = load_raptor_tree(str(latest_file))
    print(f"  ✓ Loaded {len(tree_data.get('tree_nodes', {}))} nodes")
    
    # 統計情報表示
    stats = analyze_tree_structure(tree_data)
    print(f"\n📊 Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")
    print(f"  Internal nodes: {stats['internal_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Clusters: {len(stats['clusters'])}")
    
    # 可視化出力ディレクトリ
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n🎨 Creating visualizations...")
    
    # 1. ツリー構造の可視化
    print(f"  1. Tree structure...")
    tree_viz_path = viz_dir / f'tree_structure_{timestamp}.png'
    visualize_tree_hierarchical(tree_data, str(tree_viz_path))
    
    # 2. レベル分布の可視化
    print(f"  2. Level distribution...")
    dist_viz_path = viz_dir / f'level_distribution_{timestamp}.png'
    visualize_level_distribution(tree_data, str(dist_viz_path))
    
    # 3. クラスタ分析の可視化
    print(f"  3. Cluster analysis...")
    cluster_viz_path = viz_dir / f'cluster_analysis_{timestamp}.png'
    visualize_cluster_analysis(tree_data, str(cluster_viz_path))
    
    print(f"\n" + "=" * 80)
    print("Enhanced Treg RAPTOR Tree Visualization - Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
