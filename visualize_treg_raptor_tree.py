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
import re
import math
import numpy as np
from collections import Counter
from enhanced_treg_vocab import (
    ENHANCED_LEVEL_COLOR_MAPPING, 
    determine_treg_level,
    TREG_DIFFERENTIATION_VOCAB
)

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# Treg分野でのストップワード（一般的すぎて意味が薄い単語）
TREG_STOP_WORDS = {
    # 一般的な英単語
    'and', 'the', 'for', 'with', 'from', 'that', 'this', 'which', 'these', 'those',
    'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'can', 'may',
    'also', 'such', 'than', 'more', 'most', 'very', 'well', 'much', 'many',
    'their', 'there', 'where', 'when', 'what', 'how', 'who', 'why',
    
    # Treg分野で頻出だが一般的すぎる単語
    'cell', 'cells', 'immune', 'immun', 'expression', 'expressed', 'express',
    'response', 'activity', 'function', 'level', 'levels', 'role', 'through',
    'involved', 'associated', 'related', 'pathway', 'signaling', 'signal',
    'development', 'differentiation', 'activation', 'inhibition', 'regulation',
    'via', 'including', 'between', 'during', 'after', 'before', 'however',
    'therefore', 'thus', 'moreover', 'furthermore', 'additionally',
    'could', 'would', 'should', 'might', 'must',
    
    # 短すぎる単語（3文字以下は一般的に意味が薄い）
    'by', 'in', 'on', 'at', 'to', 'of', 'or', 'as', 'is', 'it', 'an', 'be',
    'we', 'us', 'our', 'not', 'all', 'but', 'one', 'two', 'etc', 'vs',
    
    # 数字・記号類
    'fig', 'figure', 'table', 'ref', 'see', 'shown', 'data', 'study', 'studies',
    'analysis', 'examined', 'observed', 'reported', 'found', 'demonstrated',
    
    # 日本語の一般的な単語
    'ため', 'こと', 'もの', 'ある', 'いる', 'する', 'なる', 'による', 'において',
    '的', '性', '化', '用', '法', '体', '型', '系', '値', '度'
}

def is_meaningful_keyword(word: str, min_length: int = 4) -> bool:
    """
    意味のあるキーワードかどうかを判定
    
    Args:
        word: チェックする単語
        min_length: 最小文字数（デフォルト: 4）
    
    Returns:
        True if meaningful, False otherwise
    """
    word_lower = word.lower()
    
    # ストップワードチェック
    if word_lower in TREG_STOP_WORDS:
        return False
    
    # 長さチェック（4文字未満は除外）
    if len(word) < min_length:
        return False
    
    # 数字のみは除外
    if word.isdigit():
        return False
    
    # 特殊文字のみは除外
    if not any(c.isalnum() for c in word):
        return False
    
    return True

def extract_keywords_from_text(text: str, top_n: int = 3, depth: int = 0) -> list:
    """
    テキストから重要キーワードを抽出（Treg分化ドメイン特化）
    
    深さに応じて優先キーワードカテゴリを変更：
    - depth=0: Level 0 (HSC)
    - depth=1: Level 1 (CLP)
    - depth=2: Level 2 (CD4+T)
    - depth=3: Level 3 (CD25+CD127low)
    - depth=4: Level 4 (nTreg/iTreg)
    - depth=5: Level 5 (Foxp3+)
    - depth=6: Level 6 (Functional)
    
    Args:
        text: 抽出対象のテキスト
        top_n: 抽出するキーワード数
        depth: ノードの深さ（階層レベル）
    """
    keywords = []
    text_lower = text.lower()
    
    # 深さに応じた優先キーワードセットを選択
    level_vocab_map = {
        0: 'hsc_level',
        1: 'clp_level',
        2: 'cd4_t_level',
        3: 'cd25_high_cd127_low_level',
        4: 'treg_origin_level',
        5: 'foxp3_treg_level',
        6: 'functional_treg_level'
    }
    
    vocab_key = level_vocab_map.get(depth % 7, 'hsc_level')
    vocab_entry = TREG_DIFFERENTIATION_VOCAB.get(vocab_key, {})
    
    # 優先キーワードの抽出
    priority_keywords = []
    
    # ネストされた辞書構造の場合は展開
    if isinstance(vocab_entry, dict):
        for key, value in vocab_entry.items():
            if isinstance(value, dict):
                # さらにネストされている場合（例: ntreg, itreg）
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, set):
                        priority_keywords.extend(list(sub_value))
            elif isinstance(value, set):
                priority_keywords.extend(list(value))
    
    # テキスト中のキーワードを検索（意味のある単語のみ）
    found_keywords = []
    for keyword in priority_keywords:
        # 意味のあるキーワードのみ抽出
        if (keyword.lower() in text_lower and 
            is_meaningful_keyword(keyword, min_length=3)):  # 語彙辞書のキーワードは3文字以上でOK
            found_keywords.append(keyword)
    
    # 頻度カウント（長い単語を優先）
    keyword_counts = Counter(found_keywords)
    # 長さでソート（同じカウントなら長い単語を優先）
    sorted_keywords = sorted(keyword_counts.items(), 
                            key=lambda x: (x[1], len(x[0])), 
                            reverse=True)
    keywords = [kw for kw, count in sorted_keywords[:top_n]]
    
    # 不足分を単純な単語分割で補完（ストップワード除外）
    if len(keywords) < top_n:
        words = re.findall(r'\w+', text)
        # 意味のある単語のみフィルタリング（4文字以上）
        meaningful_words = [w for w in words if is_meaningful_keyword(w, min_length=5)]
        word_counts = Counter(meaningful_words)
        
        # 長さ優先でソート（同じカウントなら長い単語を優先）
        sorted_words = sorted(word_counts.items(), 
                             key=lambda x: (x[1], len(x[0])), 
                             reverse=True)
        
        for word, count in sorted_words:
            if word not in keywords:
                keywords.append(word)
            if len(keywords) >= top_n:
                break
    
    return keywords[:top_n]

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

def compute_circular_layout(G: nx.DiGraph, tree_data: dict) -> dict:
    """
    楕円形レイアウトを計算（ルートを中心に横長の同心楕円状に配置）
    
    Args:
        G: NetworkXグラフ
        tree_data: ツリーデータ（レベル情報を含む）
    
    Returns:
        pos: ノードID → (x, y) 座標のマッピング
    """
    tree = tree_data.get('tree_nodes', {})
    
    # レベルごとにノードをグループ化
    levels = {}
    for node_id in G.nodes():
        if node_id in tree:
            level = tree[node_id].get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
    
    pos = {}
    max_level = max(levels.keys()) if levels else 0
    
    # 楕円の横長比率（横を縦の2倍にする）
    ellipse_ratio = 2.0
    
    for level, nodes in levels.items():
        num_nodes = len(nodes)
        
        if level == 0:
            # ルートレベル（level 0）は中心に近く配置
            radius_x = 3.0 * ellipse_ratio
            radius_y = 3.0
        else:
            # 同心楕円の半径（レベルが高いほど外側）
            base_radius = level * 10.0  # 間隔を広げて重なりを減らす
            radius_x = base_radius * ellipse_ratio  # 横方向
            radius_y = base_radius  # 縦方向
        
        # ノードを楕円周上に等間隔に配置
        for i, node_id in enumerate(sorted(nodes)):
            angle = 2 * math.pi * i / num_nodes
            x = radius_x * math.cos(angle)
            y = radius_y * math.sin(angle)
            pos[node_id] = (x, y)
    
    return pos

def visualize_tree_hierarchical(tree_data: dict, output_path: str = None, internal_only: bool = True, show_all_levels: bool = False):
    """階層的レイアウトでツリーを可視化（キーワードラベル付き）
    
    Args:
        tree_data: ツリーデータ
        output_path: 出力パス
        internal_only: Trueの場合は内部ノードのみ表示（メモリ節約）
        show_all_levels: Trueの場合はすべての階層を表示（リーフノードも含む）
    """
    # 大規模ツリーの場合は内部ノードのみを表示
    tree = tree_data.get('tree_nodes', {})
    
    # show_all_levelsがTrueの場合はすべてのノードを表示
    if show_all_levels:
        print(f"  📊 Showing all levels including leaf nodes ({len(tree)} total nodes)")
        G = create_tree_graph(tree_data)
    elif internal_only and len(tree) > 500:
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
    
    # ノードの色とサイズとラベルを決定
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    # ノードサイズの正規化のため、まずcluster_sizeを収集
    cluster_sizes = []
    for node_id in G.nodes():
        node = tree[node_id]
        cluster_size = node.get('cluster_size', 1)
        cluster_sizes.append(cluster_size)
    
    # クラスタサイズの最小・最大を取得
    min_cluster_size = min(cluster_sizes) if cluster_sizes else 1
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 1
    
    for node_id in G.nodes():
        node = tree[node_id]
        
        # Tregレベルを判定して色を決定
        treg_level = determine_node_treg_level(node)
        color_info = ENHANCED_LEVEL_COLOR_MAPPING.get(treg_level, {"color": "#CCCCCC"})
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        node_colors.append(color)
        
        # サイズ（クラスタサイズを正規化して適切な範囲に収める）
        cluster_size = node.get('cluster_size', 1)
        if num_nodes > 500:
            size = node_size  # 固定サイズ
        else:
            # サイズを200〜800の範囲に正規化（極端な大小を防ぐ）
            if max_cluster_size > min_cluster_size:
                normalized_size = (cluster_size - min_cluster_size) / (max_cluster_size - min_cluster_size)
                size = 200 + normalized_size * 600  # 200〜800の範囲
            else:
                size = 400  # すべて同じサイズの場合は中間値
        node_sizes.append(size)
        
        # キーワードラベル生成
        level = node.get('level', 0)
        content = node.get('content', '')
        summary = node.get('summary', '')
        text = content + ' ' + summary
        
        # キーワード抽出（2つ表示）
        keywords = extract_keywords_from_text(text, top_n=2, depth=level)
        
        # ラベル作成（レベル情報は表示しない、キーワード2つを改行で表示）
        if num_nodes > 500:
            # 大規模ツリーの場合は1つ目のみ短縮
            label = keywords[0][:8] if keywords else "Node"
        else:
            # 2つのキーワードを改行で表示
            if len(keywords) >= 2:
                label = f"{keywords[0]}\n{keywords[1]}"
            elif len(keywords) == 1:
                label = keywords[0]
            else:
                label = "Node"
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
    # 枠線を細く薄くして、ラベルの視認性を向上
    edge_width_node = 0.3 if num_nodes > 500 else 0.8  # 0.5→0.3, 2.5→0.8 に細く
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, 
                          edgecolors='gray', linewidths=edge_width_node)  # black→gray に薄く
    
    # ラベルを描画
    if num_nodes <= 200:  # ノード数が多い場合はラベル制限
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, 
                               font_size=font_size, font_weight='bold')
    
    # タイトルと統計情報
    title = f"Enhanced Treg RAPTOR Tree Visualization (with Keywords)\n"
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

def visualize_tree_circular(tree_data: dict, output_path: str = None, internal_only: bool = True, show_all_levels: bool = False):
    """円形レイアウトでツリーを可視化（キーワードラベル付き）
    
    Args:
        tree_data: ツリーデータ
        output_path: 出力パス
        internal_only: Trueの場合は内部ノードのみ表示（メモリ節約）
        show_all_levels: Trueの場合はすべての階層を表示（リーフノードも含む）
    """
    # 大規模ツリーの場合は内部ノードのみを表示
    tree = tree_data.get('tree_nodes', {})
    
    # show_all_levelsがTrueの場合はすべてのノードを表示
    if show_all_levels:
        print(f"  📊 Showing all levels including leaf nodes ({len(tree)} total nodes)")
        G = create_tree_graph(tree_data)
        tree_for_layout = tree_data
    elif internal_only and len(tree) > 500:
        print(f"  ⚠️  Large tree detected ({len(tree)} nodes). Showing internal nodes only.")
        filtered_tree = {
            'tree_nodes': {
                k: v for k, v in tree.items() 
                if not v.get('is_leaf', False)
            }
        }
        G = create_tree_graph(filtered_tree)
        tree_for_layout = filtered_tree
    else:
        G = create_tree_graph(tree_data)
        tree_for_layout = tree_data
    
    stats = analyze_tree_structure(tree_data)
    
    # 図のサイズを設定（楕円形に合わせて横長に）
    num_nodes = len(G.nodes())
    
    if num_nodes > 500:
        fig_width = 36  # 横長
        fig_height = 24
        node_size = 20
        font_size = 5
    else:
        fig_width = 32  # 横長
        fig_height = 20
        node_size = 50
        font_size = 7
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    
    # 円形レイアウト計算（横長楕円形）
    pos = compute_circular_layout(G, tree_for_layout)
    
    # ノードの色とサイズとラベルを決定
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    # ノードサイズの正規化のため、まずcluster_sizeを収集
    cluster_sizes = []
    for node_id in G.nodes():
        node = tree[node_id]
        cluster_size = node.get('cluster_size', 1)
        cluster_sizes.append(cluster_size)
    
    # クラスタサイズの最小・最大を取得
    min_cluster_size = min(cluster_sizes) if cluster_sizes else 1
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 1
    
    for node_id in G.nodes():
        node = tree[node_id]
        
        # Tregレベルを判定して色を決定
        treg_level = determine_node_treg_level(node)
        color_info = ENHANCED_LEVEL_COLOR_MAPPING.get(treg_level, {"color": "#CCCCCC"})
        color = color_info["color"] if isinstance(color_info, dict) else color_info
        node_colors.append(color)
        
        # サイズ（クラスタサイズを正規化して適切な範囲に収める）
        level = node.get('level', 0)
        cluster_size = node.get('cluster_size', 1)
        
        if num_nodes > 500:
            size = node_size  # 固定サイズ
        else:
            # レベル0（ルート）は少し大きめだが極端には大きくしない
            if level == 0:
                size = 1000
            else:
                # サイズを300〜900の範囲に正規化
                if max_cluster_size > min_cluster_size:
                    normalized_size = (cluster_size - min_cluster_size) / (max_cluster_size - min_cluster_size)
                    size = 300 + normalized_size * 600  # 300〜900の範囲
                else:
                    size = 600  # すべて同じサイズの場合は中間値
        node_sizes.append(size)
        
        # キーワードラベル生成
        content = node.get('content', '')
        summary = node.get('summary', '')
        text = content + ' ' + summary
        
        # キーワード抽出（2つ表示）
        keywords = extract_keywords_from_text(text, top_n=2, depth=level)
        
        # ラベル作成（レベル情報は表示しない、キーワード2つを改行で表示）
        if num_nodes > 500:
            # 大規模ツリーの場合は1つ目のみ短縮
            label = keywords[0][:8] if keywords else "Node"
        else:
            # 2つのキーワードを改行で表示
            if len(keywords) >= 2:
                label = f"{keywords[0]}\n{keywords[1]}"
            elif len(keywords) == 1:
                label = keywords[0]
            else:
                label = "Node"
        node_labels[node_id] = label
    
    # エッジを先に描画
    if len(G.edges()) > 0:
        edge_width = 0.8 if num_nodes > 500 else 2.0
        arrow_size = 8 if num_nodes > 500 else 15
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              arrows=True, arrowsize=arrow_size, arrowstyle='->', 
                              width=edge_width, alpha=0.3)
    
    # ノードを描画（枠線を細く薄くして、ラベルの視認性を向上）
    edge_width_node = 0.3 if num_nodes > 500 else 0.8  # 0.5→0.3, 2.0→0.8 に細く
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, 
                          edgecolors='gray', linewidths=edge_width_node)  # black→gray に薄く
    
    # ラベルを描画
    if num_nodes <= 200:
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, 
                               font_size=font_size, font_weight='bold')
    
    # タイトルと統計情報
    title = f"Enhanced Treg RAPTOR Tree - Elliptical Layout\n"
    title += f"Total Nodes: {stats['total_nodes']} | "
    title += f"Leaves: {stats['leaf_nodes']} | "
    title += f"Internal: {stats['internal_nodes']} | "
    title += f"Max Depth: {stats['max_depth']}"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
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
    
    ax.legend(handles=legend_patches, loc='upper right', 
             fontsize=11, framealpha=0.9)
    
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Circular visualization saved: {output_path}")
    
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
    
    # レベルごとのノード数を詳細表示
    print(f"\n📊 Nodes per Level:")
    for level in sorted(stats['levels'].keys()):
        count = stats['levels'][level]
        print(f"  Level {level}: {count} nodes")
    
    # 可視化出力ディレクトリ
    viz_dir = results_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n🎨 Creating visualizations...")
    
    # すべての階層を表示する設定
    # 注意: リーフノードを含むと2144ノードになり、可視化が重くなります
    # 内部ノード(120)のみの場合は高速で見やすいです
    show_all = False  # すべての階層を表示する場合はTrue、内部ノードのみの場合はFalse
    
    if show_all:
        print(f"\n⚠️  WARNING: Showing all {stats['total_nodes']} nodes (including {stats['leaf_nodes']} leaf nodes)")
        print(f"  This may take several minutes and produce very large images...")
    
    # 1. ツリー構造の可視化（階層型）- すべての階層
    print(f"  1. Tree structure (hierarchical) - {'All levels' if show_all else 'Internal nodes only'}...")
    tree_viz_path = viz_dir / f'tree_structure_{timestamp}.png'
    visualize_tree_hierarchical(tree_data, str(tree_viz_path), show_all_levels=show_all)
    
    # 2. ツリー構造の可視化（円形）- すべての階層
    print(f"  2. Tree structure (circular) - {'All levels' if show_all else 'Internal nodes only'}...")
    circular_viz_path = viz_dir / f'tree_structure_circular_{timestamp}.png'
    visualize_tree_circular(tree_data, str(circular_viz_path), show_all_levels=show_all)
    
    # 3. レベル分布の可視化
    print(f"  3. Level distribution...")
    dist_viz_path = viz_dir / f'level_distribution_{timestamp}.png'
    visualize_level_distribution(tree_data, str(dist_viz_path))
    
    # 4. クラスタ分析の可視化
    print(f"  4. Cluster analysis...")
    cluster_viz_path = viz_dir / f'cluster_analysis_{timestamp}.png'
    visualize_cluster_analysis(tree_data, str(cluster_viz_path))
    
    print(f"\n" + "=" * 80)
    print("Enhanced Treg RAPTOR Tree Visualization - Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
