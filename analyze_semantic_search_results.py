"""
セマンティック検索 vs キーワード検索の比較分析
Comparison Analysis: Semantic Search vs Keyword Search
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_comparison_results(json_file: Path):
    """比較結果のJSONファイルを読み込む"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_performance(results):
    """パフォーマンスデータを分析"""
    
    print("="*80)
    print("セマンティック検索 vs キーワード検索 - 性能比較レポート")
    print("Performance Comparison Report: Semantic vs Keyword Search")
    print("="*80)
    print()
    
    # データ収集
    keyword_times = []
    semantic_times = []
    hybrid_times = []
    keyword_scores = []
    semantic_scores = []
    hybrid_scores = []
    
    query_details = []
    
    for result in results:
        query_id = result['query_id']
        query = result['query']
        
        # 速度データ
        kw_time = result['keyword']['time']
        sem_time = result['semantic']['time']
        hyb_time = result['hybrid']['time']
        
        keyword_times.append(kw_time)
        semantic_times.append(sem_time)
        hybrid_times.append(hyb_time)
        
        # スコアデータ
        kw_score = result['keyword']['top_score']
        sem_score = result['semantic']['top_score']
        hyb_score = result['hybrid']['top_score']
        
        keyword_scores.append(kw_score)
        semantic_scores.append(sem_score)
        hybrid_scores.append(hyb_score)
        
        # トップ結果のノードID
        kw_top = result['keyword']['top_results'][0]['node_id'] if result['keyword']['top_results'] else 'N/A'
        sem_top = result['semantic']['top_results'][0]['node_id'] if result['semantic']['top_results'] else 'N/A'
        hyb_top = result['hybrid']['top_results'][0]['node_id'] if result['hybrid']['top_results'] else 'N/A'
        
        query_details.append({
            'id': query_id,
            'query': query[:60] + '...' if len(query) > 60 else query,
            'kw_time': kw_time,
            'sem_time': sem_time,
            'hyb_time': hyb_time,
            'kw_score': kw_score,
            'sem_score': sem_score,
            'hyb_score': hyb_score,
            'kw_top': kw_top,
            'sem_top': sem_top,
            'hyb_top': hyb_top,
            'same_top': kw_top == sem_top
        })
    
    # 1. 速度比較
    print("📊 1. 検索速度の比較 (Search Speed Comparison)")
    print("-" * 80)
    print(f"{'手法':<20} {'平均時間':>12} {'最小時間':>12} {'最大時間':>12} {'標準偏差':>12}")
    print("-" * 80)
    
    import numpy as np
    
    print(f"{'キーワード検索':<20} {np.mean(keyword_times):>11.4f}s {np.min(keyword_times):>11.4f}s {np.max(keyword_times):>11.4f}s {np.std(keyword_times):>11.4f}s")
    print(f"{'セマンティック検索':<20} {np.mean(semantic_times):>11.4f}s {np.min(semantic_times):>11.4f}s {np.max(semantic_times):>11.4f}s {np.std(semantic_times):>11.4f}s")
    print(f"{'ハイブリッド検索':<20} {np.mean(hybrid_times):>11.4f}s {np.min(hybrid_times):>11.4f}s {np.max(hybrid_times):>11.4f}s {np.std(hybrid_times):>11.4f}s")
    
    print()
    print("⚡ 速度比率 (Speed Ratio):")
    sem_ratio = np.mean(semantic_times) / np.mean(keyword_times)
    hyb_ratio = np.mean(hybrid_times) / np.mean(keyword_times)
    print(f"  セマンティック検索 / キーワード検索: {sem_ratio:.2f}x")
    print(f"  ハイブリッド検索 / キーワード検索: {hyb_ratio:.2f}x")
    
    # 2. スコア比較
    print()
    print("📈 2. 検索精度の比較 (Search Accuracy Comparison)")
    print("-" * 80)
    print(f"{'手法':<20} {'平均スコア':>12} {'最小スコア':>12} {'最大スコア':>12} {'標準偏差':>12}")
    print("-" * 80)
    
    print(f"{'キーワード検索':<20} {np.mean(keyword_scores):>12.4f} {np.min(keyword_scores):>12.4f} {np.max(keyword_scores):>12.4f} {np.std(keyword_scores):>12.4f}")
    print(f"{'セマンティック検索':<20} {np.mean(semantic_scores):>12.4f} {np.min(semantic_scores):>12.4f} {np.max(semantic_scores):>12.4f} {np.std(semantic_scores):>12.4f}")
    print(f"{'ハイブリッド検索':<20} {np.mean(hybrid_scores):>12.4f} {np.min(hybrid_scores):>12.4f} {np.max(hybrid_scores):>12.4f} {np.std(hybrid_scores):>12.4f}")
    
    print()
    print("⚠️ 注意: スコアの尺度が異なるため直接比較はできません")
    print("   キーワード検索: マッチ単語数 (整数)")
    print("   セマンティック検索: コサイン類似度 (0-1)")
    print("   ハイブリッド検索: 重み付け合成スコア (0-1)")
    
    # 3. クエリ別詳細
    print()
    print("📋 3. クエリ別の詳細比較 (Query-by-Query Details)")
    print("-" * 80)
    
    for detail in query_details:
        print(f"\nQ{detail['id']}: {detail['query']}")
        print(f"  速度: KW={detail['kw_time']:.4f}s, SEM={detail['sem_time']:.4f}s, HYB={detail['hyb_time']:.4f}s")
        print(f"  スコア: KW={detail['kw_score']:.4f}, SEM={detail['sem_score']:.4f}, HYB={detail['hyb_score']:.4f}")
        print(f"  トップ結果: KW={detail['kw_top']}, SEM={detail['sem_top']}, HYB={detail['hyb_top']}")
        
        if detail['same_top']:
            print(f"  ✅ キーワードとセマンティックで同じトップ結果")
        else:
            print(f"  ⚠️ キーワードとセマンティックで異なるトップ結果")
    
    # 4. 一致率分析
    print()
    print("🎯 4. トップ結果の一致率 (Top Result Agreement)")
    print("-" * 80)
    
    same_count = sum(1 for d in query_details if d['same_top'])
    agreement_rate = (same_count / len(query_details)) * 100
    
    print(f"キーワード検索とセマンティック検索でトップ結果が一致: {same_count}/{len(query_details)} ({agreement_rate:.1f}%)")
    
    # 5. 推奨事項
    print()
    print("💡 5. 推奨事項と結論 (Recommendations & Conclusions)")
    print("="*80)
    
    if sem_ratio < 1.5:
        print("✅ セマンティック検索の速度オーバーヘッドは許容範囲内 (<1.5x)")
    else:
        print(f"⚠️ セマンティック検索は{sem_ratio:.1f}倍遅い - キャッシュ最適化を推奨")
    
    if agreement_rate > 70:
        print(f"✅ 高い一致率({agreement_rate:.1f}%) - 両手法は類似の結果を返す")
    elif agreement_rate > 40:
        print(f"⚠️ 中程度の一致率({agreement_rate:.1f}%) - セマンティック検索が異なる視点を提供")
    else:
        print(f"❌ 低い一致率({agreement_rate:.1f}%) - 手法の違いが大きい")
    
    print()
    print("📌 総合推奨:")
    print("  1. ハイブリッド検索を推奨 - キーワードの高速性とセマンティックの精度を両立")
    print("  2. 重み調整: keyword_weight=0.4, semantic_weight=0.6 が現在の設定")
    print("  3. 埋め込みキャッシュの活用で初回以降の速度を改善")
    
    print()
    print("="*80)

def main():
    # 最新の比較結果ファイルを探す
    results_dir = Path("results")
    comparison_files = sorted(results_dir.glob("semantic_search_comparison_*.json"))
    
    if not comparison_files:
        print("❌ エラー: 比較結果ファイルが見つかりません")
        return
    
    latest_file = comparison_files[-1]
    print(f"📂 分析対象ファイル: {latest_file.name}\n")
    
    # データ読み込み
    results = load_comparison_results(latest_file)
    
    # 分析実行
    analyze_performance(results)
    
    # CSVエクスポート（オプション）
    export_to_csv = True
    if export_to_csv:
        csv_file = results_dir / f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        data = []
        for result in results:
            data.append({
                'Query_ID': result['query_id'],
                'Query': result['query'],
                'Keyword_Time': result['keyword']['time'],
                'Semantic_Time': result['semantic']['time'],
                'Hybrid_Time': result['hybrid']['time'],
                'Keyword_Score': result['keyword']['top_score'],
                'Semantic_Score': result['semantic']['top_score'],
                'Hybrid_Score': result['hybrid']['top_score'],
                'Keyword_Top': result['keyword']['top_results'][0]['node_id'] if result['keyword']['top_results'] else 'N/A',
                'Semantic_Top': result['semantic']['top_results'][0]['node_id'] if result['semantic']['top_results'] else 'N/A',
                'Hybrid_Top': result['hybrid']['top_results'][0]['node_id'] if result['hybrid']['top_results'] else 'N/A'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 CSV出力: {csv_file}")

if __name__ == "__main__":
    main()
