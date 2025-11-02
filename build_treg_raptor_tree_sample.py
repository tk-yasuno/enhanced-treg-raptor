#!/usr/bin/env python3
"""
Enhanced Treg RAPTOR Tree Builder
拡張Treg分化語彙システムを統合したRAPTORツリー構築

Features:
- 7-layer Treg differentiation hierarchy
- Clinical marker support (CD127low, TSDR, nTreg/iTreg)
- GPU-accelerated embedding generation
- Automatic level determination and labeling
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# パス設定
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# 拡張Treg語彙システムをインポート
from enhanced_treg_vocab import (
    ENHANCED_LEVEL_COLOR_MAPPING,
    TREG_DIFFERENTIATION_VOCAB,
    determine_treg_level,
    generate_enhanced_treg_label
)

# 既存のRAPTORビルダーをインポート
try:
    from true_raptor_builder import TrueRAPTORTree, RAPTORNode
except ImportError:
    print("Warning: Could not import TrueRAPTORTree. Using simplified version...")
    TrueRAPTORTree = None
    RAPTORNode = None


class EnhancedTregRAPTORBuilder:
    """拡張Treg分化階層を使用したRAPTORツリー構築"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RAPTOR Tree初期化
        if TrueRAPTORTree:
            self.raptor = TrueRAPTORTree()
        else:
            print("⚠️ TrueRAPTORTree not available. Creating stub...")
            self.raptor = None
        
        self.tree_data = {}
        
    def create_sample_treg_documents(self) -> List[str]:
        """サンプルTreg分化経路文書を生成"""
        
        documents = [
            # Level 0: HSC
            "Hematopoietic stem cells (HSC) in the bone marrow possess self-renewal capacity and multipotency. "
            "These Lin- Sca-1+ c-Kit+ CD150+ cells give rise to all blood cell lineages through asymmetric division.",
            
            "HSC niche in bone marrow provides critical signals for stem cell maintenance. "
            "SCF, TPO, and CXCL12 regulate HSC quiescence and proliferation through specific receptor interactions.",
            
            # Level 1: CLP
            "Common lymphoid progenitors (CLP) express IL-7R and Flt3, marking commitment to lymphoid lineage. "
            "These cells lose myeloid potential while retaining capacity to generate B cells, T cells, and NK cells.",
            
            "IL-7 signaling through IL-7R is essential for CLP survival and proliferation. "
            "Flt3 ligand promotes lymphoid lineage expansion and maintains progenitor pool.",
            
            # Level 2: CD4+ T cells
            "CD4+ T helper cells recognize peptide antigens presented on MHC class II molecules. "
            "Positive and negative thymic selection ensures proper TCR repertoire formation.",
            
            "Thymic selection produces CD4+CD8- single positive T cells with diverse TCR specificities. "
            "Cortical thymic epithelial cells mediate positive selection through MHC-II interactions.",
            
            # Level 3: CD4+CD25+CD127low
            "Human Treg identification requires CD4+CD25high CD127low phenotype as clinical standard. "
            "IL-2Rα (CD25) high expression combined with IL-7Rα (CD127) low expression distinguishes Treg from activated effector T cells.",
            
            "CD127 low expression serves as critical marker for human Treg purity assessment. "
            "Flow cytometry gating on CD4+CD25+CD127low identifies regulatory T cell population with high specificity.",
            
            "Clinical Treg isolation protocols use CD25+CD127low markers for therapeutic cell preparation. "
            "Magnetic bead separation based on these markers achieves >95% Treg purity.",
            
            # Level 4: nTreg (thymic origin)
            "Natural Treg (nTreg) develop in thymus through recognition of self-antigens presented by AIRE+ medullary epithelial cells. "
            "These Helios+ Nrp1+ cells possess high-affinity TCR for self-peptide-MHC complexes.",
            
            "Thymic Treg selection requires intermediate TCR affinity to self-antigens. "
            "AIRE-dependent expression of tissue-restricted antigens drives nTreg repertoire formation.",
            
            "Helios expression marks thymic-derived Treg with stable suppressive function. "
            "Neuropilin-1 (Nrp1) serves as additional marker distinguishing nTreg from peripherally induced Treg.",
            
            # Level 4: iTreg (peripheral origin)
            "Peripheral induced Treg (iTreg) develop in gut-associated lymphoid tissue through TGF-beta exposure. "
            "Retinoic acid from intestinal dendritic cells synergizes with TGF-beta to drive Foxp3 induction.",
            
            "iTreg conversion from naive CD4+ T cells requires TGF-beta and antigen stimulation. "
            "These Helios- cells maintain mucosal tolerance to dietary antigens and commensal bacteria.",
            
            "Gut microbiota metabolites promote peripheral Treg induction through TGF-beta-dependent mechanisms. "
            "Short-chain fatty acids enhance iTreg differentiation in intestinal lamina propria.",
            
            # Level 5: Foxp3+ stable Treg
            "Stable Foxp3+ Treg exhibit TSDR demethylation at CNS2 region ensuring epigenetic stability. "
            "CD45RA+ resting Treg represent bona fide suppressive population with stable lineage commitment.",
            
            "TSDR demethylation distinguishes stable Treg from transiently activated CD4+ T cells. "
            "Bisulfite sequencing analysis confirms CNS2 demethylation status for Treg purity assessment.",
            
            "Epigenetically stable Treg maintain Foxp3 expression even under inflammatory conditions. "
            "CNS2 enhancer region demethylation locks in regulatory T cell identity through chromatin remodeling.",
            
            # Level 5: Foxp3+ transient
            "Activated effector CD4+ T cells transiently express Foxp3 upon strong TCR stimulation. "
            "These CD45RO+ cells lack suppressive function despite temporary Foxp3 positivity.",
            
            "Transient Foxp3 expression in activated T cells shows TSDR methylation pattern. "
            "Activation-induced Foxp3 does not confer regulatory phenotype or suppressive capability.",
            
            "TCR stimulation induces transient Foxp3 in non-regulatory CD4+ T cells. "
            "TSDR methylation status discriminates true Treg from activated Foxp3+ effector cells.",
            
            # Level 6: Functional Treg
            "Functional Treg suppress immune responses through IL-10 and TGF-beta production. "
            "CTLA-4 mediates contact-dependent suppression by competing for CD80/CD86 costimulation.",
            
            "IL-10-producing Treg control intestinal inflammation and maintain mucosal homeostasis. "
            "TGF-beta secretion by Treg induces iTreg differentiation and amplifies regulatory responses.",
            
            "CTLA-4 expression enables Treg-mediated suppression through multiple mechanisms. "
            "LAG-3 and PD-1 contribute to multi-layered immunosuppressive function.",
            
            "Suppressive Treg produce IL-35 to inhibit effector T cell proliferation. "
            "Contact-dependent mechanisms include granzyme B-mediated cytolysis and metabolic disruption.",
            
            # Mixed content for testing
            "CD4+CD25+CD127low Foxp3+ Treg with stable TSDR demethylation produce IL-10 for immunosuppression. "
            "These cells represent functionally competent regulatory T cells with multiple suppressive mechanisms.",
            
            "Thymic nTreg and peripheral iTreg both contribute to immune tolerance. "
            "TGF-beta drives iTreg conversion while AIRE-dependent selection generates nTreg in thymus.",
        ]
        
        return documents
    
    def build_treg_raptor_tree(self, documents: Optional[List[str]] = None):
        """拡張Treg階層を使用したRAPTORツリー構築"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("="*80)
        print("Enhanced Treg RAPTOR Tree Builder")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 文書の準備
        if documents is None:
            print("\n📄 Creating sample Treg differentiation documents...")
            documents = self.create_sample_treg_documents()
        
        print(f"  Total documents: {len(documents)}")
        
        # 階層判定と統計
        print("\n🔍 Analyzing document levels...")
        level_counts = {}
        doc_levels = []
        
        for i, doc in enumerate(documents):
            level = determine_treg_level(doc)
            doc_levels.append(level)
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("\n📊 Document distribution by level:")
        for level in sorted(level_counts.keys()):
            level_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
            count = level_counts[level]
            percentage = (count / len(documents)) * 100
            print(f"  Level {level} ({level_info['name']}): {count} docs ({percentage:.1f}%)")
        
        # RAPTORツリー構築
        if self.raptor:
            print("\n🌳 Building RAPTOR tree...")
            start_time = time.time()
            
            try:
                # ドキュメントIDを生成
                document_ids = [f"doc_{i}" for i in range(len(documents))]
                self.raptor.build_raptor_tree(documents, document_ids)
                tree = self.raptor.nodes  # ノード辞書を取得
                elapsed = time.time() - start_time
                
                print(f"  ✓ Tree built in {elapsed:.2f} seconds")
                
                # ツリー統計
                if tree:
                    self._analyze_tree_structure(tree, doc_levels)
                
                # 保存
                output_file = self.results_dir / f"enhanced_treg_raptor_{timestamp}.json"
                
                # RAPTORNodeを辞書に変換
                tree_dict = {}
                for node_id, node in tree.items():
                    tree_dict[node_id] = {
                        'node_id': str(node.node_id),
                        'parent_id': str(node.parent_id) if node.parent_id else None,
                        'children': [str(c) for c in node.children],
                        'level': int(node.level),
                        'content': node.content[:500],  # 内容を短縮
                        'summary': node.summary[:500] if node.summary else '',
                        'is_leaf': bool(node.is_leaf),
                        'cluster_id': int(node.cluster_id) if node.cluster_id is not None else None,
                        'cluster_size': int(node.cluster_size),
                        'source_documents': [str(d) for d in node.source_documents]
                    }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'total_documents': len(documents),
                        'doc_levels': doc_levels,
                        'level_distribution': level_counts,
                        'total_nodes': len(tree_dict),
                        'tree_nodes': tree_dict
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Results saved to: {output_file}")
                
            except Exception as e:
                print(f"\n❌ Error building tree: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️ RAPTOR tree builder not available. Skipping tree construction.")
            
            # レベル分析のみ保存
            output_file = self.results_dir / f"enhanced_treg_analysis_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'documents': documents,
                    'doc_levels': doc_levels,
                    'level_distribution': level_counts
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Analysis saved to: {output_file}")
        
        print("\n" + "="*80)
        print("Enhanced Treg RAPTOR Tree Builder - Complete")
        print("="*80)
    
    def _analyze_tree_structure(self, tree: Dict, doc_levels: List[int]):
        """ツリー構造を分析"""
        print("\n📈 Tree structure analysis:")
        
        # ノード数カウント
        def count_nodes(node, depth=0):
            if not node or not isinstance(node, dict):
                return {'total': 0, 'max_depth': depth}
            
            clusters = node.get('clusters', {})
            if not clusters:
                return {'total': 1, 'max_depth': depth}
            
            total = 1
            max_depth = depth
            
            for cluster_data in clusters.values():
                if isinstance(cluster_data, dict) and 'children' in cluster_data:
                    child_stats = count_nodes(cluster_data['children'], depth + 1)
                    total += child_stats['total']
                    max_depth = max(max_depth, child_stats['max_depth'])
            
            return {'total': total, 'max_depth': max_depth}
        
        stats = count_nodes(tree)
        print(f"  Total nodes: {stats['total']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Leaf documents: {len(doc_levels)}")


def main():
    """メイン実行関数"""
    
    builder = EnhancedTregRAPTORBuilder()
    builder.build_treg_raptor_tree()


if __name__ == "__main__":
    main()
