"""
Enhanced Treg RAPTOR Tree Builder - 16x Scale
拡張Treg語彙システムを使用した16倍スケールRAPTORツリー構築
PubMed APIから実際のTreg関連文献を収集

Based on: scale_8x_tree_builder.py from 2_immune_cell_differentiation_rag
Date: 2025-11-02
"""

import sys
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# 親ディレクトリのモジュールをインポート
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from true_raptor_builder import TrueRAPTORTree
from enhanced_treg_vocab import determine_treg_level, generate_enhanced_treg_label, ENHANCED_LEVEL_COLOR_MAPPING


class EnhancedTregRAPTOR16xBuilder:
    """16倍スケール用拡張Treg RAPTOR構築システム（PubMed統合 + 並列処理）"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / 'results'
        self.cache_dir = self.base_dir / 'pubmed_cache'
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 80倍スケール設定（制約緩和版 - 実際の収集数は制約なし）
        self.scale = "80x"
        self.target_documents = 27 * 80  # 2160文書（参考値、実際は制約なし）
        
        # PubMed API設定
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = "researcher@example.com"
        
        # 並列処理設定（PubMed APIレート制限考慮）
        self.max_workers = 3  # 3 requests/sec制限のため、3スレッドに抑制
        self.request_delay = 0.4  # リクエスト間隔400ms（2.5 req/sec相当）
        
        # ログ設定
        self.setup_logging()
        
    def setup_logging(self):
        """ログシステムの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"treg_80x_build_{timestamp}.log"
        
        # ロガー設定
        self.logger = logging.getLogger('EnhancedTregRAPTOR80xBuilder')
        self.logger.setLevel(logging.INFO)
        
        # 既存ハンドラーをクリア
        self.logger.handlers.clear()
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマット設定
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラー追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Enhanced Treg RAPTOR 16x Builder Log initialized: {self.log_file}")
        
    def log_info(self, message: str):
        """情報ログ"""
        self.logger.info(message)
        
    def log_error(self, message: str):
        """エラーログ"""
        self.logger.error(message)
    
    def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
        """PubMedで論文を検索してPMIDリストを取得（レート制限対応）"""
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'email': self.email,
            'tool': 'enhanced_treg_rag'
        }
        
        try:
            time.sleep(self.request_delay)  # レート制限対策
            response = requests.get(f"{self.pubmed_base_url}esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            self.log_info(f"  ✓ Found {len(pmids)} articles for query: '{query[:50]}...'")
            return pmids
            
        except Exception as e:
            self.log_error(f"  ✗ PubMed search error: {e}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """PMIDリストから論文詳細を取得"""
        if not pmids:
            return []
        
        # キャッシュチェック
        cache_key = hashlib.md5(','.join(sorted(pmids)).encode()).hexdigest()
        cache_file = self.cache_dir / f"articles_{cache_key}.json"
        
        if cache_file.exists():
            self.log_info(f"  ✓ Loading {len(pmids)} articles from cache")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # APIから取得
        articles = []
        batch_size = 200
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                time.sleep(0.34)  # NCBI rate limit: 3 requests/second
                response = requests.get(f"{self.pubmed_base_url}efetch.fcgi", params=params, timeout=30)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                for article_elem in root.findall('.//PubmedArticle'):
                    try:
                        pmid = article_elem.findtext('.//PMID')
                        title = article_elem.findtext('.//ArticleTitle') or ""
                        
                        # Abstract取得
                        abstract_texts = article_elem.findall('.//AbstractText')
                        abstract = ' '.join([ab.text for ab in abstract_texts if ab.text]) if abstract_texts else ""
                        
                        if abstract:
                            articles.append({
                                'pmid': pmid,
                                'title': title,
                                'abstract': abstract
                            })
                    except:
                        continue
                
                self.log_info(f"  ✓ Fetched batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}")
                
            except Exception as e:
                self.log_error(f"  ✗ Fetch error for batch: {e}")
        
        # キャッシュ保存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        return articles
    
    def _fetch_level_articles(self, level: int, queries: List[str], articles_per_level: int, seen_pmids: set) -> tuple:
        """単一レベルの文献を並列で収集（スレッドセーフ、Level 0は制限あり）"""
        level_articles = []
        local_seen = set()  # ローカルの重複チェック
        
        # Level 0 (HSC)の制限値を設定（極端な偏りを防ぐ - 200件に削減）
        level_0_max = 200  # 過大を防ぐため大幅削減（400→200）
        
        # 全クエリを実行
        for query in queries:
            # Level 0の場合、制限に達したら停止
            if level == 0 and len(level_articles) >= level_0_max:
                self.log_info(f"  Level 0: Reached limit of {level_0_max} articles, stopping collection")
                break
            
            try:
                # 検索実行（1クエリあたり最大200件）
                max_results_per_query = 200
                pmids = self.search_pubmed(query, max_results=max_results_per_query)
                
                if pmids:
                    # 記事詳細取得
                    articles = self.fetch_article_details(pmids)
                    
                    # 重複除去と収集
                    for article in articles:
                        # Level 0の場合は制限チェック
                        if level == 0 and len(level_articles) >= level_0_max:
                            break
                        
                        if (article['pmid'] not in seen_pmids and 
                            article['pmid'] not in local_seen):
                            local_seen.add(article['pmid'])
                            article['expected_level'] = level
                            article['query'] = query
                            level_articles.append(article)
                
                self.log_info(f"  Level {level}: {len(level_articles)} from '{query[:40]}...'")
                
            except Exception as e:
                self.log_error(f"  ✗ Query failed for level {level}, query '{query}': {e}")
        
        return level, level_articles, local_seen
    
    def collect_treg_documents_from_pubmed(self) -> List[Dict[str, Any]]:
        """PubMedから実際のTreg関連文献を並列収集（Level 0制限、他は無制限）"""
        self.log_info(f"\n📡 Collecting Treg documents from PubMed...")
        self.log_info(f"  🚀 Parallel processing with {self.max_workers} workers")
        self.log_info(f"  ⚖️  Level 0 (HSC) limited to ~200 docs (strongly reduced), others unlimited")
        
        # 7層それぞれの検索クエリ（拡張版：各レベル8クエリ）
        treg_queries = {
            0: [  # HSC - Tregとの関連を排除、HSC特異的に（8クエリ）
                "hematopoietic stem cell lineage commitment NOT regulatory",
                "Lin-Sca1+cKit+ HSC bone marrow niche",
                "HSC self-renewal quiescence NOT Treg",
                "stem cell factor SCF TPO hematopoiesis",
                "long-term HSC LT-HSC repopulation NOT lymphocyte",
                "HSC aging stress myeloid bias NOT immune",
                "bone marrow stromal cell HSC maintenance NOT Treg",
                "SLAM markers CD150 CD48 HSC identification"
            ],
            1: [  # CLP - Treg前段階のリンパ球前駆細胞に焦点（8クエリ）
                "common lymphoid progenitor IL-7R NOT Treg",
                "CLP lymphoid commitment Flt3 NOT regulatory",
                "IL7Ralpha Flt3 progenitor lymphopoiesis",
                "lymphoid lineage specification B T NK",
                "pre-pro-B cell lymphoid development NOT regulatory",
                "Ikaros Pu.1 lymphoid transcription factor",
                "thymus seeding progenitor early T lineage",
                "DN1 thymic progenitor T cell commitment"
            ],
            2: [  # CD4+ T - Treg分化前のnaive T cellに焦点（8クエリ）
                "CD4+ T cell thymic selection NOT regulatory",
                "naive CD4 T cell TCR repertoire NOT Treg",
                "positive selection MHC class II thymus",
                "CD4 single positive thymocyte NOT Foxp3",
                "conventional CD4 T cell effector Th1 Th2 Th17",
                "TCR signaling strength CD4 lineage NOT regulatory",
                "naive T cell homeostasis IL-7 survival",
                "CD62L CCR7 naive T cell lymph node homing"
            ],
            3: [  # CD4+CD25+CD127low - Treg表面マーカーに特化（8クエリ）
                "CD4+CD25+CD127low regulatory T cell",
                "CD127low IL-7Ralpha Treg surface marker",
                "CD25high CD127low Treg isolation flow cytometry",
                "IL-7R negative CD25 positive Treg phenotype",
                "CD25 IL-2 receptor alpha Treg activation",
                "GITR CD25 Treg phenotypic marker",
                "CD39 CD73 Treg ectonucleotidase suppression",
                "LAG-3 CD49b Tr1 regulatory subset"
            ],
            4: [  # nTreg/iTreg - thymic vs peripheral developmentに特化（8クエリ）
                "thymic regulatory T cell AIRE medulla",
                "peripheral induced Treg TGF-beta retinoic acid",
                "Helios positive nTreg thymic origin",
                "TSDR demethylation Treg lineage stability",
                "Foxp3 CNS2 epigenetic natural induced",
                "gut microbiota induced Treg oral tolerance",
                "vitamin A retinoic acid iTreg differentiation",
                "neuropilin-1 nTreg iTreg distinction marker"
            ],
            5: [  # Foxp3+ Treg - Foxp3転写因子機能に特化（8クエリ）
                "Foxp3 transcription factor regulatory T cell",
                "Foxp3 IPEX syndrome immune dysregulation",
                "Scurfin Foxp3 gene mutation",
                "Foxp3 CNS enhancer regulatory element",
                "Foxp3 isoform alternative splicing function",
                "Foxp3 protein interaction NFAT AP-1",
                "Foxp3 acetylation ubiquitination regulation",
                "Foxp3 Eos Helios complex transcriptional repression"
            ],
            6: [  # Functional Treg - 抑制機能メカニズムに特化（8クエリ）
                "Treg suppression CTLA-4 mechanism",
                "regulatory T cell IL-10 TGF-beta cytokine",
                "Treg effector memory CD44 CD62L",
                "tissue-resident regulatory T cell VAT muscle",
                "Treg contact-dependent suppression granzyme perforin",
                "Treg metabolic reprogramming fatty acid oxidation",
                "Treg stability plasticity inflammatory environment",
                "ex-Treg Foxp3 instability autoimmunity"
            ]
        }
        
        all_articles = []
        articles_per_level = self.target_documents // 8  # 各レベル約270件 (8 levels: 0-7)
        seen_pmids = set()
        
        # 並列処理でレベルごとに収集
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for level, queries in treg_queries.items():
                future = executor.submit(self._fetch_level_articles, level, queries, articles_per_level, seen_pmids)
                futures[future] = level
            
            # 結果を収集
            for future in as_completed(futures):
                try:
                    level, level_articles, local_seen = future.result()
                    all_articles.extend(level_articles)
                    seen_pmids.update(local_seen)
                    self.log_info(f"  ✓ Level {level} ({ENHANCED_LEVEL_COLOR_MAPPING[level]['name']}): collected {len(level_articles)} articles")
                except Exception as e:
                    level = futures[future]
                    self.log_error(f"  ✗ Level {level} collection failed: {e}")
        
        self.log_info(f"\n  ✓ Total collected: {len(all_articles)} articles from PubMed")
        return all_articles
        
        for level in range(8):
            level_articles = []
            queries = treg_queries[level]
            
            for query in queries:
                if len(level_articles) >= articles_per_level:
                    break
                
                pmids = self.search_pubmed(query, max_results=articles_per_level // len(queries) + 50)
                if pmids:
                    articles = self.fetch_article_details(pmids)
                    level_articles.extend(articles)
            
            # 重複削除
            seen_pmids = set()
            unique_articles = []
            for article in level_articles:
                if article['pmid'] not in seen_pmids:
                    seen_pmids.add(article['pmid'])
                    unique_articles.append(article)
            
            # 必要数まで取得
            level_articles = unique_articles[:articles_per_level]
            
            self.log_info(f"  Level {level}: Collected {len(level_articles)} articles")
            all_articles.extend(level_articles)
        
        self.log_info(f"  ✓ Total collected: {len(all_articles)} PubMed articles")
        return all_articles
    
    def create_treg_documents_16x(self):
        """PubMedから収集した実際の文献でTreg文書を生成"""
        self.log_info(f"\n📄 Creating {self.target_documents} Treg documents from PubMed...")
        
        # PubMedから文献を収集
        pubmed_articles = self.collect_treg_documents_from_pubmed()
        
        if len(pubmed_articles) < 100:
            self.log_error(f"⚠️ Too few articles collected: {len(pubmed_articles)}")
            return []
        
        documents = []
        
        for idx, article in enumerate(pubmed_articles):
            # タイトルとアブストラクトを結合
            text = f"{article['title']}. {article['abstract']}"
            
            # enhanced_treg_vocabでTregレベルを判定
            determined_level = determine_treg_level(text)
            
            # ラベル生成
            label = generate_enhanced_treg_label(text, determined_level, idx, 1)
            
            documents.append({
                'id': f'doc_{idx}',
                'pmid': article['pmid'],
                'title': article['title'],
                'text': text,
                'determined_level': determined_level,
                'label': label
            })
        
        self.log_info(f"  ✓ Created {len(documents)} documents from PubMed articles")
        
        # レベル判定結果の確認
        matches = len(documents)
        self.log_info(f"  ✓ Successfully processed {matches} articles with level determination")
        
        # Level 0の文書数を制限（後処理フィルタリング）
        level_0_limit = 500  # Level 0の最大文書数
        level_0_docs = [doc for doc in documents if doc['determined_level'] == 0]
        other_docs = [doc for doc in documents if doc['determined_level'] != 0]
        
        if len(level_0_docs) > level_0_limit:
            self.log_info(f"\n⚖️  Level 0 filtering: {len(level_0_docs)} → {level_0_limit} docs")
            # ランダムサンプリングで制限
            import random
            random.seed(42)  # 再現性のため
            level_0_docs = random.sample(level_0_docs, level_0_limit)
            documents = level_0_docs + other_docs
            self.log_info(f"  ✓ Filtered to {len(documents)} total documents")
        
        return documents
    
    def analyze_document_distribution(self, documents):
        """文書のTregレベル分布を分析（determined_levelを使用）"""
        self.log_info(f"\n📊 Analyzing document distribution by Treg level...")
        
        level_counts = {i: 0 for i in range(8)}
        for doc in documents:
            level = doc['determined_level']  # enhanced_treg_vocabによる判定レベルを使用
            level_counts[level] += 1
        
        total = len(documents)
        level_names = [
            "HSC",
            "CLP", 
            "CD4+T",
            "CD25+CD127low",
            "nTreg",
            "Foxp3+",
            "Functional",
            "iTreg"
        ]
        
        for level in range(8):
            count = level_counts[level]
            percentage = (count / total * 100) if total > 0 else 0
            self.log_info(f"  Level {level} ({level_names[level]:15s}): {count:3d} docs ({percentage:5.1f}%)")
        
        return level_counts
    
    def build_16x_raptor_tree(self):
        """16倍スケールでRAPTORツリーを構築"""
        
        self.log_info("🚀 ENHANCED TREG RAPTOR 80X SCALE CONSTRUCTION STARTED")
        self.log_info("=" * 70)
        self.log_info(f"Scale: {self.scale}")
        self.log_info(f"Target documents: {self.target_documents}")
        
        total_start = time.time()
        
        try:
            # Phase 1: 文書生成
            self.log_info("\n📄 Phase 1: Document Generation")
            doc_start = time.time()
            
            documents = self.create_treg_documents_16x()
            level_dist = self.analyze_document_distribution(documents)
            
            doc_time = time.time() - doc_start
            self.log_info(f"✓ Document generation completed in {doc_time:.2f}s")
            
            # Phase 2: RAPTOR初期化
            self.log_info("\n🌳 Phase 2: RAPTOR Tree Initialization")
            init_start = time.time()
            
            # TrueRAPTORTreeをTop-down戦略で初期化
            raptor = TrueRAPTORTree()
            raptor.clustering_strategy = "top_down"  # Top-downクラスタリング
            raptor.initial_clusters = 8  # Tregの8レベルに対応 (0-7: added iTreg as Level 7)
            raptor.max_cluster_size = 50  # 大規模データセット用に調整
            
            init_time = time.time() - init_start
            self.log_info(f"✓ RAPTOR initialized with top-down clustering in {init_time:.2f}s")
            
            # Embedding品質確認
            self.log_info(f"\n🔍 Phase 2.5: Verifying Embedding Quality...")
            doc_texts = [doc['text'] for doc in documents]
            embedding_stats = raptor.verify_embeddings(doc_texts, sample_size=10)
            
            # Embedding統計を表示
            if embedding_stats:
                self.log_info(f"  次元数: {embedding_stats['embedding_dim']}")
                self.log_info(f"  平均ノルム: {embedding_stats['mean_norm']:.3f} ± {embedding_stats['std_norm']:.3f}")
                self.log_info(f"  値の範囲: [{embedding_stats['min_value']:.3f}, {embedding_stats['max_value']:.3f}]")
                if 'avg_cosine_similarity' in embedding_stats:
                    self.log_info(f"  サンプル間コサイン類似度: {embedding_stats['avg_cosine_similarity']:.3f} ± {embedding_stats['std_cosine_similarity']:.3f}")
            
            # Phase 3: ツリー構築
            self.log_info(f"\n🔨 Phase 3: Building RAPTOR Tree with {len(documents)} documents...")
            build_start = time.time()
            
            doc_ids = [doc['id'] for doc in documents]
            
            raptor.build_raptor_tree(doc_texts, doc_ids)
            tree = raptor.nodes  # TrueRAPTORTreeはself.nodesにツリーを保存
            
            build_time = time.time() - build_start
            self.log_info(f"✓ Tree built in {build_time:.2f}s")
            
            # ツリー統計
            total_nodes = len(tree)
            max_depth = max((node.level for node in tree.values()), default=0)
            leaf_count = sum(1 for node in tree.values() if node.is_leaf)
            
            self.log_info(f"\n📈 Tree Structure Analysis:")
            self.log_info(f"  Total nodes: {total_nodes}")
            self.log_info(f"  Max depth: {max_depth}")
            self.log_info(f"  Leaf documents: {leaf_count}")
            self.log_info(f"  Internal nodes: {total_nodes - leaf_count}")
            
            # クラスタリング品質統計を取得
            clustering_stats = raptor.get_clustering_stats()
            
            # Phase 4: 結果保存
            self.log_info("\n💾 Phase 4: Saving Results")
            save_start = time.time()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # ツリー保存
            results = {
                'timestamp': timestamp,
                'scale': self.scale,
                'total_documents': len(documents),
                'doc_levels': [doc['determined_level'] for doc in documents],
                'doc_labels': [doc['label'] for doc in documents],
                'level_distribution': level_dist,
                'build_time_seconds': build_time,
                'total_nodes': total_nodes,
                'max_depth': max_depth,
                'leaf_count': leaf_count,
                'clustering_stats': clustering_stats,  # 追加: クラスタリング品質統計
                'tree_nodes': {}
            }
            
            # ツリーノードを辞書に変換
            for node_id, node in tree.items():
                results['tree_nodes'][node_id] = {
                    'node_id': str(node.node_id),
                    'parent_id': str(node.parent_id) if node.parent_id else None,
                    'children': [str(c) for c in node.children],
                    'level': int(node.level),
                    'content': node.content[:500],
                    'summary': node.summary[:500] if node.summary else '',
                    'is_leaf': bool(node.is_leaf),
                    'cluster_id': int(node.cluster_id) if node.cluster_id is not None else None,
                    'cluster_size': int(node.cluster_size),
                    'source_documents': [str(d) for d in node.source_documents[:30]]
                }
            
            # JSON保存
            output_path = self.results_dir / f'enhanced_treg_raptor_80x_{timestamp}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.log_info(f"✓ Results saved: {output_path.name}")
            
            # 文書メタデータ保存
            docs_path = self.results_dir / f'treg_documents_80x_{timestamp}.json'
            doc_metadata = []
            for doc in documents:
                metadata = {
                    'id': doc['id'],
                    'pmid': doc.get('pmid', ''),
                    'title': doc.get('title', ''),
                    'determined_level': doc['determined_level'],
                    'label': doc['label'],
                    'text_length': len(doc['text'])
                }
                doc_metadata.append(metadata)
            
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(doc_metadata, f, indent=2, ensure_ascii=False)
            
            self.log_info(f"✓ Document metadata saved: {docs_path.name}")
            
            save_time = time.time() - save_start
            total_time = time.time() - total_start
            
            # 最終サマリー
            self.log_info("\n✅ ENHANCED TREG RAPTOR 80X SCALE CONSTRUCTION COMPLETED")
            self.log_info("=" * 70)
            self.log_info(f"📊 Final Results Summary:")
            self.log_info(f"   Scale: {self.scale}")
            self.log_info(f"   Total execution time: {total_time:.1f}s")
            self.log_info(f"   Documents processed: {len(documents)}")
            self.log_info(f"   Tree nodes created: {total_nodes}")
            self.log_info(f"   Tree depth: {max_depth}")
            self.log_info(f"   Build time: {build_time:.1f}s")
            self.log_info(f"   Save time: {save_time:.1f}s")
            
            # クラスタリング品質統計を表示
            if clustering_stats and 'avg_silhouette' in clustering_stats:
                self.log_info(f"\n📈 Clustering Quality Metrics:")
                self.log_info(f"   Strategy: Balanced (Silhouette 0.5 + DBI 0.5), k=2~5")
                self.log_info(f"   Avg Silhouette: {clustering_stats['avg_silhouette']:.3f} (higher is better, range: -1~1)")
                self.log_info(f"   Avg DBI: {clustering_stats['avg_dbi']:.3f} (lower is better, range: 0~∞)")
                self.log_info(f"   Avg Cluster Count: {clustering_stats['avg_k']:.1f}")
                self.log_info(f"   Evaluations: {len(clustering_stats['silhouette_scores'])}")
            
            self.log_info(f"\n📁 Output Files:")
            self.log_info(f"   Tree JSON: {output_path.name}")
            self.log_info(f"   Documents: {docs_path.name}")
            self.log_info(f"   Log file: {self.log_file.name}")
            
            self.log_info(f"\n🎯 Next Steps:")
            self.log_info(f"   1. Visualize: python visualize_treg_raptor_tree.py")
            self.log_info(f"   2. Query: Use enhanced Treg vocabulary for semantic search")
            self.log_info(f"   3. Analyze: Review level distribution and clustering quality")
            
            return True
            
        except Exception as e:
            error_time = time.time() - total_start
            self.log_error(f"❌ 80X SCALE CONSTRUCTION FAILED: {str(e)}")
            self.log_error(f"   Execution time before failure: {error_time:.1f}s")
            
            import traceback
            traceback.print_exc()
            
            return False


def main():
    """メイン実行関数"""
    import sys
    import io
    # UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("🚀 ENHANCED TREG RAPTOR 80X SCALE CONSTRUCTION")
    print("=" * 70)
    
    # 16倍スケールビルダー初期化
    builder = EnhancedTregRAPTOR16xBuilder()
    
    try:
        success = builder.build_16x_raptor_tree()
        
        if success:
            print(f"\n✅ 16x Scale Treg RAPTOR Construction completed successfully!")
            print(f"📝 Detailed log: {builder.log_file}")
            print(f"📁 Output files saved in: {builder.results_dir}")
        else:
            print(f"\n❌ 80x Scale Construction failed.")
            print(f"📝 Check log file: {builder.log_file}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Construction interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
