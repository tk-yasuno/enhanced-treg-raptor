"""
Enhanced Treg Differentiation Vocabulary System
Treg分化経路の詳細階層構造

新階層構造（7層）:
Level 0: HSC (造血幹細胞)
Level 1: CLP (共通リンパ球前駆細胞) 
Level 2: CD4+ T (CD4陽性T細胞)
Level 3: CD4+CD25+ T (CD25高発現T細胞)
Level 4: nTreg/iTreg Origin (胸腺由来/末梢誘導)
Level 5: Foxp3+ Treg (Foxp3発現制御性T細胞)
Level 6: Functional Treg (サイトカイン産生機能的Treg)
"""

# 拡張された階層別色分けマッピング（7層構造）
ENHANCED_LEVEL_COLOR_MAPPING = {
    0: {
        "color": "#E74C3C", 
        "name": "HSC", 
        "description": "Hematopoietic Stem Cell",
        "markers": ["Lin-", "Sca-1+", "c-Kit+", "CD34-", "CD150+"],
        "keywords": ["stem cell", "multipotent", "self-renewal", "bone marrow"]
    },
    1: {
        "color": "#3498DB", 
        "name": "CLP", 
        "description": "Common Lymphoid Progenitor",
        "markers": ["IL-7R+", "Flt3+", "Lin-", "Sca-1+"],
        "keywords": ["lymphoid", "IL-7", "progenitor", "commitment"]
    },
    2: {
        "color": "#9B59B6", 
        "name": "CD4+T", 
        "description": "CD4+ T Helper Cell",
        "markers": ["CD4+", "TCR+", "CD3+", "CD8-"],
        "keywords": ["helper T", "MHC-II", "TCR", "thymic selection"]
    },
    3: {
        "color": "#1ABC9C", 
        "name": "CD4+CD25+CD127low", 
        "description": "CD4+CD25high CD127low T Cell",
        "markers": ["CD4+", "CD25high", "CD127low", "IL-7Rαlow", "GITR+"],
        "human_treg_markers": ["CD4+", "CD25+", "CD127low/−", "IL-7Rα low"],
        "keywords": ["CD25 high expression", "CD127 low", "IL-7R alpha low", "IL-2R", "activated T cell", "Treg precursor", "human Treg identification"]
    },
    4: {
        "color": "#F39C12", 
        "name": "nTreg/iTreg", 
        "description": "Thymic/Peripheral Origin Treg",
        "markers": {
            "nTreg": ["thymic", "Helios+", "Nrp1+", "natural"],
            "iTreg": ["peripheral", "Helios-", "induced", "antigen-driven"]
        },
        "keywords": ["thymic selection", "peripheral conversion", "TGF-beta induction", "antigen recognition"]
    },
    5: {
        "color": "#16A085", 
        "name": "Foxp3+Treg", 
        "description": "Foxp3-expressing Regulatory T Cell",
        "markers": ["Foxp3+", "CD4+", "CD25high", "CD127low", "CTLA-4+"],
        "stability_markers": {
            "stable": ["TSDR demethylation", "CNS2 demethylation", "CD45RA+", "resting Treg"],
            "transient": ["TSDR methylated", "activated effector", "CD45RO+", "temporary Foxp3"]
        },
        "keywords": ["Foxp3", "transcription factor", "epigenetic stability", "Treg identity", "TSDR", "demethylation", "CD45RA", "stable vs transient"]
    },
    6: {
        "color": "#27AE60", 
        "name": "Functional Treg", 
        "description": "Cytokine-producing Suppressive Treg",
        "markers": ["Foxp3+", "IL-10+", "TGF-beta+", "CTLA-4+"],
        "cytokines": ["IL-10", "TGF-beta", "IL-35"],
        "mechanisms": ["CTLA-4", "LAG-3", "PD-1", "contact-dependent"],
        "keywords": ["immunosuppression", "tolerance", "cytokine production", "suppressive function"]
    },
    7: {
        "color": "#95A5A6", 
        "name": "ROOT", 
        "description": "Root Node",
        "markers": [],
        "keywords": ["hierarchy root", "top level"]
,
    7: {
        "color": "#27AE60",
        "name": "iTreg (Induced)",
        "description": "Peripherally Induced Regulatory T Cell",
        "markers": ["induced", "peripheral", "TGF-β+", "RA+", "gut-associated"],
        "keywords": ["peripheral conversion", "TGF-beta induced", "retinoic acid", "oral tolerance", "gut immunity", "iTreg"],
        "origin": "peripheral",
        "stability": "variable"
    }
    }
}

# 詳細なTreg分化語彙体系
TREG_DIFFERENTIATION_VOCAB = {
    # Level 0: HSC (造血幹細胞)
    "hsc_level": {
        "japanese": {
            '造血幹細胞', 'HSC', '幹細胞', '多能性', '自己複製',
            '骨髄', '骨髄ニッチ', 'Lin-', 'Sca-1+', 'c-Kit+',
            'CD34-', 'CD150+', 'CD48-', 'SLAM', 'LSK',
            'SCF', 'TPO', 'Flt3L', 'IL-3', 'Notch', 'Wnt'
        },
        "english": {
            'hematopoietic stem cell', 'HSC', 'stem cell', 'multipotency', 
            'self-renewal', 'bone marrow', 'niche', 'quiescence',
            'Lin-', 'Sca-1+', 'c-Kit+', 'CD34-', 'CD150+', 'CD48-',
            'SCF', 'TPO', 'Flt3L', 'cytokine signaling'
        }
    },
    
    # Level 1: CLP (共通リンパ球前駆細胞)
    "clp_level": {
        "japanese": {
            'CLP', '共通リンパ球前駆細胞', 'リンパ球前駆細胞',
            'IL-7R', 'IL-7受容体', 'Flt3', 'Lin-', 'Sca-1+',
            'リンパ球系列', 'B細胞', 'T細胞', 'NK細胞',
            'IL-7', 'Flt3L', 'CXCL12', 'ストローマ細胞'
        },
        "english": {
            'common lymphoid progenitor', 'CLP', 'lymphoid progenitor',
            'IL-7R+', 'IL-7 receptor', 'Flt3+', 'lymphoid lineage',
            'B cell potential', 'T cell potential', 'NK potential',
            'IL-7 signaling', 'Flt3L', 'CXCL12', 'stromal support'
        }
    },
    
    # Level 2: CD4+ T細胞
    "cd4_t_level": {
        "japanese": {
            'CD4陽性T細胞', 'CD4+T細胞', 'ヘルパーT細胞',
            'CD4', 'CD3', 'TCR', 'T細胞受容体',
            'MHC-II', 'クラスII', '抗原提示', '胸腺選択',
            '正の選択', '負の選択', 'ダブルポジティブ',
            'Th1', 'Th2', 'Th17', 'CD28', 'CD80', 'CD86'
        },
        "english": {
            'CD4+ T cell', 'helper T cell', 'CD4 positive',
            'TCR', 'T cell receptor', 'MHC class II', 'antigen recognition',
            'thymic selection', 'positive selection', 'negative selection',
            'double positive', 'CD4+CD8+', 'single positive',
            'costimulation', 'CD28', 'B7', 'activation'
        }
    },
    
    # Level 3: CD4+CD25+ T細胞（CD25高発現 + CD127低発現）
    "cd25_high_cd127_low_level": {
        "japanese": {
            'CD25高発現', 'CD25high', 'CD25+', 'IL-2受容体',
            'IL-2Rα', 'CD127低発現', 'CD127low', 'CD127-', 'IL-7Rα低発現',
            'GITR', 'GITR陽性', '活性化T細胞', 'Treg前駆細胞', 
            'IL-2応答性', '高親和性IL-2受容体', 'CD25発現上昇',
            'ヒトTreg同定', 'CD4+CD25+CD127low', 'Tregマーカー'
        },
        "english": {
            'CD25 high expression', 'CD25high', 'CD25+', 'IL-2 receptor',
            'IL-2R alpha', 'CD127low', 'CD127 low', 'CD127-', 'IL-7Rα low',
            'IL-7R alpha low', 'GITR+', 'GITR positive',
            'activated T cell', 'Treg precursor', 'IL-2 responsiveness',
            'high-affinity IL-2R', 'upregulated CD25',
            'human Treg identification', 'CD4+CD25+CD127low', 'Treg marker combination'
        }
    },
    
    # Level 4: nTreg/iTreg（由来による分類）
    "treg_origin_level": {
        "ntreg": {
            "japanese": {
                '胸腺由来Treg', 'nTreg', '自然発生Treg', 'ナチュラルTreg',
                'Helios陽性', 'Nrp1陽性', '胸腺選択', '自己抗原認識',
                '中枢性免疫寛容', '胸腺髄質', 'AIRE', 
                '組織特異的抗原', 'TCR親和性', '高親和性TCR'
            },
            "english": {
                'thymic Treg', 'natural Treg', 'nTreg', 'tTreg',
                'Helios+', 'Nrp1+', 'thymic selection', 'self-antigen',
                'central tolerance', 'thymic medulla', 'AIRE',
                'tissue-specific antigen', 'high-affinity TCR', 'natural selection'
            }
        },
        "itreg": {
            "japanese": {
                '末梢誘導Treg', 'iTreg', '誘導性Treg', 'インデュースドTreg',
                'Helios陰性', '末梢転換', 'TGF-β誘導', '抗原認識',
                '末梢性免疫寛容', 'レチノイン酸', 'RA', 'TGF-β',
                '腸管免疫', '粘膜免疫', '食餌抗原', '腸内細菌'
            },
            "english": {
                'peripheral Treg', 'induced Treg', 'iTreg', 'pTreg',
                'Helios-', 'peripheral conversion', 'TGF-beta induced',
                'antigen-driven', 'peripheral tolerance', 'retinoic acid',
                'gut immunity', 'mucosal tolerance', 'dietary antigen',
                'microbiota', 'conversion from effector T'
            }
        }
    },
    
    # Level 5: Foxp3+ Treg（Foxp3発現 - 安定性による分類）
    "foxp3_treg_level": {
        "stable_treg": {
            "japanese": {
                'Foxp3陽性', 'Foxp3発現', 'Foxp3+Treg', '安定Treg',
                'FOXP3転写因子', 'Treg分化', 'Treg同一性',
                'エピジェネティック安定性', 'DNA脱メチル化', 'TSDR',
                'Treg特異的脱メチル化領域', 'TSDR脱メチル化', 'CNS1', 'CNS2', 'CNS3',
                'CNS2脱メチル化', 'Foxp3安定性', 'Treg系譜', '系譜決定',
                'CD45RA陽性', 'ナイーブTreg', '静止型Treg', 'rTreg',
                '真のTreg', '安定発現', '恒常的発現'
            },
            "english": {
                'Foxp3 positive', 'Foxp3+', 'Foxp3 expression', 'stable Treg',
                'FOXP3 transcription factor', 'Treg identity', 'lineage commitment',
                'epigenetic stability', 'DNA demethylation', 'TSDR',
                'Treg-specific demethylated region', 'TSDR demethylation',
                'CNS1', 'CNS2', 'CNS3', 'CNS2 demethylation',
                'Foxp3 stability', 'lineage determination', 'master regulator',
                'CD45RA+', 'naive Treg', 'resting Treg', 'rTreg',
                'bona fide Treg', 'stable expression', 'constitutive Foxp3'
            }
        },
        "transient_foxp3": {
            "japanese": {
                '一過性Foxp3発現', '一時的Foxp3', '活性化誘導Foxp3',
                'TSDRメチル化', 'エピジェネティック不安定',
                'CD45RO陽性', 'エフェクターT細胞', '活性化T細胞',
                '非Treg', '偽Treg', 'Foxp3+非Treg', 
                '一過性発現', '不安定発現', 'TCR刺激誘導'
            },
            "english": {
                'transient Foxp3 expression', 'temporary Foxp3', 'activation-induced Foxp3',
                'TSDR methylated', 'epigenetically unstable',
                'CD45RO+', 'effector T cell', 'activated T cell',
                'non-Treg', 'pseudo-Treg', 'Foxp3+ non-Treg',
                'transient expression', 'unstable expression', 'TCR-induced',
                'activation-dependent', 'non-suppressive Foxp3+'
            }
        },
        "discrimination": {
            "japanese": {
                'CD45RA/CD45RO', 'TSDR解析', 'エピジェネティック解析',
                '真のTreg識別', 'Treg純度', '機能的Treg',
                'メチル化解析', 'バイサルファイト法', 'PCR法'
            },
            "english": {
                'CD45RA/CD45RO discrimination', 'TSDR analysis', 'epigenetic profiling',
                'bona fide Treg identification', 'Treg purity', 'functional Treg',
                'methylation analysis', 'bisulfite sequencing', 'Treg stability assay'
            }
        }
    },
    
    # Level 6: Functional Treg（機能的Treg - サイトカイン産生）
    "functional_treg_level": {
        "cytokines": {
            "japanese": {
                'IL-10産生', 'TGF-β産生', 'IL-35産生',
                'インターロイキン10', 'トランスフォーミング増殖因子β',
                '抑制性サイトカイン', 'サイトカイン分泌',
                '免疫抑制サイトカイン', '抗炎症サイトカイン'
            },
            "english": {
                'IL-10 production', 'TGF-beta production', 'IL-35 production',
                'interleukin-10', 'transforming growth factor beta',
                'suppressive cytokine', 'anti-inflammatory cytokine',
                'cytokine secretion', 'immunosuppressive cytokine'
            }
        },
        "mechanisms": {
            "japanese": {
                'CTLA-4', 'LAG-3', 'PD-1', 'PD-L1', 'TIGIT',
                '接触依存性抑制', '細胞間接触', '共刺激阻害',
                'CD80/CD86競合', 'IDO誘導', 'トリプトファン代謝',
                'cAMP移行', 'グランザイムB', 'パーフォリン'
            },
            "english": {
                'CTLA-4', 'LAG-3', 'PD-1', 'PD-L1', 'TIGIT',
                'contact-dependent suppression', 'cell-cell contact',
                'costimulation blockade', 'CD80/CD86 competition',
                'IDO induction', 'tryptophan metabolism', 'cAMP transfer',
                'granzyme B', 'perforin', 'cytolysis'
            }
        },
        "functions": {
            "japanese": {
                '免疫抑制機能', '抑制機能', '免疫寛容',
                'エフェクターT細胞抑制', '炎症抑制', '組織修復',
                '自己免疫抑制', '同種免疫抑制', '腫瘍免疫',
                '移植免疫寛容', 'アレルギー抑制'
            },
            "english": {
                'immunosuppression', 'suppressive function', 'immune tolerance',
                'effector T cell suppression', 'inflammation control',
                'tissue repair', 'autoimmunity prevention', 'allograft tolerance',
                'tumor immunity', 'transplant tolerance', 'allergy suppression'
            }
        }
    }
}

# 階層判定関数の拡張（優先度調整版 - 改善版）
def determine_treg_level(content):
    """
    コンテンツから詳細なTreg分化階層レベルを判定
    
    改善された判定ロジック:
    1. Treg特異的マーカーを優先（Level 3-6）
    2. 前駆細胞マーカーは排他的に判定（Level 0-2）
    3. 文脈依存の重み付け
    
    Returns:
        int: 0-6のレベル番号
    """
    content_lower = content.lower()
    
    # === Treg特異的レベル（Level 3-6）を優先判定 ===
    
    # Level 6: Functional Treg (サイトカイン産生・抑制機能) - 最も特異的
    functional_cytokines = ['il-10', 'il-35', 'tgf-beta secret', 'tgf-β secret', 
                            'il-10 produc', 'il-10+', 'il-10 secret']
    
    functional_mechanisms = ['suppressive function', 'immunosuppression', 'immune suppression',
                            'contact-dependent suppress', 'effector suppress', 
                            'ctla-4 mediat', 'lag-3 express', 'immune tolerance mechanism']
    
    functional_keywords = ['suppressor cell', 'suppressive capacity', 'suppressive activity',
                          'regulatory function', 'tolerogenic', 'anti-inflammatory']
    
    # 機能的マーカーのスコアリング
    functional_score = 0
    functional_score += sum(1 for m in functional_cytokines if m in content_lower)
    functional_score += sum(1 for m in functional_mechanisms if m in content_lower)
    functional_score += sum(0.5 for m in functional_keywords if m in content_lower)
    
    # スコア2以上、またはサイトカイン+メカニズムの組み合わせ
    has_cytokine = any(m in content_lower for m in functional_cytokines)
    has_mechanism = any(m in content_lower for m in functional_mechanisms)
    
    if functional_score >= 2 or (has_cytokine and has_mechanism):
        return 6
    
    # Level 5: Foxp3+ Treg - Foxp3に特化
    foxp3_specific = ['foxp3 express', 'foxp3+', 'foxp3 positive', 'scurfin', 
                      'foxp3 transcript', 'ipex', 'foxp3 gene', 'foxp3 protein']
    
    # Foxp3があり、かつ機能的マーカーが少ない場合
    if any(marker in content_lower for marker in foxp3_specific):
        # 安定性/一過性マーカー
        stability_markers = ['tsdr demethyl', 'cd45ra', 'cns2 demethyl', 'epigenetic']
        transient_markers = ['transient foxp3', 'temporary', 'activation-induced foxp3']
        
        if any(m in content_lower for m in stability_markers + transient_markers):
            return 5
        # 機能的マーカーがなければLevel 5
        elif functional_score < 2:
            return 5
    
    # Level 4: nTreg/iTreg - 由来特異的マーカー
    # nTreg: 胸腺由来の明確な証拠
    ntreg_specific = ['thymic treg', 'natural treg', 'ntreg', 'ttreg', 
                      'helios+ treg', 'nrp1+ treg', 'aire medulla']
    
    # iTreg: 末梢誘導の明確な証拠（TGF-β誘導文脈）
    itreg_specific = ['induced treg', 'itreg', 'ptreg', 'peripheral treg conversion',
                      'tgf-beta induc', 'tgf-β induc', 'retinoic acid treg',
                      'gut-associated treg', 'oral tolerance treg']
    
    # nTreg特異的マーカーがあればLevel 4
    if any(m in content_lower for m in ntreg_specific):
        return 4
    
    # iTreg特異的マーカーがあればLevel 7（新設）
    if any(m in content_lower for m in itreg_specific):
        return 7
    
    # Level 3: CD25high + CD127low - Treg表面マーカー特異的
    cd25_cd127_specific = ['cd4+cd25+cd127low', 'cd4+ cd25+ cd127low', 'cd25high cd127low',
                           'cd25+ cd127-', 'cd127low treg', 'cd127- treg',
                           'il-7rαlow treg', 'il-7r alpha low regulatory']
    
    if any(m in content_lower for m in cd25_cd127_specific):
        # Foxp3やnTreg/iTregの明確な言及がなければLevel 3
        advanced_markers = ['foxp3', 'thymic treg', 'induced treg', 'ntreg', 'itreg']
        if not any(adv in content_lower for adv in advanced_markers):
            return 3
    
    # === 前駆細胞レベル（Level 0-2）- Tregマーカーがない場合のみ ===
    
    # Tregキーワードの存在チェック
    treg_keywords = ['regulatory t', 'treg', 'foxp3', 'cd25+', 'suppressive', 'tolerance']
    has_treg_context = any(kw in content_lower for kw in treg_keywords)
    
    # Tregコンテキストがない場合のみ前駆細胞判定
    if not has_treg_context:
        # Level 0: HSC - 極めて特異的なHSCマーカー
        hsc_very_specific = ['lin-sca-1+c-kit+', 'lin- sca-1+ c-kit+', 'lsk cell',
                            'cd34-cd150+', 'cd34- cd150+', 'slam marker hsc',
                            'long-term hsc', 'lt-hsc', 'quiescent hsc']
        
        if any(m in content_lower for m in hsc_very_specific):
            return 0
        
        # HSC一般的マーカー（他のキーワードとの組み合わせ）
        hsc_general = ['hematopoietic stem', 'hsc', 'bone marrow niche']
        hsc_context = ['self-renewal', 'multipotent', 'quiescence', 'stem cell niche']
        
        if any(hsc in content_lower for hsc in hsc_general):
            if any(ctx in content_lower for ctx in hsc_context):
                # リンパ球やT細胞の言及がなければHSC
                if not any(lymph in content_lower for lymph in ['lymphoid', 't cell', 'b cell']):
                    return 0
        
        # Level 1: CLP - リンパ球前駆細胞特異的
        clp_specific = ['common lymphoid progenitor', 'clp', 'lymphoid progenitor',
                       'il-7r+ progenitor', 'flt3+ il-7r+', 'lymphoid lineage commitment']
        
        if any(m in content_lower for m in clp_specific):
            # T細胞の具体的な言及がなければCLP
            if not any(t in content_lower for t in ['cd4+', 'cd8+', 'tcr', 'thymocyte']):
                return 1
        
        # Level 2: CD4+ T - ヘルパーT細胞（Tregではない）
        cd4_specific = ['cd4+ t cell', 'cd4 positive t cell', 'helper t cell',
                       'th1 cell', 'th2 cell', 'th17 cell', 
                       'naive cd4', 'effector cd4']
        
        if any(m in content_lower for m in cd4_specific):
            # CD25やTregマーカーがなければCD4+ T
            return 2
    
    # デフォルト: どれにも該当しない場合
    # Tregコンテキストがあっても特定できない場合はLevel 0（より慎重な分類）
    # 明確な特徴がない場合は未分類として扱う
    # if has_treg_context:
    #     return 4  # 旧ロジック: 不明確なものをL4に入れていた
    
    # Treg文脈があるが特定のレベルに分類できない場合
    # → より一般的なnTreg/iTreg マーカーで再判定
    if has_treg_context:
        # 一般的なnTreg関連語（胸腺、自然）
        general_ntreg = ['thymus', 'thymic', 'natural regulatory', 'central tolerance']
        # 一般的なiTreg関連語（誘導、末梢、腸管）
        general_itreg = ['peripheral', 'induced', 'conversion', 'gut', 'intestin', 
                         'mucosal', 'tgf', 'oral tolerance']
        
        # 一般的なiTreg文脈があればLevel 7
        if any(m in content_lower for m in general_itreg):
            return 7
        
        # 一般的なnTreg文脈があればLevel 4  
        if any(m in content_lower for m in general_ntreg):
            return 4
        
        # それでも不明だが明確にTreg関連なら、デフォルトでnTreg寄りと判断
        # （胸腺由来が基本形のため）
        if 'regulatory t' in content_lower or 'cd25+' in content_lower:
            return 4
    
    # 完全に不明な場合はLevel 0
    return 0


# 階層特異的ラベル生成
def generate_enhanced_treg_label(content, level, cluster_id, cluster_size):
    """
    拡張Treg階層に基づくラベル生成
    
    Args:
        content: テキストコンテンツ
        level: 階層レベル (0-7)
        cluster_id: クラスターID
        cluster_size: クラスターサイズ
        
    Returns:
        str: 階層特異的ラベル
    """
    if level not in ENHANCED_LEVEL_COLOR_MAPPING:
        level = 0
    
    level_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
    base_name = level_info["name"]
    
    # レベル特異的キーワード抽出
    content_lower = content.lower()
    detected_markers = []
    
    if level == 6:  # Functional Treg
        cytokines = []
        if 'il-10' in content_lower:
            cytokines.append('IL-10')
        if 'tgf-beta' in content_lower or 'tgf-β' in content_lower:
            cytokines.append('TGF-β')
        if 'ctla-4' in content_lower:
            cytokines.append('CTLA-4')
        if cytokines:
            return f"{base_name}\n{'+'.join(cytokines)}\n(n={cluster_size})"
    
    elif level == 5:  # Foxp3+ Treg
        stability_type = ""
        if 'tsdr demethyl' in content_lower or 'cd45ra' in content_lower:
            stability_type = "stable"
        elif 'transient' in content_lower or 'cd45ro' in content_lower:
            stability_type = "transient"
        
        if 'foxp3' in content_lower:
            if stability_type == "stable":
                return f"{base_name}\nFoxp3+ stable\nTSDR demethyl\n(n={cluster_size})"
            elif stability_type == "transient":
                return f"{base_name}\nFoxp3+ transient\nCD45RO+\n(n={cluster_size})"
            else:
                return f"{base_name}\nFoxp3+\n(n={cluster_size})"
    
    elif level == 4:  # nTreg/iTreg
        if 'thymic' in content_lower or 'helios+' in content_lower:
            return f"{base_name}\nnTreg-thymic\n(n={cluster_size})"
        elif 'peripheral' in content_lower or 'induced' in content_lower:
            return f"{base_name}\niTreg-peripheral\n(n={cluster_size})"
    
    
    elif level == 7:  # iTreg (末梢誘導Treg) - 新設
        if 'peripheral' in content_lower or 'induced' in content_lower or 'itreg' in content_lower:
            if 'tgf-beta' in content_lower or 'tgf-β' in content_lower:
                return f"iTreg (Induced)\nperipheral-TGF-β\n(n={cluster_size})"
            elif 'gut' in content_lower or 'oral tolerance' in content_lower:
                return f"iTreg (Induced)\ngut-associated\n(n={cluster_size})"
            else:
                return f"iTreg (Induced)\nperipheral\n(n={cluster_size})"
        else:
            return f"iTreg (Induced)\n(n={cluster_size})"

    elif level == 3:  # CD25high + CD127low
        cd127_status = "CD127low" if 'cd127' in content_lower or 'il-7r' in content_lower else ""
        if cd127_status:
            return f"{base_name}\nCD25high CD127low\nIL-2Rα+/IL-7Rα−\n(n={cluster_size})"
        else:
            return f"{base_name}\nCD25high\nIL-2Rα high\n(n={cluster_size})"
    
    # デフォルトラベル
    return f"{base_name}\nCluster {cluster_id}\n(n={cluster_size})"

# 後方互換性のための関数マッピング
LEVEL_COLOR_MAPPING = ENHANCED_LEVEL_COLOR_MAPPING
generate_immune_label = generate_enhanced_treg_label

def validate_immune_terminology(label):
    """免疫学用語の妥当性検証"""
    if not label or len(label) < 2:
        return False, "Label too short"
    
    # ASCIIチェック
    try:
        label.encode('ascii')
        return True, "Valid ASCII label"
    except UnicodeEncodeError:
        return False, "Contains non-ASCII characters"

def extract_level_keywords(content, level):
    """レベル特異的キーワード抽出"""
    content_lower = content.lower()
    keywords = set()
    
    if level in ENHANCED_LEVEL_COLOR_MAPPING:
        level_info = ENHANCED_LEVEL_COLOR_MAPPING[level]
        if 'markers' in level_info and isinstance(level_info['markers'], list):
            keywords.update([m.lower() for m in level_info['markers']])
        if 'keywords' in level_info:
            keywords.update([k.lower() for k in level_info['keywords']])
    
    # コンテンツ内のキーワードマッチング
    found_keywords = [kw for kw in keywords if kw in content_lower]
    return found_keywords[:5]  # 上位5個を返す
