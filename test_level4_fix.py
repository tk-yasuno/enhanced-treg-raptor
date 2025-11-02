"""修正されたenhanced_treg_vocab.pyの動作テスト"""
from enhanced_treg_vocab import determine_treg_level

test_cases = [
    ("thymic treg natural regulatory cells", 4, "nTreg特異的"),
    ("induced treg peripheral conversion TGF-beta", 7, "iTreg特異的"),
    ("regulatory t cell suppression", 0, "不明確なTregコンテキスト"),
    ("foxp3 positive regulatory", 5, "Foxp3+ Treg"),
    ("il-10 secreting suppressive function", 6, "Functional Treg"),
]

print("=" * 70)
print("🧪 Level 4分離テスト結果")
print("=" * 70)

all_pass = True
for text, expected, description in test_cases:
    result = determine_treg_level(text)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    
    print(f"\n{status} {description}")
    print(f"  テキスト: {text[:50]}...")
    print(f"  期待レベル: {expected}, 実際: {result}")

print("\n" + "=" * 70)
if all_pass:
    print("✅ すべてのテストが合格しました！")
else:
    print("⚠️ 一部のテストが失敗しました")
print("=" * 70)

# レベル分布の予測
print("\n📊 期待される改善:")
print("  - Level 4 (nTreg): 大幅減少（43% → 15-20%程度）")
print("  - Level 7 (iTreg): 新設（15-20%程度）")
print("  - Level 0: 増加（不明確なケースを含む）")
