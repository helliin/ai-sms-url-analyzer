import pandas as pd


# ============================================================
# LLM KARAR STRATEJİSİ
# ============================================================

def should_use_llm(combined_score):
    """
    Combined skorun kararsız olduğu durumda
    LLM kullanılmasına karar verir.

    30-49 arası skorlar belirsiz kabul edilir.
    """

    return 30 <= combined_score < 50


# ============================================================
# TEST VERİLERİ
# ============================================================

test_scores = [
    10,
    22,
    29,
    30,
    34,
    39,
    40,
    44,
    49,
    50,
    65,
    82
]


# ============================================================
# LLM KARARLARINI HESAPLA
# ============================================================

results = []


for score in test_scores:

    use_llm = should_use_llm(score)

    results.append({
        "combined_score": score,
        "use_llm": use_llm
    })


# ============================================================
# SONUÇLARI GÖSTER
# ============================================================

df = pd.DataFrame(results)


print("\nLLM karar stratejisi:")
print(df.to_string(index=False))


print("\nLLM kullanılacak mesaj sayısı:")

print(
    df["use_llm"].sum()
)


print("\nLLM kullanılmayacak mesaj sayısı:")

print(
    (~df["use_llm"]).sum()
)
