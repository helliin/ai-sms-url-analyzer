import pandas as pd


# ============================================================
# LLM BELİRSİZLİK ARALIĞI
# ============================================================

LLM_LOW = 30
LLM_HIGH = 50


def should_use_llm(combined_score):
    """
    Combined score değerine göre
    LLM'nin devreye girip girmeyeceğini belirler.
    """

    return (
        LLM_LOW <= combined_score < LLM_HIGH
    )


# ============================================================
# TEST VERİSİ
# ============================================================

test_scores = pd.Series([
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
])


print("Combined skorlar:")
print(test_scores.tolist())


# ============================================================
# LLM KARARLARI
# ============================================================

llm_decisions = test_scores.apply(
    should_use_llm
)


print("\nLLM kullanılacak mı?")

print(llm_decisions.tolist())


# ============================================================
# SONUÇLARI GÖSTER
# ============================================================

results = pd.DataFrame({
    "combined_score": test_scores,
    "use_llm": llm_decisions
})


print("\nSonuç:")

print(results)