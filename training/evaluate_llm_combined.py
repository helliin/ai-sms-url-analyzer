import sys
from pathlib import Path
import re
import math

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PROJE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR)
)


# ============================================================
# 2. ANALYZERLAR
# ============================================================

from backend.ai.rule_analyzer_en import analyze_rules_en
from backend.ai.url_analyzer import analyze_url
from backend.ai.llm_analyzer import analyze_with_llm


# ============================================================
# 3. DATASET
# ============================================================

df = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


print("İlk dataset boyutu:")
print(df.shape)


# Duplicate temizle

df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)


print("\nDuplicate temizlendikten sonra:")
print(df.shape)


X = df["message"]
y = df["label"]


# ============================================================
# 4. TRAIN / VALIDATION / TEST
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)


print("\nTrain boyutu:")
print(X_train.shape)


print("\nValidation boyutu:")
print(X_validation.shape)


print("\nTest boyutu:")
print(X_test.shape)


# ============================================================
# 5. ML MODELİ
# ============================================================

vectorizer = TfidfVectorizer()


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


model = LinearSVC()


model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 6. ML SCORE
# ============================================================

ml_decision_scores = model.decision_function(
    X_test_tfidf
)


def normalize_ml_score(score):
    """
    SVM decision score değerini
    0-100 arasına dönüştürür.
    """

    probability = 1 / (
        1 + math.exp(-score)
    )

    return probability * 100


ml_scores_normalized = pd.Series(
    ml_decision_scores,
    index=X_test.index
).apply(
    normalize_ml_score
)


# ============================================================
# 7. RULE ANALYZER
# ============================================================

rule_results = X_test.apply(
    analyze_rules_en
)


rule_scores = rule_results.apply(
    lambda result: result["risk_score"]
)


# ============================================================
# 8. URL ANALYZER
# ============================================================

URL_PATTERN = r"https?://\S+|www\.\S+"


def extract_urls(message):

    return re.findall(
        URL_PATTERN,
        message
    )


def analyze_message_urls(message):

    urls = extract_urls(
        message
    )


    if not urls:

        return {
            "url_count": 0,
            "url_score": 0
        }


    url_results = []


    for url in urls:

        url = url.rstrip(
            ".,!?;:)]}"
        )


        result = analyze_url(
            url
        )


        url_results.append(
            result
        )


    highest_risk = max(
        url_results,
        key=lambda result: result["risk_score"]
    )


    return {
        "url_count": len(urls),
        "url_score": highest_risk["risk_score"]
    }


url_results = X_test.apply(
    analyze_message_urls
)


url_scores = url_results.apply(
    lambda result: result["url_score"]
)


url_counts = url_results.apply(
    lambda result: result["url_count"]
)


# ============================================================
# 9. COMBINED MODEL PARAMETRELERİ
# ============================================================

ML_WEIGHT = 0.90
RULE_WEIGHT = 0.05
URL_WEIGHT = 0.05

THRESHOLD = 40


print("\nCombined model parametreleri:")

print(
    "ML weight:",
    ML_WEIGHT
)

print(
    "Rule weight:",
    RULE_WEIGHT
)

print(
    "URL weight:",
    URL_WEIGHT
)

print(
    "Threshold:",
    THRESHOLD
)


# ============================================================
# 10. COMBINED SCORE
# ============================================================

combined_scores = (

    ml_scores_normalized * ML_WEIGHT

    + rule_scores * RULE_WEIGHT

    + url_scores * URL_WEIGHT

)


# ============================================================
# 11. COMBINED MODEL KARARI
# ============================================================

combined_predictions = (

    combined_scores >= THRESHOLD

).map({

    True: "spam",

    False: "ham"

})


# ============================================================
# 12. GEMINI STRATEJİSİ
# ============================================================

LLM_LOWER = 20
LLM_UPPER = 40

# Ücretsiz API kotasını korumak için
# maksimum gerçek Gemini çağrısı

MAX_LLM_CALLS = 10


def should_use_llm(score):

    """
    Combined skor 20-40 arasındaysa
    mesaj Gemini tarafından incelenebilir.
    """

    return (

        LLM_LOWER
        <= score
        < LLM_UPPER

    )


print("\n========================================")
print("GERÇEK GEMINI LLM STRATEJİSİ")
print("========================================")

print(
    "LLM alt sınır:",
    LLM_LOWER
)

print(
    "LLM üst sınır:",
    LLM_UPPER
)

print(
    "Maksimum Gemini çağrısı:",
    MAX_LLM_CALLS
)


# ============================================================
# 13. GEMINI İLE FINAL KARAR
# ============================================================

final_predictions = []

llm_used_count = 0

llm_success_count = 0

llm_error_count = 0

llm_changed_count = 0

llm_details = []


for message, score, combined_prediction in zip(

    X_test,

    combined_scores,

    combined_predictions

):


    # --------------------------------------------------------
    # LLM kullanılacak mı?
    # --------------------------------------------------------

    if (

        should_use_llm(score)

        and llm_used_count < MAX_LLM_CALLS

    ):

        llm_used_count += 1


        print(
            f"\nGemini çağrısı "
            f"{llm_used_count}/{MAX_LLM_CALLS}"
        )


        try:

            llm_result = analyze_with_llm(
                message
            )


            llm_prediction = llm_result.get(
                "prediction",
                "unknown"
            )


            llm_success_count += 1


            # ------------------------------------------------
            # Gemini geçerli karar verdiyse
            # ------------------------------------------------

            if llm_prediction in [

                "spam",
                "ham"

            ]:

                final_prediction = (
                    llm_prediction
                )


                # Gemini Combined Model kararını
                # değiştirdi mi?

                if (

                    final_prediction
                    != combined_prediction

                ):

                    llm_changed_count += 1


                llm_details.append({

                    "combined_prediction":
                        combined_prediction,

                    "llm_prediction":
                        llm_prediction,

                    "changed":
                        final_prediction
                        != combined_prediction,

                    "risk_score":
                        llm_result.get(
                            "risk_score",
                            0
                        ),

                    "confidence":
                        llm_result.get(
                            "confidence",
                            0
                        )

                })


                final_predictions.append(
                    final_prediction
                )


            else:

                # Gemini geçersiz sonuç döndürürse
                # Combined Model korunur.

                final_predictions.append(
                    combined_prediction
                )


        except Exception as e:

            llm_error_count += 1


            print(
                "Gemini hatası:",
                e
            )


            # API hatası olduğunda
            # Combined Model kararını koru.

            final_predictions.append(
                combined_prediction
            )


    else:

        # ----------------------------------------------------
        # Gemini kullanılmıyorsa
        # Combined Model kullanılır.
        # ----------------------------------------------------

        final_predictions.append(
            combined_prediction
        )


# ============================================================
# 14. PANDAS SERIES
# ============================================================

final_predictions = pd.Series(
    final_predictions,
    index=X_test.index
)


# ============================================================
# 15. SONUÇ TABLOSU
# ============================================================

results = pd.DataFrame({

    "message":
        X_test,

    "true_label":
        y_test,

    "ml_score":
        ml_scores_normalized,

    "rule_score":
        rule_scores,

    "url_score":
        url_scores,

    "combined_score":
        combined_scores,

    "combined_prediction":
        combined_predictions,

    "final_prediction":
        final_predictions,

    "url_count":
        url_counts

})


# ============================================================
# 16. COMBINED MODEL PERFORMANSI
# ============================================================

combined_accuracy = accuracy_score(

    y_test,

    combined_predictions

)


combined_precision = precision_score(

    y_test,

    combined_predictions,

    pos_label="spam"

)


combined_recall = recall_score(

    y_test,

    combined_predictions,

    pos_label="spam"

)


combined_f1 = f1_score(

    y_test,

    combined_predictions,

    pos_label="spam"

)


# ============================================================
# 17. GEMINI SONRASI PERFORMANS
# ============================================================

final_accuracy = accuracy_score(

    y_test,

    final_predictions

)


final_precision = precision_score(

    y_test,

    final_predictions,

    pos_label="spam"

)


final_recall = recall_score(

    y_test,

    final_predictions,

    pos_label="spam"

)


final_f1 = f1_score(

    y_test,

    final_predictions,

    pos_label="spam"

)


# ============================================================
# 18. SADECE COMBINED MODEL
# ============================================================

print("\n========================================")
print("SADECE COMBINED MODEL")
print("========================================")


print(
    "Accuracy:",
    round(combined_accuracy, 4)
)


print(
    "Spam Precision:",
    round(combined_precision, 4)
)


print(
    "Spam Recall:",
    round(combined_recall, 4)
)


print(
    "Spam F1:",
    round(combined_f1, 4)
)


# ============================================================
# 19. COMBINED + GERÇEK GEMINI
# ============================================================

print("\n========================================")
print("COMBINED + GERÇEK GEMINI")
print("========================================")


print(
    "Accuracy:",
    round(final_accuracy, 4)
)


print(
    "Spam Precision:",
    round(final_precision, 4)
)


print(
    "Spam Recall:",
    round(final_recall, 4)
)


print(
    "Spam F1:",
    round(final_f1, 4)
)


# ============================================================
# 20. GEMINI KULLANIM BİLGİSİ
# ============================================================

print("\n========================================")
print("GEMINI KULLANIM BİLGİSİ")
print("========================================")


print(
    "Toplam test mesajı:",
    len(X_test)
)


print(
    "Gemini kullanılan mesaj:",
    llm_used_count
)


print(
    "Başarılı Gemini çağrısı:",
    llm_success_count
)


print(
    "Gemini hatası:",
    llm_error_count
)


print(
    "Gemini kullanım oranı:",
    round(
        llm_used_count
        / len(X_test)
        * 100,
        2
    ),
    "%"
)


print(
    "Gemini'nin kararı değiştirdiği mesaj:",
    llm_changed_count
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(

    y_test,

    final_predictions,

    labels=["ham", "spam"]

)


print("\n========================================")
print("COMBINED + GEMINI CONFUSION MATRIX")
print("========================================")


print(matrix)


print(
    "\nTrue Ham → Ham:",
    matrix[0][0]
)


print(
    "True Ham → Spam (False Positive):",
    matrix[0][1]
)


print(
    "True Spam → Ham (False Negative):",
    matrix[1][0]
)


print(
    "True Spam → Spam:",
    matrix[1][1]
)


# ============================================================
# 22. GEMINI KARAR DEĞİŞİKLİKLERİ
# ============================================================

changed_results = results[

    results["combined_prediction"]
    !=
    results["final_prediction"]

]


print("\n========================================")
print("GEMINI'NİN KARARI DEĞİŞTİRDİĞİ MESAJLAR")
print("========================================")


print(
    "Değişen mesaj sayısı:",
    len(changed_results)
)


if len(changed_results) > 0:

    print(

        changed_results[

            [

                "true_label",

                "combined_score",

                "combined_prediction",

                "final_prediction"

            ]

        ].head(20)

    )

else:

    print(
        "Gemini şu ana kadar "
        "hiçbir Combined Model kararını değiştirmedi."
    )


# ============================================================
# 23. BİTİŞ
# ============================================================

print("\n========================================")
print("TEST TAMAMLANDI")
print("========================================")

print(
    "Gemini maksimum çağrı limiti:",
    MAX_LLM_CALLS
)

print(
    "Gerçekleşen Gemini çağrısı:",
    llm_used_count
)

print(
    "Başarılı Gemini çağrısı:",
    llm_success_count
)

print(
    "Gemini hata sayısı:",
    llm_error_count
)