
const analyzeButton = document.getElementById("analyzeButton");

const messageInput = document.getElementById("messageInput");
const urlInput = document.getElementById("urlInput");

const riskScore = document.getElementById("riskScore");
const riskStatus = document.getElementById("riskStatus");
const riskFactors = document.getElementById("riskFactors");


analyzeButton.addEventListener("click", async function () {

    const message = messageInput.value.trim();
    const url = urlInput.value.trim();


    // SMS boşsa analiz yapma
    if (!message) {
        alert("Lütfen bir SMS mesajı girin.");
        return;
    }


    // Butonu geçici olarak pasif yap
    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analiz ediliyor...";


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/analyze",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    message: message,
                    url: url
                })
            }
        );


        // Backend hata döndürürse
        if (!response.ok) {
            throw new Error(
                "Sunucu hatası: " + response.status
            );
        }


        // Backend'den gelen JSON
        const result = await response.json();

        console.log("Backend sonucu:", result);


        // =========================
        // RİSK SKORU
        // =========================

        riskScore.textContent =
            result.overall_risk_score + " / 100";


        // =========================
        // RİSK DURUMU
        // =========================

        const score = result.overall_risk_score;


        if (score >= 70) {

            riskStatus.textContent = "Yüksek Risk";

        } else if (score >= 40) {

            riskStatus.textContent = "Şüpheli";

        } else {

            riskStatus.textContent = "Düşük Risk";
        }


        // =========================
        // RİSK FAKTÖRLERİ
        // =========================

        riskFactors.innerHTML = "";


        // ML sonucu
        if (result.ml_analysis) {

            const mlItem = document.createElement("li");

            mlItem.textContent =
                "ML Tahmini: " +
                result.ml_analysis.prediction;

            riskFactors.appendChild(mlItem);
        }


        // Rule Analyzer sonuçları
        if (
            result.rule_analysis &&
            result.rule_analysis.detected_categories
        ) {

            result.rule_analysis.detected_categories.forEach(
                function (category) {

                    const ruleItem =
                        document.createElement("li");

                    ruleItem.textContent =
                        "Kural: " + category;

                    riskFactors.appendChild(ruleItem);
                }
            );
        }


        // =========================
        // URL ANALİZİ
        // =========================

        if (result.urls && result.urls.length > 0) {

            result.urls.forEach(function (urlResult) {

                // URL'nin kendisi
                const urlItem =
                    document.createElement("li");

                urlItem.textContent =
                    "URL: " + urlResult.url;

                riskFactors.appendChild(urlItem);


                // URL risk skoru
                const urlScoreItem =
                    document.createElement("li");

                urlScoreItem.textContent =
                    "URL Risk Skoru: " +
                    urlResult.risk_score +
                    " / 100";

                riskFactors.appendChild(urlScoreItem);


                // URL risk faktörleri
                if (
                    urlResult.risk_factors &&
                    urlResult.risk_factors.length > 0
                ) {

                    urlResult.risk_factors.forEach(
                        function (factor) {

                            const factorItem =
                                document.createElement("li");

                            factorItem.textContent =
                                "URL Riski: " + factor;

                            riskFactors.appendChild(factorItem);
                        }
                    );
                }

            });

        }


        // Hiç risk faktörü bulunmadıysa
        if (riskFactors.children.length === 0) {

            const item =
                document.createElement("li");

            item.textContent =
                "Belirgin bir risk faktörü tespit edilmedi.";

            riskFactors.appendChild(item);
        }


    } catch (error) {

        console.error("Analiz hatası:", error);

        alert("Hata: " + error.message);

    } finally {

        // Butonu tekrar aktif et
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analiz Et";
    }

});

