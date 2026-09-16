const landingPage = document.getElementById("landingPage");
const appPage = document.getElementById("appPage");

const gmailInput = document.getElementById("gmailInput");
const startWithGmailButton =
    document.getElementById("startWithGmailButton");
const loginStatus =
    document.getElementById("loginStatus");

const continueWithoutGmailButton =
    document.getElementById("continueWithoutGmailButton");

const messageInput =
    document.getElementById("messageInput");
const urlInput =
    document.getElementById("urlInput");
const analyzeButton =
    document.getElementById("analyzeButton");

const riskScore =
    document.getElementById("riskScore");
const riskStatus =
    document.getElementById("riskStatus");
const riskFactors =
    document.getElementById("riskFactors");

const llmResult =
    document.getElementById("llmResult");

const loadGmailButton =
    document.getElementById("loadGmailButton");
const gmailStatus =
    document.getElementById("gmailStatus");
const gmailList =
    document.getElementById("gmailList");
const gmailBody =
    document.getElementById("gmailBody");

const connectGmailButton =
    document.getElementById("connectGmailButton");

const gmailConnectedStatus =
    document.getElementById("gmailConnectedStatus");

let gmailLoaded = false;
let gmailVisible = false;
let connectedGmailAddress = "";


/* =========================
   SAYFA BAŞLANGICI
========================= */

document.addEventListener("DOMContentLoaded", () => {

    if (landingPage) {
        landingPage.style.display = "flex";
    }

    if (appPage) {
        appPage.style.display = "none";
    }

    if (riskScore) {
        riskScore.textContent = "0";
    }

    if (riskStatus) {
        riskStatus.textContent =
            "Henüz analiz yapılmadı";
    }

    /*
     * Daha önce Gmail bağlandıysa
     * bağlantı durumunu hatırla.
     */

    const savedGmailAddress =
        localStorage.getItem(
            "connectedGmailAddress"
        );

    if (savedGmailAddress) {

        connectedGmailAddress =
            savedGmailAddress;

        if (connectGmailButton) {
            connectGmailButton.style.display =
                "none";
        }

        if (gmailConnectedStatus) {

            gmailConnectedStatus.style.display =
                "block";

            gmailConnectedStatus.textContent =
                `✓ ${savedGmailAddress} Bağlı`;
        }
    }

});


/* =========================
   GMAIL İLE BAŞLA
========================= */

if (startWithGmailButton) {

    startWithGmailButton.addEventListener(
        "click",
        async () => {

            const gmailAddress =
                gmailInput.value.trim();

            if (!gmailAddress) {

                alert(
                    "Lütfen Gmail adresinizi girin."
                );

                gmailInput.focus();

                return;
            }

            if (
                !gmailAddress
                    .toLowerCase()
                    .endsWith("@gmail.com")
            ) {

                alert(
                    "Lütfen geçerli bir Gmail adresi girin."
                );

                gmailInput.focus();

                return;
            }

            connectedGmailAddress =
                gmailAddress;

            startWithGmailButton.disabled =
                true;

            startWithGmailButton.textContent =
                "Gmail'e bağlanıyor...";

            loginStatus.textContent =
                "Google hesabınız doğrulanıyor...";

            try {

                const response =
                    await fetch(
                        `http://127.0.0.1:8000/gmail/emails?email=${encodeURIComponent(connectedGmailAddress)}`
                    );

                const data =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Gmail bağlantısı kurulamadı."
                    );
                }

                /*
                 * Gmail bağlantısını
                 * tarayıcıda hatırla.
                 */

                localStorage.setItem(
                    "connectedGmailAddress",
                    connectedGmailAddress
                );

                /*
                 * Giriş ekranını kapat.
                 */

                if (landingPage) {
                    landingPage.style.display =
                        "none";
                }

                if (appPage) {
                    appPage.style.display =
                        "block";
                }

                /*
                 * Gmail bölümünü göster.
                 */

                const gmailSection =
                    document.querySelector(
                        ".gmail-section"
                    );

                if (gmailSection) {
                    gmailSection.style.display =
                        "block";
                }

                gmailLoaded =
                    true;

                gmailVisible =
                    true;

                if (gmailList) {
                    gmailList.style.display =
                        "flex";
                }

                if (gmailBody) {
                    gmailBody.style.display =
                        "none";
                }

                renderGmailEmails(
                    data.emails || []
                );

                if (loadGmailButton) {

                    loadGmailButton.textContent =
                        "Mailleri Kapat";
                }

                if (gmailStatus) {

                    gmailStatus.textContent =
                        `${connectedGmailAddress} hesabından mailler getirildi.`;
                }

                /*
                 * Üstte Gmail bağlantı durumunu göster.
                 */

                if (connectGmailButton) {

                    connectGmailButton.style.display =
                        "none";
                }

                if (gmailConnectedStatus) {

                    gmailConnectedStatus.style.display =
                        "block";

                    gmailConnectedStatus.textContent =
                        `✓ ${connectedGmailAddress} Bağlı`;
                }

            } catch (error) {

                console.error(
                    "Gmail bağlantı hatası:",
                    error
                );

                alert(
                    "Gmail bağlantısı kurulamadı.\n\n" +
                    error.message
                );

                loginStatus.textContent =
                    "Gmail bağlantısı kurulamadı.";

            } finally {

                startWithGmailButton.disabled =
                    false;

                startWithGmailButton.textContent =
                    "Gmail ile Başla";
            }

        }
    );

}


/* =========================
   SONRADAN GMAIL BAĞLA
========================= */

if (connectGmailButton) {

    connectGmailButton.addEventListener(
        "click",
        async () => {

            const gmailAddress =
                prompt(
                    "Bağlamak istediğiniz Gmail adresini girin:"
                );

            if (!gmailAddress) {
                return;
            }

            const trimmedEmail =
                gmailAddress.trim();

            if (
                !trimmedEmail
                    .toLowerCase()
                    .endsWith("@gmail.com")
            ) {

                alert(
                    "Lütfen geçerli bir Gmail adresi girin."
                );

                return;
            }

            connectedGmailAddress =
                trimmedEmail;

            connectGmailButton.disabled =
                true;

            connectGmailButton.textContent =
                "Gmail'e bağlanıyor...";

            try {

                const response =
                    await fetch(
                        `http://127.0.0.1:8000/gmail/emails?email=${encodeURIComponent(connectedGmailAddress)}`
                    );

                const data =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Gmail bağlantısı kurulamadı."
                    );
                }

                /*
                 * Gmail bağlantısını
                 * tarayıcıda hatırla.
                 */

                localStorage.setItem(
                    "connectedGmailAddress",
                    connectedGmailAddress
                );

                /*
                 * ÖNEMLİ:
                 *
                 * Kullanıcı daha önce
                 * "Gmail Olmadan Devam Et"
                 * seçtiyse landing page'e dönme.
                 *
                 * Aynı uygulama ekranında kal.
                 */

                if (landingPage) {
                    landingPage.style.display =
                        "none";
                }

                if (appPage) {
                    appPage.style.display =
                        "block";
                }

                /*
                 * Gmail bölümünü göster.
                 */

                const gmailSection =
                    document.querySelector(
                        ".gmail-section"
                    );

                if (gmailSection) {
                    gmailSection.style.display =
                        "block";
                }

                gmailLoaded =
                    true;

                gmailVisible =
                    true;

                if (gmailList) {
                    gmailList.style.display =
                        "flex";
                }

                if (gmailBody) {
                    gmailBody.style.display =
                        "none";
                }

                renderGmailEmails(
                    data.emails || []
                );

                if (loadGmailButton) {

                    loadGmailButton.textContent =
                        "Mailleri Kapat";
                }

                if (gmailStatus) {

                    gmailStatus.textContent =
                        `${connectedGmailAddress} hesabından mailler getirildi.`;
                }

                /*
                 * Gmail bağla butonunu gizle.
                 */

                connectGmailButton.style.display =
                    "none";

                if (gmailConnectedStatus) {

                    gmailConnectedStatus.style.display =
                        "block";

                    gmailConnectedStatus.textContent =
                        `✓ ${connectedGmailAddress} Bağlı`;
                }

            } catch (error) {

                console.error(
                    "Gmail bağlantı hatası:",
                    error
                );

                alert(
                    "Gmail bağlantısı kurulamadı.\n\n" +
                    error.message
                );

                connectedGmailAddress =
                    "";

            } finally {

                connectGmailButton.disabled =
                    false;

                connectGmailButton.textContent =
                    "📧 Gmail'i Bağla";
            }

        }
    );

}


/* =========================
   GMAIL OLMADAN DEVAM ET
========================= */

if (continueWithoutGmailButton) {

    continueWithoutGmailButton.addEventListener(
        "click",
        () => {

            /*
             * Gmail bağlantısını temizle.
             */

            connectedGmailAddress =
                "";

            localStorage.removeItem(
                "connectedGmailAddress"
            );

            if (gmailConnectedStatus) {

                gmailConnectedStatus.style.display =
                    "none";
            }

            if (connectGmailButton) {

                connectGmailButton.style.display =
                    "block";
            }

            if (landingPage) {
                landingPage.style.display =
                    "none";
            }

            if (appPage) {
                appPage.style.display =
                    "block";
            }

            const gmailSection =
                document.querySelector(
                    ".gmail-section"
                );

            /*
             * Gmail olmadan girildiğinde
             * Gmail bölümü gizli.
             */

            if (gmailSection) {

                gmailSection.style.display =
                    "none";
            }

            hideGmailBody();

        }
    );

}


/* =========================
   NORMAL MESAJ / URL ANALİZİ
========================= */

if (analyzeButton) {

    analyzeButton.addEventListener(
        "click",
        async () => {

            const message =
                messageInput.value.trim();

            const url =
                urlInput.value.trim();

            hideGmailBody();

            if (!message && !url) {

                alert(
                    "Lütfen analiz edilecek mesaj veya URL girin."
                );

                return;
            }

            analyzeButton.disabled =
                true;

            analyzeButton.textContent =
                "Analiz ediliyor...";

            try {

                const response =
                    await fetch(
                        "http://127.0.0.1:8000/analyze",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify({
                                message: message,
                                url: url
                            })
                        }
                    );

                const data =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Analiz sırasında hata oluştu."
                    );
                }

                displayAnalysis(data);

                /*
                 * SMS / URL analizini geçmişe kaydet.
                 */

                saveAnalysisToHistory(
                    data,
                    message && url
                        ? "SMS + URL"
                        : message
                            ? "SMS"
                            : "URL",
                    message && url
                        ? `${message}\nURL: ${url}`
                        : message || url
                );

            } catch (error) {

                console.error(
                    "Analiz hatası:",
                    error
                );

                alert(
                    "Analiz sırasında bir hata oluştu.\n\n" +
                    error.message
                );

            } finally {

                analyzeButton.disabled =
                    false;

                analyzeButton.textContent =
                    "Analizi Başlat";
            }

        }
    );

}


/* =========================
   GMAIL MAİLLERİNİ AÇ / KAPAT
========================= */

if (loadGmailButton) {

    loadGmailButton.addEventListener(
        "click",
        async () => {

            /*
             * MAİLLER AÇIKSA KAPAT
             */

            if (gmailVisible) {

                if (gmailList) {
                    gmailList.style.display =
                        "none";
                }

                if (gmailBody) {
                    gmailBody.style.display =
                        "none";
                }

                gmailVisible =
                    false;

                loadGmailButton.textContent =
                    "Maillerimi Görüntüle";

                if (gmailStatus) {

                    gmailStatus.textContent =
                        "Gelen kutusu kapatıldı.";
                }

                return;
            }


            /*
             * MAİLLER DAHA ÖNCE YÜKLENDİYSE
             * TEKRAR GÖSTER
             */

            if (gmailLoaded) {

                if (gmailList) {
                    gmailList.style.display =
                        "flex";
                }

                if (gmailBody) {
                    gmailBody.style.display =
                        "none";
                }

                gmailVisible =
                    true;

                loadGmailButton.textContent =
                    "Mailleri Kapat";

                if (gmailStatus) {

                    gmailStatus.textContent =
                        "Gelen kutunuz görüntüleniyor.";
                }

                return;
            }


            /*
             * MAİLLERİ İLK KEZ GETİR
             */

            if (!connectedGmailAddress) {

                alert(
                    "Önce Gmail hesabınızı bağlamanız gerekiyor."
                );

                return;
            }

            loadGmailButton.disabled =
                true;

            loadGmailButton.textContent =
                "Mailler yükleniyor...";

            if (gmailStatus) {

                gmailStatus.textContent =
                    "Gmail hesabınızdan mailler getiriliyor...";
            }

            try {

                const response =
                    await fetch(
                        `http://127.0.0.1:8000/gmail/emails?email=${encodeURIComponent(connectedGmailAddress)}`
                    );

                const data =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Mailler alınamadı."
                    );
                }

                gmailLoaded =
                    true;

                gmailVisible =
                    true;

                if (gmailList) {
                    gmailList.style.display =
                        "flex";
                }

                if (gmailBody) {
                    gmailBody.style.display =
                        "none";
                }

                renderGmailEmails(
                    data.emails || []
                );

                loadGmailButton.textContent =
                    "Mailleri Kapat";

                if (gmailStatus) {

                    gmailStatus.textContent =
                        `${(data.emails || []).length} mail bulundu.`;
                }

            } catch (error) {

                console.error(
                    "Gmail mail alma hatası:",
                    error
                );

                if (gmailStatus) {

                    gmailStatus.textContent =
                        "Mailler alınırken hata oluştu.";
                }

                alert(
                    "Gmail mailleri alınamadı.\n\n" +
                    error.message
                );

                loadGmailButton.textContent =
                    "Maillerimi Görüntüle";

            } finally {

                loadGmailButton.disabled =
                    false;
            }

        }
    );

}


/* =========================
   GMAIL MAİLLERİNİ KART OLARAK GÖSTER
========================= */

function renderGmailEmails(emails) {

    if (!gmailList) {
        return;
    }

    gmailList.innerHTML =
        "";

    if (!emails || emails.length === 0) {

        gmailList.innerHTML = `
            <div class="empty-mail">
                Gelen kutusunda mail bulunamadı.
            </div>
        `;

        return;
    }

    emails.forEach((email) => {

        const card =
            document.createElement("div");

        card.className =
            "gmail-mail-card";

        card.dataset.messageId =
            email.id || "";

        card.innerHTML = `
            <div class="gmail-mail-info">

                <div class="gmail-mail-sender">
                    ${escapeHtml(
                        email.from ||
                        "Bilinmeyen gönderen"
                    )}
                </div>

                <div class="gmail-mail-subject">
                    ${escapeHtml(
                        email.subject ||
                        "Konu yok"
                    )}
                </div>

                <div class="gmail-mail-date">
                    ${escapeHtml(
                        email.date ||
                        ""
                    )}
                </div>

            </div>

            <button
                type="button"
                class="gmail-analyze-button"
            >
                Bu Maili Analiz Et
            </button>
        `;

        const analyzeMailButton =
            card.querySelector(
                ".gmail-analyze-button"
            );

        analyzeMailButton.addEventListener(
            "click",
            () => {

                analyzeGmailEmail(
                    email.id,
                    card
                );

            }
        );

        gmailList.appendChild(card);

    });

}


/* =========================
   GMAIL MAİL ANALİZİ
========================= */

async function analyzeGmailEmail(
    messageId,
    card
) {

    const button =
        card.querySelector(
            ".gmail-analyze-button"
        );

    button.disabled =
        true;

    button.textContent =
        "Analiz ediliyor...";

    if (gmailStatus) {

        gmailStatus.textContent =
            "Mail analiz ediliyor...";
    }

    try {

        const response =
            await fetch(
                `http://127.0.0.1:8000/gmail/analyze/${encodeURIComponent(messageId)}`,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Mail analiz edilemedi."
            );
        }


        /*
         * MAIL İÇERİĞİNİ GÖSTER
         */

        showGmailBody(
            data.email
        );


        /*
         * ANALİZ SONUCUNU GÖSTER
         */

        displayAnalysis(
            data.analysis
        );


        /*
         * GMAIL ANALİZİNİ GEÇMİŞE KAYDET
         *
         * Burada da temizlenmiş içerik
         * kullanılıyor.
         */

        saveAnalysisToHistory(
            data.analysis,
            "Gmail",
            `${data.email?.subject || "Konu yok"}\n${cleanEmailBody(data.email?.body || "")}`
        );


        /*
         * ANALİZ EDİLEN KARTI KALDIR
         */

        card.classList.add(
            "removing"
        );

        setTimeout(() => {

            card.remove();

        }, 250);


        if (gmailStatus) {

            gmailStatus.textContent =
                "Mail analiz edildi.";
        }

    } catch (error) {

        console.error(
            "Gmail analiz hatası:",
            error
        );

        alert(
            "Mail analiz edilirken bir hata oluştu.\n\n" +
            error.message
        );

        button.disabled =
            false;

        button.textContent =
            "Bu Maili Analiz Et";

        if (gmailStatus) {

            gmailStatus.textContent =
                "Mail analiz edilemedi.";
        }
    }

}


/* =========================
   GÖRÜNEN MAİL İÇERİĞİNİ GİZLE
========================= */

function hideGmailBody() {

    if (gmailBody) {

        gmailBody.innerHTML =
            "";

        gmailBody.style.display =
            "none";
    }

}


/* =========================
   MAİL İÇERİĞİNİ GÖSTER
========================= */

function showGmailBody(email) {

    if (!gmailBody) {
        return;
    }

    gmailBody.innerHTML =
        "";

    gmailBody.style.display =
        "block";

    const container =
        document.createElement("div");

    container.className =
        "gmail-body-box";

    const header =
        document.createElement("div");

    header.className =
        "mail-content-header";

    header.innerHTML = `
        <strong>Mail İçeriği</strong>
        <span>
            ${escapeHtml(
                email?.subject ||
                ""
            )}
        </span>
    `;

    const content =
        document.createElement("div");

    content.className =
        "mail-content";

    const cleanBody =
        cleanEmailBody(
            email?.body || ""
        );

    content.textContent =
        cleanBody ||
        "Mail içeriği bulunamadı.";

    content.style.whiteSpace =
        "pre-wrap";

    content.style.overflowWrap =
        "anywhere";

    content.style.wordBreak =
        "break-word";

    container.appendChild(
        header
    );

    container.appendChild(
        content
    );

    gmailBody.appendChild(
        container
    );

}


/* =========================
   MAİL METNİNİ TEMİZLE
========================= */

function cleanEmailBody(body) {

    let text =
        String(body || "")
            .replace(/\r\n/g, "\n")
            .replace(/\r/g, "\n")
            .trim();

    if (!text) {
        return "";
    }


    /*
     * GMAIL'DEN GELEN REFERENCES BÖLÜMÜNÜ
     * KULLANICIYA GÖSTERME.
     *
     * ÖNEMLİ:
     * Bu sadece frontend görüntüsünü temizler.
     * Backend'in analiz ettiği ham mail değişmez.
     */

    const referencesIndex =
        text.search(
            /\n?\s*References\s*:/i
        );

    if (referencesIndex >= 0) {

        text =
            text.substring(
                0,
                referencesIndex
            ).trim();
    }


    /*
     * Mail içindeki gereksiz tekrarları temizle.
     */

    const duplicateMarkers = [
        "Profile Picture of",
        "View Messages",
        "Or send a message to"
    ];

    for (
        const marker of duplicateMarkers
    ) {

        const index =
            text.indexOf(marker);

        if (index > 0) {

            text =
                text.substring(
                    0,
                    index
                ).trim();

            break;
        }
    }


    /*
     * Gereksiz footer kısmını temizle.
     */

    const footerMarkers = [
        "© 2026 Freelancer Technology",
        "© 2025 Freelancer Technology",
        "© 2024 Freelancer Technology"
    ];

    for (
        const marker of footerMarkers
    ) {

        const index =
            text.indexOf(marker);

        if (index > 0) {

            text =
                text.substring(
                    0,
                    index
                ).trim();

            break;
        }
    }


    /*
     * Çok fazla boş satırı azalt.
     */

    text =
        text.replace(
            /\n{3,}/g,
            "\n\n"
        );

    return text.trim();

}


/* =========================
   ANALİZ SONUCUNA KAYDIR
========================= */

function scrollToAnalysisResult() {

    setTimeout(() => {

        if (riskScore) {

            riskScore.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        }

    }, 150);

}


/* =========================
   ANALİZ SONUCUNU GÖSTER
========================= */

function displayAnalysis(data) {

    if (!data) {
        return;
    }

    const score =
        Math.round(
            Number(
                data.overall_risk_score || 0
            )
        );

    if (riskScore) {

        riskScore.textContent =
            `${score}/100`;
    }

    if (riskStatus) {

        if (score >= 70) {

            riskStatus.textContent =
                "🔴 YÜKSEK RİSK";

        } else if (score >= 30) {

            riskStatus.textContent =
                "🟡 ŞÜPHELİ";

        } else {

            riskStatus.textContent =
                "🟢 DÜŞÜK RİSK";
        }
    }

    if (riskFactors) {

        riskFactors.innerHTML =
            "";

        const factors = [];

        const ruleAnalysis =
            data.rule_analysis || {};

        const mlAnalysis =
            data.ml_analysis || {};

        const urls =
            data.urls || [];

        if (
            ruleAnalysis.risk_score !==
            undefined
        ) {

            factors.push(
                `Kural analizi skoru: ${ruleAnalysis.risk_score}`
            );

        } else if (
            ruleAnalysis.score !==
            undefined
        ) {

            factors.push(
                `Kural analizi skoru: ${ruleAnalysis.score}`
            );
        }

        if (
            mlAnalysis.spam_probability !==
            undefined
        ) {

            const spamProbability =
                Number(
                    mlAnalysis.spam_probability
                );

            factors.push(
                `DistilBERT spam olasılığı: ${(spamProbability * 100).toFixed(1)}%`
            );
        }

        if (urls.length > 0) {

            factors.push(
                `${urls.length} adet URL analiz edildi.`
            );
        }

        if (factors.length === 0) {

            factors.push(
                "Belirgin bir risk faktörü bulunamadı."
            );
        }

        factors.forEach((factor) => {

            const li =
                document.createElement("li");

            li.textContent =
                factor;

            riskFactors.appendChild(
                li
            );

        });

    }


    /* =========================
       LLM İKİNCİ GÖRÜŞ
    ========================= */

    const llm =
        data.llm_analysis;

    if (!llm) {

        if (llmResult) {

            llmResult.textContent =
                "LLM analizi kullanılmadı.";
        }

        scrollToAnalysisResult();

        return;
    }

    const llmPrediction =
        document.getElementById(
            "llmPrediction"
        );

    const llmRiskScore =
        document.getElementById(
            "llmRiskScore"
        );

    const llmConfidence =
        document.getElementById(
            "llmConfidence"
        );

    const llmReason =
        document.getElementById(
            "llmReason"
        );

    const llmRiskFactors =
        document.getElementById(
            "llmRiskFactors"
        );

    const prediction =
        String(
            llm.prediction ||
            llm.label ||
            "Bilinmiyor"
        ).toUpperCase();

    if (llmPrediction) {

        llmPrediction.textContent =
            prediction;

        if (prediction === "SPAM") {

            llmPrediction.className =
                "llm-spam";

        } else {

            llmPrediction.className =
                "llm-ham";
        }
    }

    const llmScore =
        Number(
            llm.risk_score ??
            llm.riskScore ??
            0
        );

    if (llmRiskScore) {

        llmRiskScore.textContent =
            `${Math.round(llmScore)}/100`;
    }

    const confidence =
        Number(
            llm.confidence ??
            0
        );

    if (llmConfidence) {

        const confidencePercent =
            confidence <= 1
                ? confidence * 100
                : confidence;

        llmConfidence.textContent =
            `${confidencePercent.toFixed(0)}%`;
    }

    const reason =
        llm.reason ||
        llm.explanation ||
        llm.analysis ||
        "LLM tarafından ek bir açıklama verilmedi.";

    if (llmReason) {

        llmReason.textContent =
            reason;
    }

    if (llmRiskFactors) {

        llmRiskFactors.innerHTML =
            "";

        let llmFactors =
            llm.risk_factors ||
            llm.riskFactors ||
            [];

        if (!Array.isArray(llmFactors)) {

            llmFactors = [
                String(llmFactors)
            ];
        }

        if (llmFactors.length === 0) {

            const li =
                document.createElement("li");

            li.textContent =
                "Belirgin bir LLM risk faktörü belirtilmedi.";

            llmRiskFactors.appendChild(
                li
            );

        } else {

            llmFactors.forEach((factor) => {

                const li =
                    document.createElement("li");

                li.textContent =
                    String(factor);

                llmRiskFactors.appendChild(
                    li
                );

            });
        }
    }

    if (llmResult) {

        llmResult.textContent =
            "";
    }

    scrollToAnalysisResult();

}


/* =========================
   HTML GÜVENLİĞİ
========================= */

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value ?? "");

    return div.innerHTML;
}


/* =========================
   ANALİZ GEÇMİŞİ / HAFIZA
========================= */

const HISTORY_STORAGE_KEY =
    "ai_analyzer_history";

function getAnalysisHistory() {

    try {

        const history =
            localStorage.getItem(
                HISTORY_STORAGE_KEY
            );

        return history
            ? JSON.parse(history)
            : [];

    } catch (error) {

        console.error(
            "Analiz geçmişi okunamadı:",
            error
        );

        return [];
    }
}


function saveAnalysisToHistory(
    data,
    type,
    originalContent
) {

    try {

        const history =
            getAnalysisHistory();

        const analysis =
            data || {};

        const mlAnalysis =
            analysis.ml_analysis || {};

        const ruleAnalysis =
            analysis.rule_analysis || {};

        const score =
            Number(
                analysis.overall_risk_score || 0
            );

        let result =
            "Düşük Risk";

        if (score >= 70) {

            result =
                "Yüksek Risk";

        } else if (score >= 40) {

            result =
                "Orta Risk";
        }

        const historyItem = {

            id:
                Date.now(),

            type:
                type || "SMS",

            content:
                String(
                    originalContent || ""
                ).substring(0, 1000),

            riskScore:
                Math.round(score),

            result:
                result,

            spamProbability:
                mlAnalysis.spam_probability !==
                undefined
                    ? Number(
                        mlAnalysis.spam_probability
                    )
                    : null,

            ruleScore:
                ruleAnalysis.risk_score !==
                undefined
                    ? Number(
                        ruleAnalysis.risk_score
                    )
                    : (
                        ruleAnalysis.score !==
                        undefined
                            ? Number(
                                ruleAnalysis.score
                            )
                            : null
                    ),

            urlCount:
                Array.isArray(
                    analysis.urls
                )
                    ? analysis.urls.length
                    : 0,

            date:
                new Date().toLocaleString(
                    "tr-TR"
                )
        };

        history.unshift(
            historyItem
        );

        /*
         * Son 50 analizi sakla.
         */

        const limitedHistory =
            history.slice(
                0,
                50
            );

        localStorage.setItem(
            HISTORY_STORAGE_KEY,
            JSON.stringify(
                limitedHistory
            )
        );

        renderAnalysisHistory();

    } catch (error) {

        console.error(
            "Analiz geçmişi kaydedilemedi:",
            error
        );
    }
}


/* =========================
   GEÇMİŞ PANELİNİ OLUŞTUR
========================= */

function createHistorySection() {

    if (
        document.getElementById(
            "analysisHistorySection"
        )
    ) {
        return;
    }

    const section =
        document.createElement("section");

    section.id =
        "analysisHistorySection";

    section.className =
        "analysis-history-section";

    section.innerHTML = `

        <div class="history-header">

            <div>

                <h2>
                    📚 Analiz Geçmişi
                </h2>

                <p>
                    Daha önce gerçekleştirdiğiniz
                    analizleri burada görebilirsiniz.
                </p>

            </div>

            <button
                type="button"
                id="clearHistoryButton"
                class="clear-history-button"
            >
                Geçmişi Temizle
            </button>

        </div>

        <div
            id="analysisHistoryList"
            class="analysis-history-list"
        ></div>

    `;

    const resultSection =
        riskScore
            ? riskScore.closest("section")
            : null;

    if (
        resultSection &&
        resultSection.parentNode
    ) {

        resultSection.parentNode.insertBefore(
            section,
            resultSection.nextSibling
        );

    } else if (appPage) {

        appPage.appendChild(
            section
        );

    }

    const clearButton =
        document.getElementById(
            "clearHistoryButton"
        );

    if (clearButton) {

        clearButton.addEventListener(
            "click",
            () => {

                const history =
                    getAnalysisHistory();

                if (history.length === 0) {

                    alert(
                        "Silinecek analiz geçmişi bulunmuyor."
                    );

                    return;
                }

                const confirmed =
                    confirm(
                        "Tüm analiz geçmişini silmek istediğinize emin misiniz?"
                    );

                if (!confirmed) {
                    return;
                }

                localStorage.removeItem(
                    HISTORY_STORAGE_KEY
                );

                renderAnalysisHistory();

            }
        );

    }

    renderAnalysisHistory();
}


/* =========================
   GEÇMİŞİ GÖSTER
========================= */

function renderAnalysisHistory() {

    const list =
        document.getElementById(
            "analysisHistoryList"
        );

    if (!list) {
        return;
    }

    const history =
        getAnalysisHistory();

    if (history.length === 0) {

        list.innerHTML = `

            <div class="empty-history">

                <div class="empty-history-icon">
                    📭
                </div>

                <p>
                    Henüz analiz geçmişi bulunmuyor.
                </p>

                <span>
                    Yaptığınız analizler burada görünecek.
                </span>

            </div>

        `;

        return;
    }

    list.innerHTML =
        "";

    history.forEach((item) => {

        const card =
            document.createElement("div");

        card.className =
            "history-card";

        let typeIcon =
            "💬";

        if (item.type === "Gmail") {

            typeIcon =
                "📧";

        } else if (
            item.type === "URL"
        ) {

            typeIcon =
                "🔗";
        }

        let scoreClass =
            "low-risk";

        if (item.riskScore >= 70) {

            scoreClass =
                "high-risk";

        } else if (
            item.riskScore >= 40
        ) {

            scoreClass =
                "medium-risk";
        }

        const content =
            item.content
                ? escapeHtml(
                    item.content
                )
                : "İçerik bulunamadı.";

        card.innerHTML = `

            <div class="history-card-top">

                <div class="history-type">

                    <span class="history-icon">
                        ${typeIcon}
                    </span>

                    <strong>
                        ${escapeHtml(
                            item.type
                        )}
                    </strong>

                </div>

                <span class="history-date">
                    ${escapeHtml(
                        item.date
                    )}
                </span>

            </div>

            <div class="history-content">

                ${content}

            </div>

            <div class="history-card-bottom">

                <div class="history-result">

                    <span>
                        Risk
                    </span>

                    <strong class="${scoreClass}">
                        ${item.riskScore}/100
                    </strong>

                </div>

                <div class="history-status">

                    ${escapeHtml(
                        item.result
                    )}

                </div>

            </div>

        `;

        list.appendChild(
            card
        );

    });

}


/* =========================
   GEÇMİŞİ SAYFA AÇILINCA YÜKLE
========================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        setTimeout(() => {

            createHistorySection();

        }, 100);

    }
);