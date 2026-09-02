-- =========================================
-- AI SMS URL Analyzer - Initial Database Schema
-- =========================================

-- 1. Users
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- 2. SMS Messages
CREATE TABLE sms_messages (
    sms_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    received_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_sms_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);


-- 3. URLs
CREATE TABLE urls (
    url_id SERIAL PRIMARY KEY,
    sms_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_url_sms
        FOREIGN KEY (sms_id)
        REFERENCES sms_messages(sms_id)
        ON DELETE CASCADE
);


-- 4. AI Models
CREATE TABLE ai_models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    provider VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_model_version
        UNIQUE (model_name, model_version)
);


-- 5. SMS Analysis Results
CREATE TABLE sms_analysis_results (
    analysis_id SERIAL PRIMARY KEY,
    sms_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    confidence NUMERIC(5,4),
    analysis_text TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_sms_analysis_sms
        FOREIGN KEY (sms_id)
        REFERENCES sms_messages(sms_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_sms_analysis_model
        FOREIGN KEY (model_id)
        REFERENCES ai_models(model_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_sms_confidence
        CHECK (confidence >= 0 AND confidence <= 1)
);


-- 6. URL Analysis Results
CREATE TABLE url_analysis_results (
    analysis_id SERIAL PRIMARY KEY,
    url_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    confidence NUMERIC(5,4),
    analysis_text TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_url_analysis_url
        FOREIGN KEY (url_id)
        REFERENCES urls(url_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_url_analysis_model
        FOREIGN KEY (model_id)
        REFERENCES ai_models(model_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_url_confidence
        CHECK (confidence >= 0 AND confidence <= 1)
);