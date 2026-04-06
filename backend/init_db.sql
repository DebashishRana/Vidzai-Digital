-- eKYC Database Setup Script
-- Run this script in MySQL to create all required tables

CREATE DATABASE IF NOT EXISTS ekyc;
USE ekyc;

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
);

-- Verifications Table
CREATE TABLE IF NOT EXISTS verifications (
    id VARCHAR(50) PRIMARY KEY,
    user_id INT DEFAULT 1,
    document_type VARCHAR(100) DEFAULT 'Aadhar',
    document_path VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending',
    risk_score FLOAT DEFAULT 0,
    confidence FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_status (status),
    INDEX idx_created (created_at),
    INDEX idx_user (user_id)
);

-- Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(50) PRIMARY KEY,
    verification_id VARCHAR(50),
    file_path VARCHAR(255),
    extracted_text LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (verification_id) REFERENCES verifications(id) ON DELETE CASCADE,
    INDEX idx_verification (verification_id),
    INDEX idx_created (created_at)
);

-- Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(50) PRIMARY KEY,
    verification_id VARCHAR(50),
    risk_level VARCHAR(50) DEFAULT 'Medium',
    alert_type VARCHAR(100) DEFAULT 'Fraud Alert',
    status VARCHAR(20) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (verification_id) REFERENCES verifications(id) ON DELETE CASCADE,
    INDEX idx_verification (verification_id),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
);

-- Create default user
INSERT INTO users (id, name, email) VALUES (1, 'System', 'system@ekyc.local') ON DUPLICATE KEY UPDATE id=id;

-- Display confirmation
SELECT 'eKYC Database initialized successfully!' AS status;
SHOW TABLES;
