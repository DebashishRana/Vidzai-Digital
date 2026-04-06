-- Migration: Create verification_logs table for storing KYC verification records
-- Date: 2024-04-02
-- Purpose: Store all verification records with scores, status, and audit trail

USE ekyc;

CREATE TABLE IF NOT EXISTS verification_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    verification_id VARCHAR(255) UNIQUE NOT NULL,
    document_type VARCHAR(100),
    confidence_score DECIMAL(5,2),
    status VARCHAR(50),
    detected_address TEXT,
    verification_details JSON,
    face_verified BOOLEAN DEFAULT FALSE,
    tampering_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_verification_id (verification_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_confidence_score (confidence_score),
    INDEX idx_document_type (document_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create rollback version (for reverting if needed)
-- DROP TABLE IF EXISTS verification_logs;
