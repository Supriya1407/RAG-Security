# 🔐 Secure RAG System for Sensitive Data Protection

## 📌 Overview

This project implements a **Secure Retrieval-Augmented Generation (RAG) system** designed to prevent leakage of sensitive information in Large Language Model (LLM) applications. It integrates **PII redaction** and **context-based access control** to ensure privacy-preserving responses in domains like healthcare, finance, and legal systems.

## 🚀 Features

* 🔍 Automated **PII detection & redaction** (names, dates, phone numbers, etc.)
* 🔐 **Context-Based Access Control (CBAC)** for secure data retrieval
* 📚 RAG pipeline with **filtered document access**
* 📊 Redaction logging and summary reports
* 🧪 Adversarial testing for privacy leakage prevention

## 🏗️ Architecture

User Query → PII Redaction → Access Control → Document Retrieval → LLM → Secure Response

## ⚙️ Tech Stack

* Python
* spaCy, NLTK (NER & NLP)
* Presidio (PII detection)
* FAISS / Pinecone (vector database)
* WordNet (semantic filtering)

## 📈 Results

* Accurate detection and masking of sensitive data
* Works across multiple document types (medical, legal, financial)
* Maintains readability while ensuring privacy

## ⚠️ Limitations

* Limited support for non-standard identifiers
* Context mismatch in synonym-based filtering
* Scalability constraints for large datasets

## 🔮 Future Work

* Context-aware models (BERT) for improved redaction
* Parallel processing for scalability
* Integration of differential privacy

