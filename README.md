<p align="center">
	<img src="https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6a9936e4f7726409a9ce4092_LangChain_Lockup_Black%201-1.svg" alt="LangChain" width="380" />
</p>

<div align="center">

# Lean LangGraph

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-v1-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

A production-oriented FastAPI application for a LangGraph-powered RAG assistant. It combines retrieval, orchestration,
security checks, tracing, and operational metrics in a structure suitable for local development and deployment.

## Overview

This repository is intended as a practical starting point for building a production-ready retrieval-augmented generation
system with the following capabilities:

- FastAPI HTTP API
- LangGraph orchestration
- LangChain model integration
- caching and request metrics
- input and output validation
- Langfuse observability
- Test automation and coverage checks

## Why this project exists

The goal is to build a secure and observable RAG service that is useful in real deployment scenarios, not just a
notebook prototype. In production, RAG systems often fail due to weak chunking, embedding mismatches, noisy retrieval,
context overflow, and model instability. This repo includes patterns and guardrails to tackle those issues early.

## Features

- `/chat` endpoint with security validation and cache checks
- `/health` readiness endpoint
- `/metrics` summary endpoint
- `/cache/status` cache diagnostics
- prompt injection guardrails
- PII detection and masking for input and output
- structured JSON logging
- request timing and performance metrics
- FastAPI validation with Pydantic models
- pytest-based testing with coverage support