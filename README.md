# AI Marketing Agent

AI-assisted marketing analytics agent for campaign evaluation, decision support, scenario simulation, and dashboard-based analysis.

## Overview

This project analyzes marketing campaign data and generates structured recommendations based on performance, efficiency, maturity, and trend signals.

The system combines deterministic decision logic with an optional LLM-assisted advisor. It evaluates campaign health using metrics such as CPA, ROAS, 7-day trends, campaign maturity, stance, severity, and business reasons. It also includes scenario simulation and a Streamlit dashboard for reviewing campaign outputs.

The goal is to build something closer to a practical analytics decision-support tool than a static dashboard: a system that helps identify which campaigns need attention, why they matter, and what action could be considered next.

## Main Features

- Campaign evaluation using rule-based goals
- Decision outputs with stance, severity, and business reasons
- CPA, ROAS, campaign maturity, and 7-day trend analysis
- Scenario simulation for budget changes
- Streamlit dashboard for inspecting campaign outputs
- Optional LLM-assisted advisor for structured recommendations
- Deterministic fallback logic when LLM support is unavailable
- Memory and run storage for historical context

## Tech Stack

- Python
- Streamlit
- SQLite
- pandas
- Pydantic
- OpenAI API
- dotenv

## Project Structure

```text
agent/
dashboards/
data/
registry/
storage/
tools/
main.py
requirements.txt
README.md
