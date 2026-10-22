# Transportation & Assignment AI Agent

An AI agent built with Google Gemini that solves **Transportation** and **Assignment** problems. Describe a problem in plain English or paste a matrix, and the agent runs the right solver and explains the result.

## Features
- **Transportation:** North-West Corner, Least Cost, Vogel's Approximation, or the optimal solution
- **Assignment:** Hungarian method (minimise or maximise)
- Handles unbalanced problems automatically

## Setup
1. Install packages: `pip install -r requirements.txt`
2. Get a free key from [Google AI Studio](https://aistudio.google.com/apikey)
3. Create a `.env` file containing: `GEMINI_API_KEY=your-key-here`
4. Run: `python agent.py`

## Built with
Python, Google Gemini API, NumPy, SciPy
