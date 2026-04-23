# AI Agriculture System

AI-based agriculture system that uses:

- Reinforcement Learning (PPO)
- RAG (knowledge retrieval)
- Generative AI (LLM explanations)
- Physics-based models

## What it does

- Takes sensor + weather data  
- Calculates plant health  
- Decides irrigation using RL  
- Retrieves knowledge using RAG  
- Generates explanation using GenAI  

## Run

```bash
pip install -r requirements.txt
uvicorn backend.test_app:app --port 9000

## open

http://127.0.0.1:9000/status
