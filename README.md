# Selfwork Structured Output

Due esempi di **Structured Outputs** in un unico file (`structured_outputs.py`):

1. **response_format con Pydantic** – tutor di matematica (`MathReasoning`): il modello restituisce i passaggi del ragionamento e la risposta finale come oggetto Pydantic (`message.parsed`), con gestione del `refusal`.
2. **Function calling** – `openai.pydantic_function_tool` + `Enum` (`ProductSearchParameters`): da una richiesta in linguaggio naturale si estraggono categoria, sottocategoria e colore.

## Setup

```bash
poetry install
cp .env.example .env   # imposta PROVIDER=ollama oppure openai (+ OPENAI_API_KEY)
```

Con Ollama: `ollama pull llama3.2` e Ollama avviato.

## Esecuzione

```bash
poetry run python structured_outputs.py
```
