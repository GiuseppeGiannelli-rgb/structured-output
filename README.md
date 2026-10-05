# Selfwork Structured Output

Due esempi originali di **Structured Outputs** in un unico file (`structured_outputs.py`), con due tecniche diverse.

## 1. Ricettario – `response_format` + Pydantic

Una ricetta raccontata in modo informale ("mezzo chilo di spaghetti, un etto e mezzo di guanciale...") viene trasformata con `client.chat.completions.parse(response_format=Ricetta)` in un oggetto validato:

- modelli annidati (`Ricetta` → lista di `Ingrediente` con quantità numerica e unità);
- `Enum` per la difficoltà e per gli allergeni (il modello non può inventare valori);
- gestione del `refusal`.

Poiché i dati sono strutturati, il programma **ricalcola le dosi** per un altro numero di persone: con testo libero non sarebbe possibile.

## 2. Agenda – function calling con `pydantic_function_tool`

Richieste in linguaggio naturale ("Fissa una call con Marco e Giulia domani alle 15 per mezz'ora, è urgente") diventano chiamate alla funzione `crea_evento`, definita con `openai.pydantic_function_tool(CreaEvento)`:

- `tipo` e `priorita` sono `Enum`, data e ora hanno un formato preciso;
- le date relative (domani, venerdì) vengono convertite in date reali;
- gli argomenti vengono validati con Pydantic e la funzione salva l'evento **controllando i conflitti** di orario.

## Setup

```bash
poetry install
cp .env.example .env   # PROVIDER=ollama oppure openai (+ OPENAI_API_KEY)
ollama pull llama3.2   # solo se usi Ollama
```

## Esecuzione

```bash
poetry run python structured_outputs.py
```
