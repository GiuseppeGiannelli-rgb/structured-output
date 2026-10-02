"""Selfwork Structured Output.

Due esempi diversi di output strutturati visti nella video-lezione:

1. response_format con modello Pydantic -> tutor di matematica che restituisce
   i passaggi del ragionamento (MathReasoning), con gestione del refusal.
2. Function calling con pydantic_function_tool + Enum -> estrazione dei
   parametri di ricerca prodotto da una richiesta in linguaggio naturale.

Il provider (OpenAI oppure Ollama in locale) si sceglie nel file .env.
"""

import json
import os
from enum import Enum

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

if PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
else:
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # richiesto dal client ma ignorato da Ollama
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ---------------------------------------------------------------------------
# Esempio 1 - response_format con Pydantic (tutor di matematica)
# ---------------------------------------------------------------------------

class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


MATH_TUTOR_PROMPT = """
Sei un tutor di matematica. Ti verrà dato un problema di matematica.
Il tuo obiettivo è spiegare la soluzione passo dopo passo e fornire la risposta finale.
Per ogni passaggio scrivi la spiegazione e il risultato ottenuto.
Rispondi in italiano.
"""


def risolvi_problema(problema: str) -> MathReasoning | None:
    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": MATH_TUTOR_PROMPT},
            {"role": "user", "content": problema},
        ],
        response_format=MathReasoning,
    )
    message = completion.choices[0].message

    # Il modello può rifiutarsi di rispondere: in quel caso parsed è None
    if message.refusal:
        print(f"Il modello ha rifiutato la richiesta: {message.refusal}")
        return None
    return message.parsed


def esempio_1() -> None:
    print("=" * 70)
    print("ESEMPIO 1 - response_format Pydantic (MathReasoning)")
    print("=" * 70)

    problema = "Risolvi l'equazione 8x + 7 = -23"
    print(f"Problema: {problema}\n")

    risultato = risolvi_problema(problema)
    if risultato is None:
        return

    for i, step in enumerate(risultato.steps, start=1):
        print(f"Passo {i}: {step.explanation}")
        print(f"         -> {step.output}")
    print(f"\nRisposta finale: {risultato.final_answer}")

    # L'oggetto è un vero modello Pydantic: si può serializzare/validare
    print("\nJSON restituito:")
    print(risultato.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Esempio 2 - Function calling con pydantic_function_tool (ricerca prodotti)
# ---------------------------------------------------------------------------

class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"


class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str


PRODUCT_SEARCH_PROMPT = """
Sei un assistente per un negozio di abbigliamento online.
Quando ricevi una richiesta dell'utente devi chiamare SEMPRE la funzione
product_search estraendo categoria, sottocategoria e colore del prodotto.
La categoria deve essere una tra: shoes, jackets, tops, bottoms.
"""


def estrai_parametri(richiesta: str) -> ProductSearchParameters | None:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": PRODUCT_SEARCH_PROMPT},
            {"role": "user", "content": richiesta},
        ],
        tools=[
            openai.pydantic_function_tool(
                ProductSearchParameters, name="product_search"
            )
        ],
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("Il modello non ha chiamato la funzione.")
        return None

    # Gli argomenti arrivano come stringa JSON: li validiamo con Pydantic
    arguments = tool_calls[0].function.arguments
    return ProductSearchParameters.model_validate(json.loads(arguments))


def esempio_2() -> None:
    print("\n" + "=" * 70)
    print("ESEMPIO 2 - Function calling (ProductSearchParameters)")
    print("=" * 70)

    richieste = [
        "Sto cercando delle scarpe da running blu",
        "Mi serve una giacca di pelle nera per l'inverno",
        "Vorrei dei jeans slim grigi",
    ]
    for richiesta in richieste:
        parametri = estrai_parametri(richiesta)
        print(f"\nRichiesta: {richiesta}")
        if parametri:
            print(f"  category:    {parametri.category.value}")
            print(f"  subcategory: {parametri.subcategory}")
            print(f"  color:       {parametri.color}")


if __name__ == "__main__":
    print(f"Provider: {PROVIDER} | Modello: {MODEL}\n")
    esempio_1()
    esempio_2()
