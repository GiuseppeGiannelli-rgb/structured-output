"""Selfwork Structured Output – due esempi originali di output strutturati.

1. RICETTARIO (response_format + Pydantic)
   Una ricetta scritta "a voce", disordinata, viene trasformata in un oggetto
   Ricetta validato: ingredienti con quantità e unità, tempi, difficoltà (Enum),
   allergeni (lista di Enum). Dall'oggetto si calcolano poi le dosi per N persone.

2. AGENDA (function calling + pydantic_function_tool)
   Frasi in linguaggio naturale ("fissa una call con Marco domani alle 15...")
   diventano chiamate strutturate alla funzione crea_evento, che le salva in un'agenda.
   Il modello deve rispettare lo schema: tipo e priorità sono Enum, data e ora hanno
   un formato preciso.

Il provider (Ollama in locale oppure OpenAI) si sceglie nel file .env.
"""

import json
import os
from datetime import date, datetime, timedelta
from enum import Enum

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

if PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
else:
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # richiesto dal client ma ignorato da Ollama
        timeout=180,
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ---------------------------------------------------------------------------
# Esempio 1 - Ricettario: response_format con un modello Pydantic annidato
# ---------------------------------------------------------------------------

class Difficolta(str, Enum):
    facile = "facile"
    media = "media"
    difficile = "difficile"


class Allergene(str, Enum):
    glutine = "glutine"
    latte = "latte"
    uova = "uova"
    frutta_a_guscio = "frutta a guscio"
    pesce = "pesce"
    crostacei = "crostacei"
    soia = "soia"
    sedano = "sedano"


class Ingrediente(BaseModel):
    nome: str
    quantita: float = Field(description="Quantità numerica per il numero di porzioni della ricetta")
    unita: str = Field(description="g, ml, pz, cucchiai, q.b. ...")


class Ricetta(BaseModel):
    titolo: str
    porzioni: int
    tempo_preparazione_min: int
    tempo_cottura_min: int
    difficolta: Difficolta
    vegetariana: bool
    allergeni: list[Allergene]
    ingredienti: list[Ingrediente]
    passaggi: list[str] = Field(description="Passaggi in ordine, una frase ciascuno")


RICETTA_PROMPT = """
Sei un redattore di un ricettario. Ricevi la descrizione informale di una ricetta
e la trasformi in una scheda strutturata in italiano.
- Converti le quantità in numeri (es. "mezzo chilo" = 500 g, "un paio di uova" = 2 pz).
- Stima i tempi in minuti se non sono espliciti.
- Elenca solo gli allergeni realmente presenti negli ingredienti.
"""

RICETTA_TESTO = """
Allora, per la pasta alla carbonara per quattro persone mi servono mezzo chilo di
spaghetti, un etto e mezzo di guanciale e quattro tuorli, più un uovo intero.
Poi pecorino romano, diciamo 80 grammi, e pepe nero quanto basta.
Taglio il guanciale a listarelle e lo faccio rosolare in padella senza olio finché
non diventa croccante, ci vogliono 8-10 minuti. Intanto sbatto i tuorli e l'uovo con
il pecorino e tanto pepe. Butto la pasta, la scolo al dente dopo 10 minuti, la verso
nella padella col guanciale a fuoco spento e aggiungo la crema di uova mescolando
veloce con un po' di acqua di cottura. Servo subito.
"""


def estrai_ricetta(testo: str) -> Ricetta | None:
    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": RICETTA_PROMPT},
            {"role": "user", "content": testo},
        ],
        response_format=Ricetta,
    )
    message = completion.choices[0].message
    if message.refusal:  # il modello può rifiutarsi: in quel caso parsed è None
        print(f"Il modello ha rifiutato la richiesta: {message.refusal}")
        return None
    return message.parsed


def scala_dosi(ricetta: Ricetta, persone: int) -> list[Ingrediente]:
    """Usa i dati strutturati per ricalcolare le dosi: impossibile con testo libero."""
    fattore = persone / ricetta.porzioni
    return [
        ing.model_copy(update={"quantita": round(ing.quantita * fattore, 1)})
        for ing in ricetta.ingredienti
    ]


def esempio_1() -> None:
    print("=" * 70)
    print("ESEMPIO 1 - Ricettario (response_format + Pydantic)")
    print("=" * 70)

    ricetta = estrai_ricetta(RICETTA_TESTO)
    if ricetta is None:
        return

    print(f"{ricetta.titolo} | {ricetta.porzioni} porzioni | difficoltà: {ricetta.difficolta.value}")
    print(f"Preparazione {ricetta.tempo_preparazione_min} min + cottura {ricetta.tempo_cottura_min} min")
    print(f"Vegetariana: {'sì' if ricetta.vegetariana else 'no'}")
    print(f"Allergeni: {', '.join(a.value for a in ricetta.allergeni) or 'nessuno'}")

    print("\nIngredienti:")
    for ing in ricetta.ingredienti:
        print(f"  - {ing.nome}: {ing.quantita:g} {ing.unita}")

    print("\nPassaggi:")
    for i, passo in enumerate(ricetta.passaggi, start=1):
        print(f"  {i}. {passo}")

    print("\nDosi ricalcolate per 6 persone:")
    for ing in scala_dosi(ricetta, 6):
        print(f"  - {ing.nome}: {ing.quantita:g} {ing.unita}")


# ---------------------------------------------------------------------------
# Esempio 2 - Agenda: function calling con pydantic_function_tool
# ---------------------------------------------------------------------------

class TipoEvento(str, Enum):
    riunione = "riunione"
    chiamata = "chiamata"
    scadenza = "scadenza"
    personale = "personale"


class Priorita(str, Enum):
    bassa = "bassa"
    media = "media"
    alta = "alta"


class CreaEvento(BaseModel):
    """Aggiunge un evento all'agenda dell'utente."""

    titolo: str = Field(description="Titolo breve dell'evento")
    tipo: TipoEvento
    data: str = Field(description="Data nel formato YYYY-MM-DD")
    ora: str = Field(description="Ora di inizio nel formato HH:MM (24 ore)")
    durata_min: int = Field(description="Durata in minuti, 60 se non indicata")
    partecipanti: list[str] = Field(description="Nomi delle persone coinvolte, vuota se nessuna")
    priorita: Priorita


AGENDA: list[CreaEvento] = []


def crea_evento(evento: CreaEvento) -> str:
    """La funzione 'reale' che il modello chiama: salva l'evento e controlla i conflitti."""
    try:
        inizio = datetime.strptime(f"{evento.data} {evento.ora}", "%Y-%m-%d %H:%M")
    except ValueError:
        return f"Data/ora non valide: {evento.data} {evento.ora}"
    fine = inizio + timedelta(minutes=evento.durata_min)
    for altro in AGENDA:
        a_inizio = datetime.strptime(f"{altro.data} {altro.ora}", "%Y-%m-%d %H:%M")
        a_fine = a_inizio + timedelta(minutes=altro.durata_min)
        if inizio < a_fine and a_inizio < fine:
            return f"CONFLITTO con '{altro.titolo}' ({altro.data} {altro.ora})"
    AGENDA.append(evento)
    return f"Evento salvato: {evento.titolo} il {evento.data} alle {evento.ora}"


def agenda_prompt() -> str:
    oggi = date.today()
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    return (
        "Sei un assistente che gestisce l'agenda dell'utente. "
        f"Oggi è {giorni[oggi.weekday()]} {oggi.isoformat()}. "
        "Per ogni richiesta chiama la funzione crea_evento convertendo le date relative "
        "(domani, venerdì prossimo...) in date reali. Priorità alta solo se l'utente "
        "indica urgenza o importanza."
    )


def interpreta_richiesta(richiesta: str) -> CreaEvento | None:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": agenda_prompt()},
            {"role": "user", "content": richiesta},
        ],
        tools=[openai.pydantic_function_tool(CreaEvento, name="crea_evento")],
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        print("  Il modello non ha chiamato la funzione.")
        return None

    # Gli argomenti arrivano come stringa JSON: Pydantic li valida contro lo schema
    try:
        return CreaEvento.model_validate(json.loads(tool_calls[0].function.arguments))
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  Argomenti non validi: {e}")
        return None


def esempio_2() -> None:
    print("\n" + "=" * 70)
    print("ESEMPIO 2 - Agenda (function calling + pydantic_function_tool)")
    print("=" * 70)

    richieste = [
        "Fissa una call con Marco e Giulia domani alle 15 per mezz'ora, è urgente",
        "Venerdì alle 9:30 riunione di team di due ore sul nuovo progetto",
        "Ricordami dopodomani alle 18 di andare in palestra",
        "Domani alle 15:15 dentista",  # si sovrappone alla call -> conflitto
    ]
    for richiesta in richieste:
        print(f"\nRichiesta: {richiesta}")
        evento = interpreta_richiesta(richiesta)
        if evento:
            print(f"  -> {evento.model_dump_json()}")
            print(f"  {crea_evento(evento)}")

    print("\nAgenda finale:")
    for e in sorted(AGENDA, key=lambda e: (e.data, e.ora)):
        persone = f" con {', '.join(e.partecipanti)}" if e.partecipanti else ""
        print(f"  {e.data} {e.ora} ({e.durata_min} min) [{e.tipo.value}, {e.priorita.value}] {e.titolo}{persone}")


if __name__ == "__main__":
    print(f"Provider: {PROVIDER} | Modello: {MODEL}\n")
    esempio_1()
    esempio_2()
