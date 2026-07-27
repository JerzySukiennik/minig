"""Teach MiniG to turn a spoken Polish command into a Home Assistant call.

Gzowo AI already owns the plumbing — `control_room(room, service, value?)`
dispatched through its tool router, with the real entity map in
`bridge/ha-rooms.json` and readback from the live house. Today Gemini Live
does the parsing. This generates the data that lets a 178M local model do it
instead: no internet, no latency, no API key.

Why this is within reach at this size, when open-ended knowledge is not: it is
a closed set. Eleven rooms, two services, one optional number. MicroG already
scores 100% on identity (a closed memorised set) and generalised the "list N
things" format to categories it never saw, so mapping a phrasing onto one of
eleven slots is easier than what it already does.

Two design decisions follow from Jurek's choice to run this **without a Gemini
fallback**:

  Refusals are trained, not assumed. With no larger model behind it, a command
  the model does not understand must produce an honest "I don't know that
  place" rather than a guess. A guessed room switches off the wrong lights in
  a real house, so roughly a fifth of the examples are unknown rooms and
  unanswerable requests whose correct answer is to say so.

  Inflection is taught explicitly. The command says "w kuchni", "na wiacie",
  "przy drukarce"; the tool wants the nominative "kuchnia", "wiata",
  "drukarka". Polish declension is the actual mapping problem here, and it is
  exactly what broke MicroG's identity examples until accent-free variants
  were added — so every room ships with the forms people really say.

The app must still validate: anything the model emits is checked against the
live entity registry before it reaches a lamp. Training reduces how often the
model is wrong; it never makes the output trustworthy on its own.
"""

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np

U, A, EOT = "<|user|>", "<|assistant|>", "<|endoftext|>"

HA_ROOMS_JSON = Path.home() / "Downloads/Claude/Projects/AIe/Gzowo AI/v1/bridge/ha-rooms.json"

# Canonical room -> how it is actually said, split by the grammar each slot
# needs. A first version glued a bare nominative onto a light word and
# produced "Włącz wszystkie światła salon.", which is not Polish — training on
# ungrammatical commands teaches the model that they are normal.
#
#   place : goes after a light word ("Zgaś światło <w salonie>")
#   direct: the target on its own ("Zgaś <salon>") — colloquial but correct
ROOM_FORMS = {
    "salon":    {"place": ["w salonie"],                  "direct": ["salon", "salonu"]},
    "kuchnia":  {"place": ["w kuchni"],                   "direct": ["kuchnię", "kuchnia"]},
    "łazienka": {"place": ["w łazience"],                 "direct": ["łazienkę", "łazienka"]},
    "dom":      {"place": ["w domu", "w całym domu"],     "direct": ["dom", "cały dom"]},
    "wiata":    {"place": ["na wiacie", "pod wiatą"],     "direct": ["wiatę", "wiata"]},
    "kontener": {"place": ["w kontenerze"],               "direct": ["kontener"]},
    "cypel":    {"place": ["na cyplu"],                   "direct": ["cypel"]},
    "kanałek":  {"place": ["przy kanałku", "na kanałku"], "direct": ["kanałek"]},
    "latarnie": {"place": ["na latarniach"],              "direct": ["latarnie"]},
    "domki":    {"place": ["w domkach", "przy domkach"],  "direct": ["domki"]},
    "drukarka": {"place": ["przy drukarce"],              "direct": ["drukarkę", "drukarka"]},
}

ON_VERBS = ["Włącz", "Zapal", "Zapalcie", "Odpal", "Daj", "Puść"]
OFF_VERBS = ["Wyłącz", "Zgaś", "Zgaście", "Wygaś", "Ubij"]
LIGHT_WORDS = ["światło", "światła", "lampę", "lampy", "wszystkie światła"]

# Places that are not in the map. The right answer is to say so — with no
# fallback model behind this one, a confident guess is the dangerous outcome.
# Locative forms, because they follow "w" in the generated command. The
# nominative list this replaced produced "Zapal światło w stodoła."
UNKNOWN_PLACES = ["garażu", "piwnicy", "strychu", "sypialni", "biurze",
                  "tarasie", "przedpokoju", "spiżarni", "stodole", "warsztacie",
                  "ogrodzie", "pralni", "gabinecie", "werandzie", "kotłowni"]
REFUSAL = "Nie znam takiego miejsca w mapie domu."

# Things a home model will be asked that it must not answer with a tool call.
OUT_OF_SCOPE = [
    ("Jaka jest jutro pogoda?", "Nie umiem sprawdzić pogody."),
    ("Włącz muzykę.", "Nie umiem sterować muzyką."),
    ("Otwórz bramę.", "Nie znam takiego urządzenia w mapie domu."),
    ("Ustaw alarm na siódmą.", "Nie umiem ustawiać alarmów."),
    ("Podlej ogród.", "Nie znam takiego urządzenia w mapie domu."),
    ("Zamknij okna.", "Nie znam takiego urządzenia w mapie domu."),
    ("Włącz ogrzewanie.", "Nie znam takiego urządzenia w mapie domu."),
    ("Zrób kawę.", "Nie umiem tego zrobić."),
]

REPEATS = 6          # closed set: repetition is what makes a small model commit
DIM_LEVELS = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]


def strip_diacritics(text: str) -> str:
    """'w łazience' -> 'w lazience'. People type Polish without accents, and
    MicroG proved the model treats the two as unrelated strings unless both
    are taught."""
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def call(room, service, value=None):
    if value is None:
        return f'control_room(room="{room}", service="{service}")'
    return f'control_room(room="{room}", service="{service}", value={value})'


def verify_rooms_against_map(rooms):
    """The generated rooms must exist in the real map, or we are training the
    model to call things that do not exist."""
    if not HA_ROOMS_JSON.exists():
        print(f"UWAGA: nie znaleziono {HA_ROOMS_JSON} — pomijam weryfikację")
        return
    data = json.loads(HA_ROOMS_JSON.read_text(encoding="utf-8"))
    real = {r.lower() for e in data.get("lights", []) for r in e.get("rooms", [])}
    ours = set(rooms)
    missing = ours - real
    unused = real - ours
    print(f"pokoje w mapie HA: {len(real)}  |  w danych treningowych: {len(ours)}")
    if missing:
        print(f"  BŁĄD: uczymy pokoi, których nie ma w HA: {sorted(missing)}")
    if unused:
        print(f"  uwaga: w HA są pokoje bez przykładów: {sorted(unused)}")


def generate(seed=0):
    rng = np.random.default_rng(seed)
    out = []

    seen = set()

    def add(question, answer):
        # Stripping accents from a phrase that has none returns the same
        # string, so every accent-free room would otherwise contribute each of
        # its examples twice.
        if question in seen:
            return
        seen.add(question)
        out.append((question, answer))

    for room, forms in ROOM_FORMS.items():
        for service, verbs in (("turn_on", ON_VERBS), ("turn_off", OFF_VERBS)):
            for verb in verbs:
                # "Zgaś światło w salonie." — light word plus a place phrase.
                for place in forms["place"]:
                    light = LIGHT_WORDS[int(rng.integers(len(LIGHT_WORDS)))]
                    q = f"{verb} {light} {place}."
                    add(q, call(room, service))
                    add(strip_diacritics(q), call(room, service))
                # "Zgaś salon." — the room itself as the object, no light word.
                for direct in forms["direct"]:
                    q = f"{verb} {direct}."
                    add(q, call(room, service))
                    add(strip_diacritics(q), call(room, service))

        # Dimming only makes sense with turn_on, and carries the numbers the
        # new tokenizer exists to get right.
        for place in forms["place"]:
            for _ in range(4):
                v = DIM_LEVELS[int(rng.integers(len(DIM_LEVELS)))]
                verb = ["Przygaś", "Ściemnij"][int(rng.integers(2))]
                for q in (f"{verb} światło {place} na {v}%.",
                          f"Ustaw jasność {place} na {v}%."):
                    add(q, call(room, "turn_on", v))
                    add(strip_diacritics(q), call(room, "turn_on", v))

    for place in UNKNOWN_PLACES:
        for verb in ON_VERBS[:3] + OFF_VERBS[:3]:
            q = f"{verb} światło w {place}."
            add(q, REFUSAL)
            add(strip_diacritics(q), REFUSAL)

    for q, a in OUT_OF_SCOPE:
        add(q, a)
        add(strip_diacritics(q), a)

    return out


def main(out_path: Path, seed: int):
    pairs = generate(seed)
    verify_rooms_against_map(list(ROOM_FORMS))

    tool_calls = sum(1 for _, a in pairs if a.startswith("control_room"))
    refusals = len(pairs) - tool_calls
    print(f"\nwygenerowano {len(pairs):,} unikalnych par "
          f"({tool_calls:,} wywołań, {refusals:,} odmów — {refusals/len(pairs)*100:.0f}%)")
    print(f"po powtórzeniach x{REPEATS}: {len(pairs)*REPEATS:,} wierszy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(json.dumps({"instruction": q, "output": a,
                                "repeats": REPEATS}, ensure_ascii=False) + "\n")
    print(f"zapisano -> {out_path}")

    print("\nprzykłady:")
    for q, a in pairs[:4] + pairs[-4:]:
        print(f"  {q!r}\n    -> {a}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/home_sft.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.out, args.seed)
