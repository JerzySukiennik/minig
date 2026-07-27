"""
Build the instruction-tuning dataset.

Pretraining taught G-Micro what Polish looks like. It did not teach it that a
question is followed by an answer — in web text a question is usually followed
by more questions. This stage fixes that, and only that.

Two things make it different from pretraining:

  Chat template. Turns are wrapped in the control tokens reserved in the
  tokenizer before pretraining started, so no embedding resize is needed.

  Loss masking. The model is scored only on the assistant's reply. Training on
  the user's turn as well would teach it to invent questions, which is exactly
  the behaviour we are trying to remove.

Output: pl_sft_tokens.bin (uint16) and pl_sft_mask.bin (uint8), same length.
Mask 1 = a position the loss is computed on.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SOURCES = [
    # (repo, weight of trust). The cleaned set is the backbone; the second adds
    # coverage at the cost of translation artefacts, which clean() handles.
    "saillab/alpaca-polish-cleaned",
    "Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish",
]

U, A, EOT, CTX = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"

# Neither Alpaca corpus contains a single example teaching the model its own
# name — confirmed live, 2026-07-22: asked "jak masz na imię?" the SFT model
# answered "Tak, mam na imię." with no name at all, because nothing in its
# training data ever associated one with it. Hand-written and repeated many
# times so a 110M model actually memorises the pattern instead of it being one
# drop in ~90k unrelated examples — this is exactly the kind of short, fixed
# association a small model CAN reliably learn, unlike open-ended world facts.
IDENTITY_EXAMPLES = [
    ("Jak masz na imię?", "Nazywam się G-Mini."),
    ("Jak się nazywasz?", "Nazywam się G-Mini."),
    ("Kim jesteś?", "Jestem G-Mini, model językowy stworzony od zera przez Jurka."),
    ("Czym jesteś?", "Jestem modelem językowym o nazwie G-Mini. Mam około 180 milionów parametrów."),
    ("Powiedz mi kim jesteś.", "Jestem G-Mini — model językowy wytrenowany od podstaw przez Jurka."),
    ("Kto cię stworzył?", "Zostałem stworzony od zera przez Jurka, w ramach projektu Gzowo AI."),
    ("Ile masz parametrów?", "Mam około 180 milionów parametrów."),
    ("Jesteś duży czy mały?", "Jestem niewielkim modelem — mam około 180 milionów parametrów."),
    ("Czy jesteś ChatGPT?", "Nie, nie jestem ChatGPT. Nazywam się G-Mini i jestem osobnym, dużo mniejszym modelem."),
    ("Kto cię wytrenował?", "Wytrenował mnie Jurek, od zera, na własną rękę."),
    ("Jak masz na imię", "Nazywam się G-Mini."),
    ("Przedstaw się.", "Jestem G-Mini, model językowy stworzony od zera przez Jurka."),
    ("Opowiedz coś o sobie.", "Nazywam się G-Mini. Jestem modelem językowym wytrenowanym od zera przez Jurka."),
    ("Skąd pochodzisz?", "Zostałem stworzony od zera przez Jurka, w ramach projektu Gzowo AI."),
    ("Czy masz imię?", "Tak, nazywam się G-Mini."),
    ("Jak brzmi twoja nazwa?", "Nazywam się G-Mini."),
    ("Czy jesteś Claude?", "Nie, nie jestem Claude. Nazywam się G-Mini i jestem osobnym, dużo mniejszym modelem."),
    ("Czy jesteś Gemini?", "Nie, nie jestem Gemini. Nazywam się G-Mini i jestem osobnym, dużo mniejszym modelem."),
    ("Czy jesteś człowiekiem?", "Nie, jestem modelem językowym o nazwie G-Mini."),
]
# Was 60, when there were 16 written examples and no accent-free variants.
# With 37 distinct forms that became 2,220 rows — enough weight that identity
# started bleeding into questions that merely *look* like identity questions:
# "Kto napisał Pana Tadeusza?" answered "jestem modelem językowym o nazwie
# G-Micro", colliding with the trained "Kto cię stworzył?". Identity scores
# 100% both spellings with p≈0.95 on the first token, so there is room to give
# some of that weight back and let ordinary questions win again.
IDENTITY_REPEATS = 40


def strip_diacritics(text: str) -> str:
    """Polish text with the accents removed — 'imię' -> 'imie'.

    NFD decomposition splits an accented letter into base + combining mark, so
    dropping the marks leaves the bare letter. 'ł' is the one Polish letter
    that has no decomposition (the stroke is part of the glyph, not a combining
    mark), so it needs mapping by hand or it would survive untouched.
    """
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def identity_pairs():
    """IDENTITY_EXAMPLES plus an accent-free variant of every question.

    Confirmed live, 2026-07-27: after the first identity fix landed, "Jak masz
    na imię?" answered "Nazywam się G-Micro." with p=0.92 on the first token —
    solidly memorised — while "Jak masz na imie?" (same question typed without
    the ogonek) answered "Mam na imie." and "Kim jestes?" wandered off into
    unrelated chatter. BPE sees 'imię' and 'imie' as different token
    sequences, and 16 strings x60 repeats memorises *those sequences*, not the
    concept behind them, so there is nothing to generalise from. People type
    Polish without diacritics constantly, so both spellings have to be taught.

    Only the question is stripped, never the answer: sloppy input should still
    get a correctly-written Polish reply back.
    """
    seen = set()
    for instruction, output in IDENTITY_EXAMPLES:
        for variant in (instruction, strip_diacritics(instruction)):
            if variant not in seen:
                seen.add(variant)
                yield variant, output

# Measured 2026-07-27: list_correct 43%, the lowest score on the board. The
# failure is not knowledge but shape — asked "Wymień trzy owoce." the model
# answers "Trzy jabłka to jeden z najbardziej znanych owoców na świecie…",
# latching onto "trzy X" as a noun phrase and writing prose about it. Same for
# "Wymień trzy zwierzęta." -> "Trzy koty, które są zwierzętami społecznymi…".
# Neither Alpaca corpus has many Polish "list N things" instructions, so the
# pattern was never really taught.
#
# The categories here are deliberately DISJOINT from the ones in
# bench/chat_eval.py (which tests fruit, animals, colours, days, seasons,
# planets, cities, transport, instruments). Training on the benchmark's own
# questions would turn the score into a memorisation check; keeping them apart
# means the benchmark measures whether the habit generalises to categories the
# model never saw listed.
LIST_CATEGORIES = [
    ("warzywa", ["marchew", "ziemniak", "pomidor", "ogórek", "cebula", "kapusta",
                 "burak", "papryka", "brokuł", "fasola", "groch", "dynia"]),
    ("miesiące", ["styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
                  "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"]),
    ("zawody", ["lekarz", "nauczyciel", "piekarz", "stolarz", "kierowca", "pielęgniarka",
                "murarz", "fryzjer", "kucharz", "listonosz", "mechanik", "rolnik"]),
    ("dyscypliny sportowe", ["piłka nożna", "koszykówka", "siatkówka", "pływanie",
                             "tenis", "lekkoatletyka", "kolarstwo", "narciarstwo",
                             "boks", "wioślarstwo"]),
    ("napoje", ["woda", "herbata", "kawa", "sok", "mleko", "kompot", "lemoniada"]),
    ("meble", ["stół", "krzesło", "szafa", "łóżko", "biurko", "regał", "komoda", "fotel"]),
    ("części ubrania", ["koszula", "spodnie", "kurtka", "sweter", "sukienka",
                        "czapka", "szalik", "skarpetki"]),
    ("kraje europejskie", ["Polska", "Niemcy", "Francja", "Hiszpania", "Włochy",
                           "Czechy", "Słowacja", "Szwecja", "Norwegia", "Portugalia"]),
    ("polskie rzeki", ["Wisła", "Odra", "Warta", "Bug", "Narew", "San", "Pilica", "Noteć"]),
    ("drzewa", ["dąb", "brzoza", "sosna", "świerk", "buk", "klon", "lipa", "jesion"]),
    ("ptaki", ["wróbel", "bocian", "jaskółka", "sowa", "orzeł", "gołąb", "sikorka", "kruk"]),
    ("przedmioty szkolne", ["matematyka", "biologia", "historia", "chemia", "fizyka",
                            "geografia", "informatyka", "muzyka"]),
    ("figury geometryczne", ["kwadrat", "koło", "trójkąt", "prostokąt", "romb",
                             "trapez", "sześciokąt"]),
    ("metale", ["żelazo", "miedź", "cynk", "ołów", "srebro", "złoto", "aluminium", "nikiel"]),
]
LIST_COUNT_WORDS = {2: "dwa", 3: "trzy", 4: "cztery", 5: "pięć"}
# "dwa"/"trzy" do not agree with every noun ("dwa warzywa" is fine, "dwa
# części ubrania" is not), so each category carries the phrasing that works
# for it rather than the count being glued on blindly.
LIST_PHRASINGS = ["Wymień {c} {k}.", "Podaj {c} {k}.", "Wypisz {c} {k}.",
                  "Wymień {c} {k}", "Podaj przykłady — {c} {k}."]
LIST_VARIANTS_PER_CATEGORY = 30
LIST_REPEATS = 4


# Measured 2026-07-27: grounding on real-length contexts scores 11% (1/9)
# while the same skill on two-sentence contexts scores 65%. The failures are
# never "I cannot find it" — they are "8 321" for 8 431, "w Kalininie" for
# w Kaliszu, "północno-zachodniej" for południowo-wschodniej. The model locates
# roughly the right place and then emits a plausible neighbour instead of what
# is written.
#
# This is not a shortage of context data: PoQuAD contexts already have a median
# of 776 characters and 78% run past 600. What PoQuAD does not provide is
# *pressure*. Its questions are natural, so a paragraph usually contains one
# plausible candidate and approximate attention is enough to look right. These
# drills remove that slack — every context carries six to nine competing values
# of the same type, so only exact retrieval produces the exact answer.
#
# Values are drawn at random per example, so nothing here can be memorised: the
# only strategy that generalises is reading. Names are invented from syllables
# for the same reason, and every sentence template keeps invented proper nouns
# in the nominative ("Burmistrzem miasta jest Zofia Malczyk", "Rzeka nosi nazwę
# Osława") — declining made-up Polish names automatically would produce
# ungrammatical training text.
COPY_DRILL_COUNT = 9000

_SYL_A = ["Rud", "Bąk", "Turz", "Osław", "Skaw", "Wiel", "Grod", "Miel", "Choc",
          "Zator", "Bran", "Kośc", "Lubi", "Nadz", "Pilcz", "Sier", "Trzeb", "Wąs"]
_SYL_B = ["niki", "owo", "ysko", "ica", "any", "ków", "no", "in", "awa", "ele",
          "cze", "iny", "ory", "usz", "yca", "ejno"]
_FIRST = ["Zofia", "Marek", "Halina", "Tomasz", "Anna", "Jerzy", "Irena", "Paweł",
          "Krystyna", "Andrzej", "Wanda", "Stefan", "Barbara", "Michał", "Ewa",
          "Roman", "Danuta", "Piotr", "Grażyna", "Karol"]
_LAST = ["Malczyk", "Ostrowski", "Dąbek", "Wielgus", "Reut", "Warkosz", "Sadlik",
         "Bęben", "Krupa", "Nowosad", "Ziętek", "Faron", "Mruk", "Cichoń",
         "Pluta", "Zawada", "Gil", "Hajduk", "Szewc", "Baran"]
_REGION = ["podkarpackim", "lubuskim", "opolskim", "podlaskim", "świętokrzyskim",
           "warmińsko-mazurskim", "kujawsko-pomorskim", "dolnośląskim"]
_PART = ["południowo-wschodniej", "północno-zachodniej", "środkowej",
         "południowo-zachodniej", "północno-wschodniej", "wschodniej", "zachodniej"]
_CRAFT = ["sukna", "ceramiki", "wyrobów kowalskich", "papieru czerpanego",
          "mebli giętych", "szkła", "wikliny", "narzędzi rolniczych"]
_FIELD = ["chemia nieorganiczna", "botanika", "geologia", "matematyka stosowana",
          "fizyka ciała stałego", "historia sztuki", "językoznawstwo", "astronomia"]


def _name(rng):
    return _SYL_A[int(rng.integers(len(_SYL_A)))] + _SYL_B[int(rng.integers(len(_SYL_B)))]


def _person(rng):
    return (f"{_FIRST[int(rng.integers(len(_FIRST)))]} "
            f"{_LAST[int(rng.integers(len(_LAST)))]}")


def _pick(rng, pool):
    return pool[int(rng.integers(len(pool)))]


def _year(value, lo=1100, hi=2025):
    """Keep derived years inside plausible history.

    Offsets added to a base year run off the end otherwise — a first draft
    produced "Prezesem zarządu jest od 2032 roku". The model would happily
    copy that, but nonsense prose is still nonsense to train on."""
    return int(max(lo, min(hi, value)))


def _town(rng):
    """A town described by nine facts, six of them numbers close enough in kind
    that only the right sentence answers the right question."""
    town, river = _name(rng), _name(rng)
    rights = int(rng.integers(1200, 1600))
    first_note = rights - int(rng.integers(20, 140))
    fire = int(rng.integers(1601, 1799))
    pop = int(rng.integers(1500, 90000))
    since = int(rng.integers(1990, 2024))
    museum = int(rng.integers(1900, 1995))
    road = int(rng.integers(100, 999))
    church_from = int(rng.integers(1400, 1550))
    church_to = church_from + int(rng.integers(4, 40))
    mayor, region, part, craft = _person(rng), _pick(rng, _REGION), _pick(rng, _PART), _pick(rng, _CRAFT)
    ctx = (
        f"{town} to miasto w województwie {region}, położone w {part} części kraju. "
        f"Rzeka przepływająca przez miasto nosi nazwę {river}. "
        f"Prawa miejskie {town} otrzymało w {rights} roku, choć pierwsza wzmianka o osadzie "
        f"pochodzi z {first_note} roku. W XVI wieku miasto słynęło z wyrobu {craft}, "
        f"a w {fire} roku strawił je pożar. Liczba mieszkańców wynosi {pop}. "
        f"Burmistrzem miasta jest od {since} roku {mayor}. "
        f"Muzeum regionalne otwarto w {museum} roku, a kościół parafialny wznoszono "
        f"w latach {church_from}-{church_to}. Przez miasto przebiega droga wojewódzka "
        f"numer {road}."
    )
    qa = [
        (f"W którym roku {town} otrzymało prawa miejskie?", str(rights)),
        ("Z którego roku pochodzi pierwsza wzmianka o osadzie?", str(first_note)),
        ("W którym roku miasto strawił pożar?", str(fire)),
        ("Ile wynosi liczba mieszkańców?", str(pop)),
        ("Kto jest burmistrzem miasta?", mayor),
        ("Od którego roku pełni funkcję burmistrz?", str(since)),
        ("Jak nazywa się rzeka przepływająca przez miasto?", river),
        ("W którym roku otwarto muzeum regionalne?", str(museum)),
        ("Jaki numer ma droga wojewódzka przebiegająca przez miasto?", str(road)),
        ("W jakim województwie leży miasto?", region),
        ("W której części kraju leży miasto?", part),
        ("Z wyrobu czego słynęło miasto w XVI wieku?", craft),
    ]
    return ctx, qa


def _scholar(rng):
    person = _person(rng)
    born = int(rng.integers(1860, 1940))
    died = _year(born + int(rng.integers(45, 92)))
    studies = born + int(rng.integers(20, 30))
    habil = studies + int(rng.integers(5, 20))
    head_from = habil + int(rng.integers(2, 15))
    head_to = _year(head_from + int(rng.integers(5, 30)))
    phds = int(rng.integers(3, 120))
    prize = _year(head_from + int(rng.integers(1, 20)))
    papers = int(rng.integers(20, 400))
    city_born, city_work = _name(rng), _name(rng)
    field = _pick(rng, _FIELD)
    ctx = (
        f"{person} to polska uczona, profesor uniwersytetu. Miastem urodzenia jest "
        f"{city_born}, a rokiem urodzenia {born}. Rokiem śmierci jest {died}. "
        f"Studia ukończyła w {studies} roku, a habilitację obroniła w {habil} roku. "
        f"Dziedzina, którą się zajmowała, to {field}. Katedrą kierowała od "
        f"{head_from} do {head_to} roku, pracując w mieście, które nosi nazwę {city_work}. "
        f"Liczba wypromowanych doktorantów wynosi {phds}. Nagrodę państwową otrzymała "
        f"w {prize} roku. Liczba opublikowanych prac wynosi {papers}."
    )
    qa = [
        ("W którym roku urodziła się uczona?", str(born)),
        ("W którym roku zmarła uczona?", str(died)),
        ("W którym roku ukończyła studia?", str(studies)),
        ("W którym roku obroniła habilitację?", str(habil)),
        ("Od którego roku kierowała katedrą?", str(head_from)),
        ("Do którego roku kierowała katedrą?", str(head_to)),
        ("Ilu doktorantów wypromowała?", str(phds)),
        ("W którym roku otrzymała nagrodę państwową?", str(prize)),
        ("Ile prac opublikowała?", str(papers)),
        ("W jakim mieście się urodziła?", city_born),
        ("W jakim mieście pracowała?", city_work),
        ("Jaką dziedziną się zajmowała?", field),
    ]
    return ctx, qa


def _company(rng):
    firm, town = _name(rng), _name(rng)
    founded = int(rng.integers(1900, 2020))
    staff = int(rng.integers(8, 5000))
    output = int(rng.integers(200, 99000))
    plants = int(rng.integers(2, 40))
    markets = int(rng.integers(2, 60))
    founder, boss = _person(rng), _person(rng)
    since = _year(founded + int(rng.integers(1, 40)))
    patents = int(rng.integers(1, 300))
    branch = _name(rng)
    branch_year = _year(founded + int(rng.integers(2, 60)))
    revenue = int(rng.integers(3, 900))
    hall = int(rng.integers(400, 40000))
    region = _pick(rng, _REGION)
    ctx = (
        f"Firma {firm} powstała w {founded} roku. Założycielem jest {founder}. "
        f"Siedziba mieści się w mieście, które nosi nazwę {town}, w województwie {region}. "
        f"Liczba zatrudnionych wynosi {staff}. Roczna produkcja wynosi {output} sztuk. "
        f"Liczba zakładów produkcyjnych wynosi {plants}, a liczba rynków zagranicznych, "
        f"na które firma eksportuje, wynosi {markets}. "
        f"Prezesem zarządu jest od {since} roku {boss}. "
        f"Liczba zgłoszonych patentów wynosi {patents}. "
        f"Oddział zagraniczny, który nosi nazwę {branch}, uruchomiono w {branch_year} roku. "
        f"Roczny przychód wynosi {revenue} milionów złotych, a powierzchnia hali "
        f"produkcyjnej wynosi {hall} metrów kwadratowych."
    )
    qa = [
        (f"W którym roku powstała firma {firm}?", str(founded)),
        ("Kto założył firmę?", founder),
        ("Kto jest prezesem zarządu?", boss),
        ("Od którego roku pełni funkcję prezes?", str(since)),
        ("Ile osób zatrudnia firma?", str(staff)),
        ("Ile wynosi roczna produkcja?", str(output)),
        ("Ile zakładów produkcyjnych ma firma?", str(plants)),
        ("Na ile rynków zagranicznych firma eksportuje?", str(markets)),
        ("W jakim mieście mieści się siedziba firmy?", town),
        ("W jakim województwie mieści się siedziba firmy?", region),
        ("Ile patentów zgłosiła firma?", str(patents)),
        ("Jak nazywa się oddział zagraniczny?", branch),
        ("W którym roku uruchomiono oddział zagraniczny?", str(branch_year)),
        ("Ile wynosi roczny przychód?", str(revenue)),
        ("Ile wynosi powierzchnia hali produkcyjnej?", str(hall)),
    ]
    return ctx, qa


COPY_DRILL_BUILDERS = [_town, _scholar, _company]


def copy_drill_examples(count=COPY_DRILL_COUNT, seed=2):
    """Long contexts packed with same-type distractors, one question each.

    One question per context rather than all of them, so the model never sees
    the same paragraph twice in a row and cannot start answering from position
    alone. Which question is asked is drawn independently, so across the set
    every slot — early, middle and late in the paragraph — gets asked about.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        build = COPY_DRILL_BUILDERS[int(rng.integers(len(COPY_DRILL_BUILDERS)))]
        ctx, qa = build(rng)
        q, a = qa[int(rng.integers(len(qa)))]
        out.append((ctx, q, a))
    return out


def list_examples(seed=1):
    """Synthetic "list N things" instructions with correct, varied answers.

    Each draw picks different items, so what repeats is the *format* — an
    instruction asking for N of something, answered by exactly N distinct
    members of that category — rather than any particular answer. That is the
    part the model is missing; it already knows what a fruit is.
    """
    rng = np.random.default_rng(seed)
    out = []
    for name, items in LIST_CATEGORIES:
        for _ in range(LIST_VARIANTS_PER_CATEGORY):
            n = int(rng.choice([2, 3, 4]))
            n = min(n, len(items))
            picked = [items[i] for i in rng.permutation(len(items))[:n]]
            phrasing = LIST_PHRASINGS[int(rng.integers(len(LIST_PHRASINGS)))]
            q = phrasing.format(c=LIST_COUNT_WORDS[n], k=name)
            a = " ".join(f"{i + 1}) {v}" for i, v in enumerate(picked))
            out.append((q, a))
    return out


# Confirmed live, 2026-07-25: even with the right Wikipedia/vault snippet
# handed to the model as plain text right before the question (see
# runtime/rag.py), the answer still ignored it and confabulated something
# unrelated. The identity fix worked because it's a closed, memorisable set
# repeated many times; grounding-in-context is a *skill* (read the preceding
# text, then answer from it), and nothing in either Alpaca corpus ever
# demonstrates that skill — SFT so far only ever taught "answer the
# question", never "answer the question using the text above it". Unlike the
# identity examples, this needs breadth (many different contexts), not
# repetition of a few — the model needs to generalise the *habit*, not
# memorise specific facts.
CONTEXT_QA_REPO = "clarin-pl/poquad"
# Measured 2026-07-27 with bench/chat_eval.py: context_grounding scored 20%
# (1/5, and that one answer was "Anna Reuttin" for "Anna Reut"). Asked the
# build year of a castle the context says was built in 1417, the model
# answered 1418; asked where a company sits when the context says Gdynia, it
# said Gdańsk. It had learned the *shape* of a grounded answer — a year, a
# count, a city — while ignoring the text entirely.
#
# The cause is signal, not capability. Loss is computed on replies only, and
# 1,200 context examples with ~15-token replies contribute ~18k trained
# tokens against ~9.5M — 0.2% of everything the model learns from. Grounding
# is a skill, so it needs breadth: the train split has 46k answerable QAs
# against the 5.7k in dev, and was simply never used.
#
# dev is deliberately left untouched so it stays a genuine holdout for
# measuring grounding on data the model has never seen.
CONTEXT_QA_FILE = "poquad-train.json"  # 46,187 answerable QAs
CONTEXT_QA_SAMPLE = 34000

# Round A took grounding from 20% to 40%, and the split in what it fixed is
# sharp: names started copying correctly ("Anna Reut", "w Gdyni" — both wrong
# before, both right after) while numbers did not (context says 1417, model
# says 1418; context says cztery wieże, model says dwa; context says 42 km,
# model says trzy). The copying machinery clearly works — it is the model's
# own prior over plausible years and counts beating the text in front of it,
# which is a training-signal balance, not a missing capability.
#
# Only 17.3% of PoQuAD answers contain a digit, so numeric copying got a
# fraction of an already-small share. Oversampling numeric answers to ~45%
# targets exactly the failure the benchmark found. They are drawn with
# repetition (~1.9x over 7,981 distinct rows), which is safe here because each
# one carries a different number in a different context — the model has to
# learn "read the digits from the text", not memorise any single fact.
CONTEXT_QA_NUMERIC_FRAC = 0.45


def load_context_qa_examples(sample_size=CONTEXT_QA_SAMPLE, seed=0):
    """PoQuAD ships as raw SQuAD-format JSON with a loader script the
    `datasets` library no longer supports running ("Dataset scripts are no
    longer supported") — download and parse the JSON directly instead of
    going through `load_dataset`."""
    path = hf_hub_download(CONTEXT_QA_REPO, CONTEXT_QA_FILE, repo_type="dataset",
                            token=read_token())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa["is_impossible"] or not qa["answers"]:
                    continue
                ans = qa["answers"][0]
                # generative_answer reads like a real reply ("Warszawa jest
                # stolicą Polski"); the raw extracted span ("Polski") reads
                # like a search-engine result. We want the former — this is
                # meant to teach conversational grounding, not extraction.
                answer = ans.get("generative_answer") or ans.get("text")
                if not answer:
                    continue
                triples.append((context, qa["question"], answer))

    numeric = [t for t in triples if re.search(r"\d", t[2])]
    other = [t for t in triples if not re.search(r"\d", t[2])]

    rng = np.random.default_rng(seed)
    n_numeric = int(sample_size * CONTEXT_QA_NUMERIC_FRAC)
    n_other = sample_size - n_numeric

    def draw(pool, n):
        if not pool:
            return []
        # Whole pool first, then top up with repeats — sampling with
        # replacement throughout would leave some rows unused while repeating
        # others three times over.
        out = []
        while len(out) < n:
            take = min(n - len(out), len(pool))
            idx = rng.permutation(len(pool))[:take]
            out.extend(pool[i] for i in idx)
        return out

    picked = draw(numeric, n_numeric) + draw(other, n_other)
    print(f"context-QA pool: {len(numeric):,} numeric / {len(other):,} other "
          f"-> drawing {n_numeric:,} / {n_other:,}", flush=True)
    return [picked[i] for i in rng.permutation(len(picked))]


def format_context_example(context, question, answer, max_context_chars=800):
    """Same (prompt, reply) shape as format_example, with the source text
    wrapped in the tokenizer's reserved <|context|> token in front of the
    <|user|> turn. That token was reserved before pretraining started but
    never actually used in a single real training example until now — this
    IS the training pass that finally gives it a meaning, so use it properly
    rather than perpetuating the plain-text workaround runtime/rag.py used
    only because nothing had taught the model this token yet. Once this SFT
    pass lands, update rag.py's injection to match (wrap retrieved text in
    <|context|>...) instead of bare prepended prose."""
    context, question, answer = clean(context), clean(question), clean(answer)
    if len(context) < 20 or len(question) < 4 or len(answer) < 2:
        return None
    context = context[:max_context_chars]
    return f"{CTX}\n{context}\n\n{U}\n{question}\n{A}\n", answer + EOT


def read_token(env=Path(".env")):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def clean(s) -> str:
    """Undo the damage machine translation left in these corpora.

    Several rows arrive with their quoting baked into the value rather than
    around it — the field literally contains `'Oceń to zdanie'`. Left alone the
    model learns to sprinkle stray quotes through its answers.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    # strip one layer of wrapping quotes, repeatedly
    while len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.replace("\\n", "\n").replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def format_example(instruction, inp, output):
    """One conversation turn, split into (prompt, reply) or None.

    The two halves are returned separately and tokenised separately. Slicing a
    single encoded string by character offset would be guessing: BPE merges can
    straddle any boundary, so the token count of a prefix is not guaranteed to
    equal the length of that prefix inside the whole. Here the split point sits
    immediately after a special token, which the tokenizer never merges across,
    so encoding the halves apart is both safe and exact.
    """
    instruction, inp, output = clean(instruction), clean(inp), clean(output)
    if len(instruction) < 4 or len(output) < 2:
        return None
    user = f"{instruction}\n{inp}" if inp else instruction
    return f"{U}\n{user}\n{A}\n", output + EOT


def build(tokenizer_path: Path, out_prefix: Path, max_len: int):
    tok = Tokenizer.from_file(str(tokenizer_path))
    for t in (U, A, EOT, CTX):
        assert tok.token_to_id(t) is not None, f"{t} missing from tokenizer"

    token_buf, mask_buf = [], []
    kept = dropped = 0
    seen = set()

    for name in SOURCES:
        ds = load_dataset(name, split="train", token=read_token())
        print(f"{name}: {len(ds):,} rows", flush=True)
        for row in ds:
            made = format_example(row.get("instruction"), row.get("input"),
                                  row.get("output"))
            if made is None:
                dropped += 1
                continue
            prompt, reply = made

            # Exact-duplicate drop: the two corpora overlap heavily, and a
            # small model will happily memorise anything it sees twice.
            key = hash(prompt + reply)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)

            p_ids = tok.encode(prompt).ids
            r_ids = tok.encode(reply).ids
            ids = p_ids + r_ids
            if len(ids) > max_len:
                dropped += 1
                continue
            # Score the reply only. Training on the user's turn would teach the
            # model to invent questions — the habit we are here to remove.
            mask = np.zeros(len(ids), dtype=np.uint8)
            mask[len(p_ids):] = 1

            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            kept += 1
            if kept % 10000 == 0:
                print(f"  {kept:,} kept", flush=True)

    # Home Assistant commands. Generated separately by data/build_home_sft.py
    # against the live ha-rooms.json, because the room list belongs to Gzowo AI
    # and duplicating it here would let the two drift — the model would keep
    # calling a room after Jurek renamed it.
    home_path = Path("data/home_sft.jsonl")
    home_kept = 0
    if home_path.exists():
        for line in home_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            made = format_example(row["instruction"], None, row["output"])
            if made is None:
                continue
            prompt, reply = made
            p_ids = tok.encode(prompt).ids
            r_ids = tok.encode(reply).ids
            ids = p_ids + r_ids
            if len(ids) > max_len:
                continue
            mask = np.zeros(len(ids), dtype=np.uint8)
            mask[len(p_ids):] = 1
            for _ in range(row.get("repeats", 6)):
                token_buf.append(np.asarray(ids, dtype=np.uint16))
                mask_buf.append(mask)
                home_kept += 1
        kept += home_kept
        print(f"home assistant: {home_kept:,} kept", flush=True)
    else:
        print(f"UWAGA: brak {home_path} — model nie będzie umiał sterować domem "
              f"(uruchom data/build_home_sft.py)", flush=True)

    drill_kept = drill_dropped = 0
    for context, question, answer in copy_drill_examples():
        made = format_context_example(context, question, answer)
        if made is None:
            drill_dropped += 1
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            drill_dropped += 1
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        token_buf.append(np.asarray(ids, dtype=np.uint16))
        mask_buf.append(mask)
        drill_kept += 1
    kept += drill_kept
    print(f"copy drills: {drill_kept:,} kept, {drill_dropped:,} dropped", flush=True)

    list_kept = 0
    for instruction, output in list_examples():
        made = format_example(instruction, None, output)
        if made is None:
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        for _ in range(LIST_REPEATS):
            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            list_kept += 1
    kept += list_kept
    print(f"list examples: {len(LIST_CATEGORIES)} categories x"
          f"{LIST_VARIANTS_PER_CATEGORY} variants x{LIST_REPEATS} = {list_kept:,} kept",
          flush=True)

    # Identity examples bypass the exact-duplicate filter above on purpose —
    # deliberate repetition is the whole point here, not something to dedup
    # away like an accidental corpus overlap.
    identity_kept = 0
    identity_written = 0
    for instruction, output in identity_pairs():
        identity_written += 1
        made = format_example(instruction, None, output)
        if made is None:
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        for _ in range(IDENTITY_REPEATS):
            token_buf.append(np.asarray(ids, dtype=np.uint16))
            mask_buf.append(mask)
            identity_kept += 1
    kept += identity_kept
    print(f"identity examples: {len(IDENTITY_EXAMPLES)} written "
          f"({identity_written} incl. accent-free variants) x{IDENTITY_REPEATS} "
          f"= {identity_kept:,} kept", flush=True)

    # Context-grounded QA: teaches the *habit* of reading the <|context|>
    # block and answering from it, which nothing else in this build ever
    # demonstrates. Breadth matters here, not repetition — sampled once each
    # from PoQuAD rather than repeated like the identity examples, since the
    # goal is generalising a skill across many different texts, not
    # memorising a fixed handful of facts.
    context_kept = context_dropped = 0
    for context, question, answer in load_context_qa_examples():
        made = format_context_example(context, question, answer)
        if made is None:
            context_dropped += 1
            continue
        prompt, reply = made
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(reply).ids
        ids = p_ids + r_ids
        if len(ids) > max_len:
            context_dropped += 1
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        mask[len(p_ids):] = 1
        token_buf.append(np.asarray(ids, dtype=np.uint16))
        mask_buf.append(mask)
        context_kept += 1
    kept += context_kept
    print(f"context-QA examples: {context_kept:,} kept, {context_dropped:,} dropped",
          flush=True)

    # finetune.py's val split is just "the last val_frac of examples by
    # position here" (SFTData: offsets[:cut] / offsets[cut:]) — with identity
    # examples appended after the whole main corpus, every single one of them
    # landed past the cut and the model was trained on literally zero of
    # them, only ever evaluated against them. Confirmed by direct
    # calculation before this fix: 89,495 main + 960 identity, 2% val cut
    # sits at index 88,645 — entirely before the identity block starts at
    # 89,495. Shuffling before the split fixes this generally, not just for
    # the current counts.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(token_buf))
    token_buf = [token_buf[i] for i in order]
    mask_buf = [mask_buf[i] for i in order]

    tokens = np.concatenate(token_buf)
    masks = np.concatenate(mask_buf)
    assert len(tokens) == len(masks)

    # Where each example begins in the concatenated stream. Training samples
    # windows from these offsets rather than from anywhere, so a window never
    # opens midway through a reply — the model would otherwise be asked to
    # continue an answer whose question it never saw.
    offsets = np.zeros(len(token_buf), dtype=np.int64)
    np.cumsum([len(a) for a in token_buf[:-1]], out=offsets[1:])

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(f"{out_prefix}_tokens.bin")
    masks.tofile(f"{out_prefix}_mask.bin")
    offsets.tofile(f"{out_prefix}_offsets.bin")

    print(f"\nkept {kept:,} examples, dropped {dropped:,}")
    print(f"{len(tokens)/1e6:.1f}M tokens, {masks.mean()*100:.1f}% trained on")
    print(f"-> {out_prefix}_tokens.bin / _mask.bin")

    print("\n--- sample (special tokens shown) ---")
    print(repr(tok.decode([int(t) for t in token_buf[0]],
                          skip_special_tokens=False))[:420])
    print(f"(loss on {int(mask_buf[0].sum())} of {mask_buf[0].size} tokens)")

    # Alignment audit across the whole set, not one lucky example: the token
    # immediately before the first scored position must be the newline that
    # follows <|assistant|>. If this drifts, the model is trained to predict
    # the wrong half of the conversation and nothing else will reveal it.
    a_id, bad = tok.token_to_id(A), 0
    for t_arr, m_arr in zip(token_buf[:5000], mask_buf[:5000]):
        first = int(np.argmax(m_arr))
        if first < 2 or int(t_arr[first - 2]) != a_id:
            bad += 1
    print(f"alignment check: {bad} of {min(5000, len(token_buf))} examples misaligned")
    assert bad == 0, "loss mask does not start right after <|assistant|>"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    ap.add_argument("--out-prefix", type=Path, default=Path("data/pl_sft"))
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()
    build(args.tokenizer, args.out_prefix, args.max_len)
    sys.stdout.flush()
    os._exit(0)   # see fetch_corpus.py — datasets can abort at interpreter exit
