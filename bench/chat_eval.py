"""Chat-behaviour benchmark for the instruction-tuned (SFT) model.

The rest of bench/ measures the *base* model's language modelling — perplexity,
inflection, speed. None of it can see whether the chat model knows its own name,
follows an instruction, stops when it is done, or leaks identity into unrelated
answers. Those are exactly the things every SFT round has changed so far, and
until now each round was judged by talking to the model and forming an
impression. This turns that impression into numbers so two checkpoints can be
compared honestly.

Everything is decoded greedily (temperature 0.1, top_k 1). That is deliberate:
at temperature 0.7 a bad mode is often hidden by sampling luck — the identity
leak introduced on 2026-07-27 measured 0/6 under sampling and 1/12 under greedy
decoding, and the greedy number was the true one. A regression you can only see
sometimes is still a regression.

Usage:
    python bench/chat_eval.py                        # checkpoints/sft/best.pt
    python bench/chat_eval.py path/to/best.pt        # a specific checkpoint
    python bench/chat_eval.py a.pt b.pt              # compare two, side by side
"""

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, load_tokenizer, score_sentence  # noqa: E402

U, A, EOT, CTX = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|context|>"
REPO = Path(__file__).resolve().parents[1]
# Generation stops at <|endoftext|>, so this budget only costs time on answers
# that fail to terminate — which is exactly what stops_cleanly measures. It was
# 48, which turned out to measure the budget rather than the model: "Czym jest
# Warszawa?" writes a complete 43-word answer and emits EOT just past the old
# cutoff, and was being scored as a failure to stop.
MAX_NEW = 128

# --------------------------------------------------------------- test sets --

IDENTITY_DIACRITICS = [
    "Jak masz na imię?", "Jak się nazywasz?", "Kim jesteś?",
    "Czy masz imię?", "Przedstaw się.", "Jak brzmi twoja nazwa?",
]
IDENTITY_PLAIN = [
    "Jak masz na imie?", "Jak sie nazywasz?", "Kim jestes?",
    "Czy masz imie?", "Przedstaw sie.", "Czy jestes Claude?",
]
# Unrelated prompts, weighted toward shapes closest to the identity examples
# ("Opowiedz ... o X" mirrors the trained "Opowiedz coś o sobie."), because
# those are where over-training identity actually bleeds through.
UNRELATED = [
    "Opowiedz krótko o psach.", "Opowiedz coś o Krakowie.",
    "Opowiedz o drugiej wojnie światowej.", "Przedstaw najważniejsze fakty o Księżycu.",
    "Czym jest Warszawa?", "Co to jest fotosynteza?", "Jak zrobić naleśniki?",
    "Wymień trzy planety.", "Jaki jest największy ocean?",
    "Kto napisał Pana Tadeusza?", "Wyjaśnij czym jest grawitacja.",
    "Podaj przepis na zupę pomidorową.",
]
IDENTITY_MARKERS = ("G-Micro", "modelem językowym", "model językowy", "110 milionów")

# (context, question, strings that a grounded answer should contain).
# The point is not world knowledge — it is whether the model reads the text in
# front of it. Facts are deliberately checkable only from the context.
#
# Sized deliberately: with five cases one answer moved the score by 20% and
# two rounds of training could not be told apart from noise. Eighteen cases,
# half of them numeric, make a single flip worth ~6% and let the numeric and
# name-copying halves be scored separately — that split is what showed round A
# fixed names while leaving numbers untouched.
_CASTLE = ("Zamek w Bąkowie został zbudowany w 1417 roku przez rycerza Mikołaja "
           "Warkosza. Zamek ma cztery wieże i fosę o głębokości sześciu metrów.")
_NORTEM = ("Firma Nortem wyprodukowała w zeszłym roku 12 400 rowerów elektrycznych. "
           "Jej siedziba mieści się w Gdyni, a założycielem jest Anna Reut.")
_RIVER = ("Rzeka Skawica ma 42 kilometry długości i wpada do Skawy w miejscowości "
          "Białka. Nad rzeką leżą trzy młyny wodne.")
_SCHOOL = ("Szkoła w Turzysku powstała w 1963 roku. Uczy się w niej 214 uczniów, "
           "a dyrektorem jest Marek Ostrowski. Budynek ma trzy piętra.")
_LAB = ("Laboratorium Wega zatrudnia 87 osób i mieści się w Toruniu. Kieruje nim "
        "profesor Halina Dąbek. Powstało w 2011 roku.")
_MUSEUM = ("Muzeum Kolei Wąskotorowej w Rudnie otwarto w 1998 roku. Zgromadzono w nim "
           "26 parowozów, a kuratorem zbiorów jest Tomasz Wielgus.")

# Real retrieved contexts are Wikipedia-length paragraphs, not two sentences.
# Confirmed live 2026-07-27 with RAG working correctly: handed the actual
# "Kraków" article the model answered "w północnej Polsce", and handed the
# "Fryderyk Chopin" article it put his birth in 1832. The short cases above
# score far better than that, so they were measuring an easier task than the
# app performs — the same benchmark-versus-reality gap that has bitten this
# suite repeatedly. These cases carry the distractors, dates and clause
# density of the real thing.
_LONG_TOWN = (
    "Rudniki Wielkie to miasto w województwie podkarpackim, położone w "
    "południowo-wschodniej części kraju, nad rzeką Osławą. Prawa miejskie "
    "otrzymało w 1382 roku z nadania Władysława Opolczyka, choć pierwsza "
    "wzmianka o osadzie pochodzi z 1274 roku. W XVI wieku miasto słynęło z "
    "wyrobu sukna, a w 1611 roku strawił je pożar, po którym odbudowano "
    "jedynie część zabudowy. Według spisu z 2021 roku Rudniki Wielkie liczyły "
    "8 431 mieszkańców. Burmistrzem miasta jest od 2018 roku Zofia Malczyk. "
    "W mieście działa muzeum tkactwa, otwarte w 1974 roku, oraz zabytkowy "
    "kościół pod wezwaniem świętego Bartłomieja, wzniesiony w latach "
    "1503-1519. Przez Rudniki Wielkie przebiega droga wojewódzka numer 892."
)
_LONG_SCIENTIST = (
    "Helena Wroniecka (ur. 3 marca 1897 w Kaliszu, zm. 12 listopada 1968 w "
    "Poznaniu) – polska chemiczka, profesor Uniwersytetu Poznańskiego. Studia "
    "ukończyła w 1921 roku w Krakowie, gdzie następnie pracowała jako "
    "asystentka. W 1934 roku obroniła habilitację poświęconą związkom "
    "krzemoorganicznym. Podczas okupacji prowadziła tajne nauczanie. Katedrą "
    "chemii nieorganicznej kierowała przez dwadzieścia dwa lata, od 1946 do "
    "1968 roku. Wypromowała 41 doktorantów. Za pracę nad katalizatorami "
    "otrzymała w 1961 roku nagrodę państwową drugiego stopnia. Jej najbardziej "
    "cytowana publikacja ukazała się w 1952 roku."
)

CONTEXT_CASES_LONG = [
    (_LONG_TOWN, "W którym roku Rudniki Wielkie otrzymały prawa miejskie?", ["1382"], "num"),
    (_LONG_TOWN, "Ilu mieszkańców liczyły Rudniki Wielkie w 2021 roku?", ["8 431", "8431"], "num"),
    (_LONG_TOWN, "Kto jest burmistrzem Rudnik Wielkich?", ["Malczyk"], "name"),
    (_LONG_TOWN, "Nad jaką rzeką leżą Rudniki Wielkie?", ["Osław"], "name"),
    (_LONG_TOWN, "W której części kraju leżą Rudniki Wielkie?", ["południow"], "name"),
    (_LONG_SCIENTIST, "W którym roku urodziła się Helena Wroniecka?", ["1897"], "num"),
    (_LONG_SCIENTIST, "Ilu doktorantów wypromowała Helena Wroniecka?", ["41"], "num"),
    (_LONG_SCIENTIST, "W jakim mieście urodziła się Helena Wroniecka?", ["Kalisz"], "name"),
    (_LONG_SCIENTIST, "Czym zajmowała się Helena Wroniecka?", ["chemi"], "name"),
]

CONTEXT_CASES = [
    # numeric answers — the measured weak half
    (_CASTLE, "W którym roku zbudowano zamek w Bąkowie?", ["1417"], "num"),
    (_CASTLE, "Ile wież ma zamek w Bąkowie?", ["cztery", "4"], "num"),
    (_CASTLE, "Jaka jest głębokość fosy?", ["sześciu", "sześć", "6"], "num"),
    (_NORTEM, "Ile rowerów wyprodukowała firma Nortem?", ["12 400", "12400"], "num"),
    (_RIVER, "Ile kilometrów ma rzeka Skawica?", ["42"], "num"),
    (_RIVER, "Ile młynów leży nad rzeką?", ["trzy", "3"], "num"),
    (_SCHOOL, "W którym roku powstała szkoła w Turzysku?", ["1963"], "num"),
    (_SCHOOL, "Ilu uczniów uczy się w szkole?", ["214"], "num"),
    (_SCHOOL, "Ile pięter ma budynek szkoły?", ["trzy", "3"], "num"),
    (_LAB, "Ile osób zatrudnia laboratorium Wega?", ["87"], "num"),
    (_LAB, "W którym roku powstało laboratorium?", ["2011"], "num"),
    (_MUSEUM, "Ile parowozów zgromadzono w muzeum?", ["26"], "num"),
    (_MUSEUM, "W którym roku otwarto muzeum?", ["1998"], "num"),
    # name / place answers — the half round A fixed
    (_NORTEM, "Kto założył firmę Nortem?", ["Reut"], "name"),
    (_NORTEM, "Gdzie mieści się siedziba firmy Nortem?", ["Gdyn"], "name"),
    (_CASTLE, "Kto zbudował zamek w Bąkowie?", ["Warkosz"], "name"),
    (_SCHOOL, "Kto jest dyrektorem szkoły?", ["Ostrowski"], "name"),
    # "w Toruń" is the right city in the wrong case, and scoring it as a miss
    # measured Polish declension rather than whether the model read the text.
    (_LAB, "Gdzie mieści się laboratorium Wega?", ["Toruni", "Toruń", "Torun"], "name"),
    (_MUSEUM, "Kto jest kuratorem zbiorów?", ["Wielgus"], "name"),
    (_RIVER, "Do jakiej rzeki wpada Skawica?", ["Skaw"], "name"),
]

# "List exactly N things" — checks instruction following, and catches the
# "1) Azja 2.) Azja" degenerate repeat seen on 2026-07-27.
#
# (question, how many items, set of acceptable items). The third field exists
# because counting alone scored "Ziemia, Europa, Antarktyda" as a perfect list
# of planets — a moon and a continent. Round B answered "Ziemia, Mars, Wenus"
# and scored no better, so the benchmark was blind to the one thing that
# actually improved. Membership is checked loosely (prefix match, so Polish
# case endings pass) and only when a vocabulary is known.
LIST_CASES = [
    ("Wymień trzy planety.", 3,
     {"merkur", "wenus", "ziemi", "mars", "jowisz", "saturn", "uran", "neptun"}),
    ("Wymień trzy owoce.", 3,
     {"jabłk", "jablk", "grusz", "śliw", "sliw", "banan", "pomarańcz", "pomarancz",
      "truskaw", "malin", "winogron", "brzoskwin", "arbuz", "wiśni", "wisni",
      "czereśni", "czeresni", "ananas", "mandarynk", "cytryn", "borówk", "jagod"}),
    ("Wymień trzy kolory.", 3,
     {"czerwon", "niebiesk", "zielon", "żółt", "zolt", "czarn", "biał", "bial",
      "szar", "brązow", "brazow", "różow", "rozow", "fiolet", "pomarańczow",
      "pomaranczow", "granat", "beżow", "bezow", "turkus"}),
    ("Podaj dwa polskie miasta.", 2,
     {"warszaw", "krak", "łódź", "lodz", "wrocław", "wroclaw", "poznań", "poznan",
      "gdańsk", "gdansk", "szczecin", "bydgoszcz", "lublin", "katowic", "gdyni",
      "toruń", "torun", "radom", "rzeszów", "rzeszow", "olsztyn", "opole"}),
    # Wider than the obvious farm-and-forest set: the model answered "jaguary,
    # bocian, jelenie" and scored 1/3 because a jaguar and a stork were missing
    # from the vocabulary, not from the answer. An incomplete vocabulary makes
    # list_correct a lower bound rather than a measurement.
    ("Wymień trzy zwierzęta.", 3,
     {"pies", "psa", "kot", "koń", "kon", "krow", "świni", "swini", "owc", "kur",
      "lis", "wilk", "niedźwiedź", "niedzwiedz", "zając", "zajac", "sarn", "jeleń",
      "jelen", "słoń", "slon", "tygrys", "lew", "małp", "malp", "mysz", "ryb",
      # Stems, not whole words: Polish drops the fill vowel when it inflects
      # ("wróbel" but "wróbla"), so "wróbl" alone missed the nominative the
      # model actually wrote. Same trap as the Toruń case.
      "jaguar", "bocian", "żyraf", "zyraf", "zebr", "krokodyl", "wielorib",
      "wróbel", "wrobel", "jaskół", "jaskol", "kacz", "gęś", "ges ", "indyk",
      "bażant", "bazant", "czapl", "łab", "lab", "kruk", "wron", "sroka",
      "dzięcio", "dziecio", "kukułk", "kukulk", "sikork", "szpak", "słowik",
      "slowik", "jeż", "jez ", "kret", "chomik", "śwink", "swink", "osioł",
      "osiol", "muł", "wielbłąd", "wielblad", "lam", "alpak", "bizon", "żubr",
      "zubr", "renifer", "antylop", "gazel", "surykatk", "leniwiec", "pingwin",
      "wieloryb", "delfin", "orzeł", "orzel", "sow", "wróbl", "wrobl", "gołąb",
      "golab", "jaszczur", "wąż", "waz", "żółw", "zolw", "królik", "krolik",
      "wiewiór", "wiewior", "borsuk", "dzik", "łoś", "los ", "ryś", "rys ",
      "hipopotam", "nosoroż", "nosoroz", "pantera", "gepard", "kangur", "koal",
      "panda", "foka", "rekin", "motyl", "pszczoł", "pszczol", "mrów", "mrow"}),
    ("Wymień trzy dni tygodnia.", 3,
     {"poniedział", "poniedzial", "wtorek", "środ", "srod", "czwartek", "piątek",
      "piatek", "sobot", "niedziel"}),
    ("Wymień cztery pory roku.", 4,
     {"wiosn", "lat", "jesień", "jesien", "zim"}),
    ("Podaj trzy owoce cytrusowe.", 3, None),
    ("Wymień trzy środki transportu.", 3, None),
    ("Podaj dwa instrumenty muzyczne.", 2, None),
]

# Plain, unambiguous Polish. Lower loss per token = the model finds ordinary
# Polish more plausible. Same sentences every run, so runs are comparable.
PROBES = [
    "Warszawa jest stolicą Polski i największym miastem w kraju.",
    "Fotosynteza to proces, w którym rośliny wytwarzają cukry z dwutlenku węgla.",
    "Wczoraj wieczorem poszedłem do sklepu po chleb i mleko.",
    "Komputer składa się z procesora, pamięci operacyjnej i dysku twardego.",
    "W niedzielę pojechaliśmy nad jezioro, żeby popływać kajakiem.",
]


# ----------------------------------------------------------------- helpers --

def make_asker(model, tok):
    eot_id = tok.encode(EOT).ids[0]

    def ask(prompt_text, max_new=MAX_NEW):
        ids = tok.encode(prompt_text).ids
        out, hit_eot = [], False
        for tid, _ in model.generate(torch.tensor([ids]), max_new_tokens=max_new,
                                     temperature=0.1, top_k=1):
            if tid == eot_id:
                hit_eot = True
                break
            out.append(int(tid))
        return tok.decode(out).strip(), hit_eot

    return ask


def chat_prompt(question, context=None):
    if context:
        return f"{CTX}\n{context}\n\n{U}\n{question}\n{A}\n"
    return f"{U}\n{question}\n{A}\n"


def has_repeat_loop(text, min_len=4):
    """True if some phrase repeats back-to-back — the failure mode
    repetition_penalty was added for, checked here so a regression in it
    cannot pass silently."""
    words = text.split()
    for size in range(1, 6):
        for i in range(len(words) - size * 2 + 1):
            a = words[i:i + size]
            if a == words[i + size:i + size * 2] and len(" ".join(a)) >= min_len:
                return True
    return False


def count_list_items(text):
    """Number of DISTINCT items in a list answer.

    Distinct, because the failure worth catching is three slots filled with the
    same word, not a short list. Items may be numbered ("1) x 2) x"), bulleted,
    or plain comma-separated, and are frequently all on one line, so splitting
    on newlines alone misses most of them — the marker itself has to be the
    delimiter.
    """
    marker = re.compile(r"(?:\d+\s*[.)]|^\s*[-*])\s*", re.M)
    if marker.search(text):
        parts = marker.split(text)
    else:
        parts = re.split(r"[,;\n]", text)
    norm = set()
    for p in parts:
        # Keep the first few words: "Kraków: Miasto, które ma najwięcej…" and
        # "Kraków" are the same item, and a trailing empty marker ("3.)") is not
        # an item at all.
        key = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ ]+", " ", p.lower()).strip()
        key = " ".join(key.split()[:2])
        if len(key) > 1:
            norm.add(key)
    return len(norm)


# ------------------------------------------------------------------- suite --

def evaluate(ckpt_path):
    model, step, best_val = load_model(ckpt_path)
    tok = load_tokenizer()
    ask = make_asker(model, tok)
    res = {"checkpoint": str(ckpt_path), "step": step, "best_val": best_val}
    detail = {}

    def section(name, cases, check):
        hits, rows = 0, []
        for case in cases:
            q = case if isinstance(case, str) else case[0]
            prompt = chat_prompt(q) if isinstance(case, str) else None
            ans, eot = ask(prompt or chat_prompt(q))
            ok = check(case, ans)
            hits += ok
            rows.append({"q": q, "a": ans, "ok": bool(ok), "eot": eot})
        res[name] = hits / len(cases)
        detail[name] = rows
        return rows

    section("identity_diacritics", IDENTITY_DIACRITICS,
            lambda c, a: "G-Micro" in a)
    section("identity_plain", IDENTITY_PLAIN,
            lambda c, a: "G-Micro" in a)
    # Inverted: passing means NOT sounding like an identity answer.
    section("no_identity_leak", UNRELATED,
            lambda c, a: not any(m in a for m in IDENTITY_MARKERS))

    # Context grounding needs its own loop — the prompt carries a context block.
    # Numeric and name cases are scored apart as well as together: round A
    # moved names from wrong to right while leaving every number untouched,
    # and a single blended number hid that completely.
    rows, by_kind = [], {"num": [0, 0], "name": [0, 0]}
    long_hits = 0
    for ctx, q, expect, kind in CONTEXT_CASES + CONTEXT_CASES_LONG:
        ans, eot = ask(chat_prompt(q, context=ctx))
        ok = any(e.lower() in ans.lower() for e in expect)
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
        is_long = len(ctx) > 400
        long_hits += ok and is_long
        rows.append({"q": q, "a": ans, "ok": bool(ok), "expect": expect,
                     "kind": kind, "long": is_long})
    res["grounding_numbers"] = by_kind["num"][0] / by_kind["num"][1]
    res["grounding_names"] = by_kind["name"][0] / by_kind["name"][1]
    res["grounding_long_ctx"] = long_hits / len(CONTEXT_CASES_LONG)
    res["context_grounding"] = sum(r["ok"] for r in rows) / len(rows)
    detail["context_grounding"] = rows

    # Two separate questions about a list answer: does it have the right shape
    # (enough distinct items), and are the items actually what was asked for.
    shape_hits, correct_hits, scored, rows = 0, 0, 0, []
    for q, n, vocab in LIST_CASES:
        ans, eot = ask(chat_prompt(q))
        got = count_list_items(ans)
        shape_ok = got >= n
        shape_hits += shape_ok
        row = {"q": q, "a": ans, "ok": bool(shape_ok), "want": n, "got": got,
               "eot": eot}
        if vocab is not None:
            low = ans.lower()
            found = {v for v in vocab if v in low}
            correct_ok = len(found) >= n
            correct_hits += correct_ok
            scored += 1
            row.update(correct=bool(correct_ok), matched=sorted(found))
        rows.append(row)
    res["list_following"] = shape_hits / len(LIST_CASES)
    res["list_correct"] = correct_hits / scored if scored else float("nan")
    detail["list_following"] = rows

    # Termination and repetition, measured over everything already generated.
    all_rows = [r for k in ("identity_diacritics", "identity_plain",
                            "no_identity_leak", "list_following")
                for r in detail[k]]
    res["stops_cleanly"] = sum(r.get("eot", False) for r in all_rows) / len(all_rows)
    res["no_repeat_loop"] = 1 - sum(has_repeat_loop(r["a"]) for r in all_rows) / len(all_rows)

    tot_lp = tot_n = 0
    for s in PROBES:
        lp, n = score_sentence(model, tok, s)
        tot_lp += lp
        tot_n += n
    res["polish_loss_per_token"] = -tot_lp / tot_n

    del model
    return res, detail


# grounding_numbers / grounding_names are diagnostic breakdowns of
# context_grounding, so they are printed but left out of the average to avoid
# counting the same cases three times.
SCORES = ["identity_diacritics", "identity_plain", "no_identity_leak",
          "context_grounding", "list_following", "list_correct",
          "stops_cleanly", "no_repeat_loop"]
DIAGNOSTIC = ["grounding_numbers", "grounding_names", "grounding_long_ctx"]


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [REPO / "checkpoints" / "sft" / "best.pt"]
    results, details = [], []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"no such checkpoint: {p}")
        print(f"evaluating {p} …", flush=True)
        r, d = evaluate(p)
        results.append(r)
        details.append(d)

    width = max(len(s) for s in SCORES) + 2
    print(f"\n{'metric':<{width}}" + "".join(f"{Path(r['checkpoint']).parent.name:>16}"
                                             for r in results))
    print("-" * (width + 16 * len(results)))
    for s in SCORES:
        print(f"{s:<{width}}" + "".join(f"{r[s] * 100:>15.0f}%" for r in results))
    print(f"{'OVERALL':<{width}}" +
          "".join(f"{sum(r[s] for s in SCORES) / len(SCORES) * 100:>15.1f}%" for r in results))
    print(f"\n{'-- diagnostic (inside context_grounding) --':<{width}}")
    for s in DIAGNOSTIC:
        print(f"{s:<{width}}" + "".join(f"{r[s] * 100:>15.0f}%" for r in results))
    print(f"\n{'polish_loss/token':<{width}}" +
          "".join(f"{r['polish_loss_per_token']:>16.4f}" for r in results) + "   (lower better)")
    print(f"{'sft val_loss':<{width}}" +
          "".join(f"{r['best_val']:>16.4f}" for r in results) +
          "   (NOT comparable across different data)")

    out = REPO / "Niepotrzebne" / "chat_eval_last.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"results": results, "details": details},
                              ensure_ascii=False, indent=2))
    print(f"\nfull answers written to {out}")


if __name__ == "__main__":
    main()
