# MiniG

Polski model językowy ~178M parametrów, trenowany od zera. **Poziom 2** drabiny
rodziny: MicroG 110M → **MiniG** → CoreG 500M → MegaG 1B+.

Następca [MicroG](https://github.com/JerzySukiennik/microg), zbudowany po to, by
naprawić jedną konkretną wadę, którą w MicroG zdiagnozowano pomiarem — i przy
okazji przejąć sterowanie domem.

## Po co nowy model, skoro MicroG działa

MicroG myli liczby: zapytany o rok, który w podanym tekście brzmi **1417**,
odpowiada **1418**; z **8 431** robi **8 321**. Przyczyna nie leży w ilości
danych — potrojenie przykładów numerycznych podniosło skuteczność o 8 punktów i
tam się zatrzymało. Leży w tokenizerze: słownik MicroG zawiera **241 tokenów
wielocyfrowych**, w tym całe lata (`1998`, `2011`) jako pojedyncze symbole o
niemal identycznych osadzeniach. Model kopiuje pierwszy token i zgaduje resztę.

Tokenizera nie da się wymienić bez trenowania od nowa — stąd nowy model.
MiniG rozbija **każdą cyfrę osobno**, tak jak Llama i Mistral, więc przepisanie
liczby to cztery łatwe kopie zamiast trafienia w rzadki token.

## Czego się spodziewać, a czego nie

**~9% lepszy od MicroG na znak.** Tego nie widać w rozmowie i lepiej wiedzieć to
przed spaleniem dwóch tygodni quoty niż po. Perplexity nie jest porównywalna
między różnymi tokenizerami, więc większe liczby krążące wcześniej były
artefaktem — uczciwą miarą jest bit na znak.

Realnie zauważalne będzie to, co nie zależy od rozmiaru:

- **liczby przepisywane bez psucia** (nowy tokenizer),
- **sterowanie Home Assistant** komendami po polsku,
- cały dorobek dostrajania z rund A–D MicroG: grounding, listy, tożsamość
  odporna na brak polskich znaków.

## Konfiguracja i skąd się wzięła

| | wartość | uzasadnienie |
|---|---|---|
| 20 warstw × 768 | 178,5M | ściana pamięci T4 leży między 204M a 282M — powyżej micro-batch spada z 8 na 4 i przepustowość leci o 28% |
| słownik 48 000 | cyfry pojedynczo | zmierzone: 4,8% krótsze sekwencje niż 32k przy tym samym rozbiciu cyfr |
| kontekst 1024 | — | 2048 kosztuje 20–25% przepustowości, co przy tym budżecie zjada niemal całą poprawę; RoPE pozwala rozszerzyć później |
| 3,7B tokenów | 20,7 na parametr | punkt optymalny Chinchilli dla 60 h na T4×2 |

`n_embd = 768` jest **identyczne z MicroG** i to nie przypadek: dzięki temu jego
dwanaście wytrenowanych bloków wchodzi do MiniG bez zmian.

## Ciepły start

MiniG nie zaczyna od szumu. `model/warm_start.py` przenosi:

- **12 bloków MicroG** → warstwy 0–11, bez zmian;
- **8 nowych warstw** → kopie wytrenowanych bloków z **wyzerowanymi** wyjściami
  do strumienia rezydualnego, więc w kroku zerowym model liczy dokładnie to, co
  liczył MicroG, a nowa pojemność narasta od identyczności (metoda z SOLAR i
  LLaMA-Pro);
- **osadzenia** — 30 688 z 48 000 tokenów (63,9%) to te same ciągi znaków co w
  słowniku MicroG i kopiują się 1:1; pozostałe 17 312 to średnia podtokenów ze
  starego tokenizera. **Zero osadzeń losowych.**

Ciepły start nie jest założeniem: plan zakłada 2–3 h na porównanie A/B ze
startem od zera, zanim zaangażujemy resztę quoty.

## Jak to uruchomić

```bash
python data/train_tokenizer.py korpus.txt --vocab-size 48000 --out data/tokenizer.json
python data/build_home_sft.py          # komendy HA z żywego ha-rooms.json
python model/warm_start.py             # przeszczep wag z MicroG
python bench/smoke.py                  # sprawdza wszystko powyżej w minutę
```

Trening idzie przez Kaggle (T4×2), dwoma kernelami: `kaggle/01-prep.py` buduje
korpus na CPU, `kaggle/02-train.py` trenuje i wznawia się między sesjami.
Sesje giną bez ostrzeżenia, więc wagi, momenty optymalizatora, licznik kroków i
stan generatora losowego zapisują się razem.

## Sterowanie domem

Gzowo AI ma już całą instalację: `control_room(room, service, value)`, mapę encji
w `ha-rooms.json`, odczyt stanu i most z weryfikacją pochodzenia żądań. MiniG
uczy się mapować polskie zdanie na to wywołanie — **1 031 par**, pokoje
sprawdzone wobec żywej mapy, każda komenda również w wariancie bez ogonków.

Świadomie **bez fallbacku** do większego modelu, co zmienia wymagania: **19%
przykładów to odmowy**. Nieznane miejsce ma dostać uczciwe „nie znam takiego
miejsca", a nie zgadnięty pokój — bo zgadnięty pokój gasi światło w prawdziwym
domu. Aplikacja i tak musi walidować wyjście wobec rejestru encji; trening
zmniejsza częstość błędów, nie czyni wyjścia godnym zaufania.

## Ograniczenia

Nietypowe sformułowania, komendy wieloetapowe („zgaś wszystko oprócz kuchni") i
odwołania do kontekstu („a teraz to samo w salonie") będą poza zasięgiem.
Wiedza o świecie pozostaje na poziomie modelu tej wielkości — to nie jest
asystent ogólnego przeznaczenia i nie ma nim być.
