# What the data looks like

Measured on the challenge files on 2026-09-26. Every number here comes from a script run against `student_resource/dataset`. Nothing is estimated.

## Scale

| split | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

Countries: train is US (60%) and India (40%). Test is India 47%, US 38%, France 15% of Source 1. France has no labels anywhere.

Test carries 2.82 Source-2 records per Source-1 record. Train carries 2.28. Either test entities have more matches, or test has more unmatched records. The labels cannot tell which.

## The shape of the ground truth

- 7,638,365 positive pairs, and 7,638,365 distinct targets. **Every Source-2/3 record belongs to at most one Source-1 entity.** The matching problem is an assignment, not free pairing.
- No pair crosses countries.
- 5.6% of Source-1 entities have no match. The rate is the same in both countries.
- Matches per entity: 1.67 from Source 2 on average (at most 5) and 1.79 from Source 3 (at most 6).
- 26% of Source-2/3 records match nothing. They are drawn from the same generator as the matched ones, so they look like ordinary businesses, often with a near-twin in Source 1.

Those unmatched records, together with the singleton entities, are where precision is won or lost. A Source-1 entity with no true match scores 1.0 for an empty prediction and 0.0 for any match at all.

## How the noise is generated

The records are synthetic, and the noise follows a small set of rules. Mined from 371K Latin-script training pairs:

**Names**

- case changes, doubled spaces, injected accents (`Yóga`, `Frànce`)
- leetspeak typos: `J0hnson`, `F1ores`, `5ervices`, `C0m`
- legal forms swapped, dropped or moved to the front: `LLC` dropped in 31% of pairs that have it, `Inc` in 39%, `Limited` in 46%, `LLC Moncada Learning Center`
- honorifics prepended: `Shri`, `Smt`, `Sri`, `Dr`, `Mr`, `The`
- junk prefixes and suffixes: `--`, `<<`, `(ID: 81649)`, ` - 2935142937`, `#75294`
- descriptor words added or dropped: `Center`, `Services`, `Partners`, `Group`, `Clinic`, `Associates`, each around 30%
- word order shuffled: `[Clinic] Automation Creative`, `LIMITED SHIVAYA SYNTEX`
- domain renderings: `johnsonfreight.com`, `@choiceadvisory`, `#SHANKSNEWHOLD`
- alias constructions, 1.7% of pairs: `Noviorbi DBA Noble & Co`, `Dovadovadrex formerly known as Satterwhite and Massengill PLLC`. The Source-1 name always follows the marker.
- invented trade names with no relation to the Source-1 name (`Veracalo`, `Fluxarc`, `ZEPHZEPH`). About 4.6% of pairs share no name word at all. Only the address links these.

**Indian names in native script.** 23.5% of Indian Source-2 names and 13.2% of Source-3 names are phonetic renderings of the English name in Devanagari, Tamil, Kannada, Telugu, Gujarati or Bengali script. `इंडो प्रोडक्ट्स प्राइवेट लिमिटेड` is `Indo Products Private Limited`. The rendering is word by word, so token positions line up with the Source-1 name. A dictionary learned from those alignments has 1,347 entries and covered every native token in a held-out 20% of pairs. `anyascii` alone gets `imdo prodkts praivet limited`.

**Addresses**

- missing in 3.4% of Source-2/3 records, plus placeholder components (`null`, `NULL`, `<NULL>`, `N/A`)
- components reordered: `GREENSBORO, NC, 19 1/2 STARDUST TRAIL`
- street types abbreviated both ways, and sometimes expanded wrongly (`ST` becomes `SAINT`)
- state as a code (`NC`, `MH`), a full name (`North Carolina`, `Maharashtra`), or native script (`महाराष्ट्र`). The only non-Latin address fragments in the data are 16 Indian state names.
- Indian labels vary: `House No 655`, `DOOR NO 499 HOUSE NO 655`, `H.no 35`, `#19`
- house numbers altered (`230` becomes `1`), units and PO boxes added
- alternate city names: `Bombay`/`Mumbai`, `Poona`/`Pune`, `Buffalo`/`Kenmore`, `Tomah`/`Town Of Byron`

## State is a safe blocking key

When both sides carry a state, US pairs agree 100% of the time. India agrees 98.8%, and the disagreements are all Telangana against Andhra Pradesh on Hyderabad addresses (the 2014 split). Every Source-1 record has a state. 3.8% of targets do not, mostly because the whole address is missing.

## France

Only three regions appear (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) and about eighteen cities. Source 1 writes the region. Sources 2/3 mix the region and the department (`Gironde`, `Nord`, `Loire-Atlantique`). Street types use French abbreviations (`R.`, `BD`, `PL`, `ALL`, `RTE`, `CHEM`, `N°`), and legal forms are `SARL`, `SAS`, `SASU`, `SCI`, `EURL`, `SNC`.

Names come from a small French vocabulary (`Établissements`, `École`, `Sport`, `Union`, `Comité`). Distinct businesses sit on the same street with similar names:

    S1  Établissements Defense SASU | 121 Route de Bordeaux Petit Piquey, Lège-Cap-Ferret
    S2  Établissements Medi SAS     | 47 - ROUTE DE BORDEAUX PETIT PIQUEY, LEGE-CAP-FERRET
    S2  Établissements Defense Groupe SASU | 125 ROUTE DE BORDEAUX PETTI PIQUEY, LÈGE CAP FERRET

A model trained on US and India has seen neither this vocabulary nor this density of near-twins. France is where the leaderboard score is most likely to fall below the validation score.
