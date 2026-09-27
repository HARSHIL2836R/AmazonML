"""Static word lists used by normalization.

Everything here is general linguistic or postal knowledge (legal-form suffixes,
street-type abbreviations, state and region names). None of it identifies a
specific business, so it stays inside the challenge's no-external-lookup rule.

Country handling is open-set: a country without an entry here still gets the
generic normalization; it only misses the country-specific canonical forms.
"""

# --------------------------------------------------------------------------
# Business names
# --------------------------------------------------------------------------

# Legal-form tokens -> canonical form. Keys are post-normalization tokens
# (lowercase, accents stripped, dots removed, so "L.L.C." is already "llc").
LEGAL_FORMS = {
    # US / generic English
    "llc": "llc", "inc": "inc", "incorporated": "inc", "lnc": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp", "pllc": "pllc",
    "plc": "plc", "pc": "pc", "pa": "pa",
    # India
    "pvt": "pvt", "private": "pvt", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "scm": "scm", "gie": "gie", "ets": "ets", "etablissements": "ets",
    # Other common forms, so an unseen country is not left bare
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "pty": "pty",
    "srl": "srl", "spa": "spa", "sl": "sl", "ab": "ab", "oy": "oy",
}

# Leading honorifics and filler that the noise generator prepends.
HONORIFICS = {
    "the", "shri", "sri", "shree", "smt", "dr", "mr", "mrs", "ms", "messrs", "m s",
}

# Alias markers. The Source-1 name follows the marker in the training data
# ("Noviorbi DBA Noble & Co" -> S1 "Noble & Co"). Multi-word markers first.
ALIAS_MARKERS = [
    "formerly known as", "doing business as", "also known as", "known as",
    "trading as", "t/a", "d/b/a", "a/k/a", "f/k/a", "dba", "aka", "fka", "formerly",
]

# Country words that the generator adds or drops inside names: "(India)".
COUNTRY_WORDS = {"india", "france", "usa", "us", "america"}

# Leetspeak substitutions applied only inside tokens that mix letters and
# digits ("J0hnson", "F1ores", "5ervices"); pure numbers are left alone.
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Top-level domains stripped from domain-style names ("johnsonfreight.com").
TLDS = ["co.in", "com", "net", "org", "in", "co", "fr", "biz", "info", "io", "us"]

# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

# Street-type and unit words -> canonical short form (all countries).
STREET_TYPES = {
    "street": "st", "str": "st", "saint": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl",
    "trail": "trl", "highway": "hwy", "parkway": "pkwy", "terrace": "ter",
    "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w",
    "apartment": "apt", "building": "bldg", "floor": "fl", "first": "1st",
    "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sainte": "ste", "mount": "mt",
    # French
    "allee": "allee", "all": "allee", "route": "rte", "chemin": "chem", "chm": "chem",
    "ch": "chem", "impasse": "imp", "cours": "crs", "residence": "res",
    "etage": "fl",
}

# Address label words that carry no identity ("House No 655" vs "No 655").
ADDRESS_LABELS = {
    "no", "n", "number", "num", "door", "house", "h", "hno", "hn", "flat", "plot",
    "shop", "office", "off", "room", "unit", "suite", "pmb", "po", "box", "bis", "ter",
    "null", "none", "na", "nil",
}

# Tokens that only appear as a French street type; applied for France only
# because a bare "r" elsewhere is not a street type.
FR_STREET_TYPES = {"r": "rue"}

CITY_ALIASES = {
    "bombay": "mumbai", "poona": "pune", "bangalore": "bengaluru", "calcutta": "kolkata",
    "madras": "chennai", "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram",
    "vishakhapatnam": "visakhapatnam", "ahmadabad": "ahmedabad", "orissa": "odisha",
    "keralam": "kerala", "baroda": "vadodara", "mysore": "mysuru", "cochin": "kochi",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli and daman and diu": "dn",
    # Native-script spellings observed in Sources 2/3 (the only non-Latin
    # address fragments in the data are these state names).
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka",
    "தமிழ்நாடு": "tn", "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "tg",
    "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od",
}
# Two-letter codes used by Source 3 ("MH", "TG", plus older "OR", "TS", "CT", "UT").
IN_STATE_CODES = {c: c for c in set(IN_STATES.values())} | {"or": "od", "ts": "tg", "ct": "cg", "ut": "uk"}

# France: departments and regions both map to the region code, because
# Source 1 uses the region ("Nouvelle-Aquitaine") while Sources 2/3 mix in
# the department ("Gironde"). Metropolitan France only.
FR_REGIONS = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre",
    "centre val de loire": "cvl", "corse": "cor", "grand est": "ges",
    "hauts de france": "hdf", "ile de france": "idf", "normandie": "nor",
    "nouvelle aquitaine": "naq", "occitanie": "occ", "pays de la loire": "pdl",
    "provence alpes cote d azur": "pac", "provence alpes cote dazur": "pac",
}
FR_DEPARTMENTS = {
    "ara": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire",
            "puy de dome", "rhone", "savoie", "haute savoie"],
    "bfc": ["cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire", "yonne",
            "territoire de belfort"],
    "bre": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "cor": ["corse du sud", "haute corse"],
    "ges": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse",
            "moselle", "bas rhin", "haut rhin", "vosges"],
    "hdf": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "idf": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"],
    "nor": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "naq": ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde",
            "landes", "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne",
            "haute vienne"],
    "occ": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot",
            "lozere", "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pdl": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "pac": ["alpes de haute provence", "hautes alpes", "alpes maritimes", "bouches du rhone",
            "var", "vaucluse"],
}
FR_STATES = dict(FR_REGIONS)
for _region, _deps in FR_DEPARTMENTS.items():
    for _dep in _deps:
        FR_STATES[_dep] = _region

# country label (lowercased) -> {normalized component: state code}
STATE_TABLES = {
    "us": US_STATES | {c: c for c in set(US_STATES.values())},
    "india": IN_STATES | IN_STATE_CODES,
    "france": FR_STATES,
}
