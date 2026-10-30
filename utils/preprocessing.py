"""
WanderWise - Text Preprocessing Pipeline
Information Retrieval Laboratory Project
Demonstrating Practical 6 & 7

Pipeline steps:
1. Lowercasing
2. Special character & punctuation removal
3. Tokenization
4. Stopword removal (self-contained, no external downloads)
5. Stemming (Porter Stemmer implementation, self-contained)
6. Whitespace normalization
"""

import re
import string

# Self-contained comprehensive standard English stopwords list
# Guarantees zero network calls or nltk.download failures in Streamlit Cloud
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", 
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", 
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", 
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", 
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", 
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves", "want", "like", "visit", "trip", "destination", "place", "places",
    "suggest", "find", "show", "give", "looking", "love", "need", "prefer"
}

class PorterStemmer:
    """
    Standard Martin Porter's 1980 Stemming Algorithm implementation.
    Pure Python, deterministic, fast, and self-contained.
    """
    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i):
        if self.b[i] in 'aeiou':
            return False
        if self.b[i] == 'y':
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self):
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i):
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i):
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in 'wxY':
            return False
        return True

    def _ends(self, s):
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        if self.b[o:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length:]
        self.k = self.j + length

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self):
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in 'lsz':
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def _step2(self):
        c = self.b[self.k - 1] if self.k > self.k0 else ''
        if c == 'a':
            if self._ends("ational"): self._r("ate")
            elif self._ends("tional"): self._r("tion")
        elif c == 'c':
            if self._ends("enci"): self._r("ence")
            elif self._ends("anci"): self._r("ance")
        elif c == 'e':
            if self._ends("izer"): self._r("ize")
        elif c == 'l':
            if self._ends("bli"): self._r("ble")
            elif self._ends("alli"): self._r("al")
            elif self._ends("entli"): self._r("ent")
            elif self._ends("eli"): self._r("e")
            elif self._ends("ousli"): self._r("ous")
        elif c == 'o':
            if self._ends("ization"): self._r("ize")
            elif self._ends("ation"): self._r("ate")
            elif self._ends("ator"): self._r("ate")
        elif c == 's':
            if self._ends("alism"): self._r("al")
            elif self._ends("iveness"): self._r("ive")
            elif self._ends("fulness"): self._r("ful")
            elif self._ends("ousness"): self._r("ous")
        elif c == 't':
            if self._ends("aliti"): self._r("al")
            elif self._ends("iviti"): self._r("ive")
            elif self._ends("biliti"): self._r("ble")

    def _step3(self):
        c = self.b[self.k]
        if c == 'e':
            if self._ends("icate"): self._r("al")
            elif self._ends("ative"): self._r("")
            elif self._ends("alize"): self._r("al")
        elif c == 'i':
            if self._ends("iciti"): self._r("al")
        elif c == 'l':
            if self._ends("ical"): self._r("al")
            elif self._ends("ful"): self._r("")
        elif c == 's':
            if self._ends("ness"): self._r("")

    def _step4(self):
        if self.k <= self.k0 + 1:
            return
        c = self.b[self.k - 1]
        if c == 'a':
            if self._ends("al"): pass
            else: return
        elif c == 'c':
            if self._ends("ance") or self._ends("ence"): pass
            else: return
        elif c == 'e':
            if self._ends("er"): pass
            else: return
        elif c == 'i':
            if self._ends("ic"): pass
            else: return
        elif c == 'l':
            if self._ends("able") or self._ends("ible"): pass
            else: return
        elif c == 'n':
            if self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent"): pass
            else: return
        elif c == 'o':
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in 'st': pass
            elif self._ends("ou"): pass
            else: return
        elif c == 's':
            if self._ends("ism"): pass
            else: return
        elif c == 't':
            if self._ends("ate") or self._ends("iti"): pass
            else: return
        elif c == 'u':
            if self._ends("ous"): pass
            else: return
        elif c == 'v':
            if self._ends("ive"): pass
            else: return
        elif c == 'z':
            if self._ends("ize"): pass
            else: return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self):
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, p):
        p = str(p).lower().strip()
        if len(p) <= 2:
            return p
        self.b = p
        self.k = len(p) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[:self.k + 1]


_stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Convert text to lowercase, remove punctuation, numbers, and special characters.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Replace punctuation and special characters with spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Normalize multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """
    Split cleaned text into individual word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Filter out common English stop words and query conversational filler words.
    """
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]

def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Apply Porter Stemming to a list of tokens.
    """
    return [_stemmer.stem(token) for token in tokens]

def preprocess_tokens(text: str) -> list[str]:
    """
    Full pipeline returning preprocessed token list:
    Clean -> Tokenize -> Remove Stopwords -> Stem.
    """
    raw_tokens = tokenize(text)
    filtered = remove_stopwords(raw_tokens)
    stemmed = stem_tokens(filtered)
    return stemmed

def preprocess_text(text: str) -> str:
    """
    Full pipeline returning space-joined normalized preprocessed string:
    Ready for TfidfVectorizer or query processing.
    """
    tokens = preprocess_tokens(text)
    return " ".join(tokens)
