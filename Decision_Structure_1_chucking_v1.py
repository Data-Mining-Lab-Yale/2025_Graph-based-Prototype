# ------------------------------------------------------------
# Clause + phrase chunk splitter (spaCy)
# ------------------------------------------------------------
# Install once (from a terminal):
#   pip install spacy
#   python -m spacy download en_core_web_sm
#
# Usage example at the bottom.
# ------------------------------------------------------------

from typing import List, Dict, Any
import spacy
from spacy.tokens import Doc, Span, Token

# Load spaCy English model outside functions so you can reuse it
# You can pass your own nlp if you prefer (e.g., a larger model)
nlp = spacy.load("en_core_web_sm")

def get_phrase_chunks(sent: Span) -> List[Dict[str, Any]]:
    """
    Shallow phrase chunks:
      - NP chunks via spaCy's noun_chunks
      - Light VP chunks using verb heads and their particles/aux
      - PP chunks as preposition heads plus their objects
      - Temporal/adverbial chunks as simple ADVs or time NPs
    Returns a list of dicts with text and coarse type.
    """
    chunks = []

    # 1) NP chunks (built in)
    for np in sent.noun_chunks:
        chunks.append({"text": np.text, "type": "NP", "span": (np.start_char, np.end_char)})

    # 2) VP chunks (very light rule)
    # group a VERB or AUX with its auxiliaries and particles
    for token in sent:
        if token.pos_ in {"VERB", "AUX"} and token.dep_ != "aux":
            # collect auxiliaries and particles attached to this verb
            aux_tokens = [t for t in token.children if t.dep_ == "aux"]
            prt_tokens = [t for t in token.children if t.dep_ == "prt"]
            group = sorted([token] + aux_tokens + prt_tokens, key=lambda t: t.i)
            start = group[0].idx
            end = group[-1].idx + len(group[-1])
            chunks.append({"text": sent.doc.text[start:end], "type": "VP", "span": (start, end)})

    # 3) PP chunks
    # prepositions with pobj
    for token in sent:
        if token.dep_ == "prep":
            pobj = next((t for t in token.children if t.dep_ == "pobj"), None)
            if pobj:
                start = token.idx
                end = pobj.idx + len(pobj)
                chunks.append({"text": sent.doc.text[start:end], "type": "PP", "span": (start, end)})

    # 4) Simple adverbials or temporal expressions
    # this is conservative; many time expressions are NP chunks already
    for token in sent:
        if token.pos_ == "ADV" and token.dep_ in {"advmod", "npmod", "obl"}:
            chunks.append({"text": token.text, "type": "ADVP", "span": (token.idx, token.idx + len(token))})

    # Deduplicate by span, keep stable order
    seen = set()
    uniq = []
    for c in sorted(chunks, key=lambda x: (x["span"][0], x["span"][1])):
        if c["span"] not in seen:
            seen.add(c["span"])
            uniq.append({"text": c["text"], "type": c["type"]})
    return uniq


def get_clause_spans(sent: Span) -> List[Span]:
    """
    Heuristic clause segmentation inside one sentence.
    Rules of thumb:
      - Start a clause at a finite verb that has a subject
      - Also treat coordinated verbs (conj) with own subject or shared subject as clause starts
      - Keep fragments like initial discourse markers as their own clause if they do not attach to a verb
    Returns list of spaCy Span objects.
    """
    doc = sent.doc
    clause_heads = []

    # candidate heads: verbs or auxiliaries with an explicit or inherited subject
    def has_subject(tok: Token) -> bool:
        return any(c.dep_ in {"nsubj", "nsubjpass", "csubj"} for c in tok.children)

    # collect main verb heads
    for tok in sent:
        if tok.pos_ in {"VERB", "AUX"}:
            if has_subject(tok):
                clause_heads.append(tok)
            # coordinated verbs that likely begin their own clause
            elif tok.dep_ == "conj" and tok.head.pos_ in {"VERB", "AUX"}:
                clause_heads.append(tok)

    # fallback: if no verb found, treat the entire sentence as one clause
    if not clause_heads:
        return [sent]

    # sort and make clause spans from head to the right until next head
    clause_heads = sorted(set(clause_heads), key=lambda t: t.i)
    spans: List[Span] = []
    for i, head in enumerate(clause_heads):
        start_i = head.left_edge.i  # include possible subject on the left
        # do not move left past the sentence start
        start_i = max(start_i, sent.start)
        end_i = (clause_heads[i + 1].left_edge.i if i + 1 < len(clause_heads) else sent.end)
        spans.append(doc[start_i:end_i])

    # include leading fragment before first clause head if any tokens exist
    first = spans[0]
    if first.start > sent.start:
        frag = doc[sent.start:first.start]
        if frag.text.strip():
            spans = [frag] + spans

    return spans


def split_text_into_clauses_and_phrases(text: str, nlp_model=None) -> Dict[str, Any]:
    """
    Main API:
      Input: raw text (sentence or subsentence)
      Output: dict with sentence-wise clause and phrase splits
    """
    nlp_local = nlp_model or nlp
    doc: Doc = nlp_local(text)

    results = []
    for sent in doc.sents:
        # clauses
        clauses = get_clause_spans(sent)

        # phrase chunks per clause
        clause_entries = []
        for cl in clauses:
            clause_entries.append({
                "clause_text": cl.text,
                "phrases": get_phrase_chunks(cl)
            })

        results.append({
            "sentence_text": sent.text,
            "clauses": clause_entries
        })

    return {"text": text, "sentences": results}


def pretty_print_split(result: Dict[str, Any]) -> None:
    """
    Print a human friendly view.
    """
    print(f"INPUT: {result['text']}\n")
    for si, s in enumerate(result["sentences"], 1):
        print(f"Sentence {si}: {s['sentence_text']}")
        for ci, c in enumerate(s["clauses"], 1):
            print(f"  Clause {ci}: {c['clause_text']}")
            for p in c["phrases"]:
                print(f"    [{p['type']}] {p['text']}")
        print()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "we ran out of time yesterday afternoon",
        "SO sorry--we ran out of time yesterday afternoon and Dr. Person1 had",
        'The good doctor is swamped in a emails and she wants to make sure he sees it'
    ]
    for s in samples:
        res = split_text_into_clauses_and_phrases(s)
        pretty_print_split(res)
