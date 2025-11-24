# ------------------------------------------------------------
# Decision Structure: clause + phrase chunk splitter (spaCy)
# - Tight VP bounds (verb + aux + particles)
# - Phrasal verbs: "ran out of time" grouped
# - Copulas: "is swamped" grouped
# - PP spans robust (handles nested preps like "out of time")
# - No use of .subtree.end (avoids generator errors)
# ------------------------------------------------------------
# Install once:
#   pip install spacy
#   python -m spacy download en_core_web_sm
# ------------------------------------------------------------

from typing import List, Dict, Any, Tuple, Optional
import spacy
from spacy.tokens import Doc, Span, Token

nlp = spacy.load("en_core_web_sm")


# ------------------------------ helpers ------------------------------

def span_text(doc: Doc, start: int, end: int) -> str:
    """Return text for token-span [start, end) by token indices."""
    if start >= end:
        return ""
    return doc.text[doc[start].idx : doc[end - 1].idx + len(doc[end - 1])]

def add_chunk(spans: List[Tuple[int,int,str]], start: int, end: int, ctype: str):
    """Collect a chunk span if non-empty."""
    if start < end:
        spans.append((start, end, ctype))

def has_subject(tok: Token) -> bool:
    return any(c.dep_ in {"nsubj", "nsubjpass", "csubj"} for c in tok.children)

def is_copula_head(tok: Token) -> bool:
    """Predicate head with a copula child, e.g., 'doctor [is] swamped'."""
    if tok.pos_ in {"ADJ", "NOUN", "PROPN"}:
        return any(c.dep_ == "cop" for c in tok.children)
    return False

def first_child(tok: Token, dep_label: str) -> Optional[Token]:
    """Return the first direct child with a given dependency label, else None."""
    for c in tok.children:
        if c.dep_ == dep_label:
            return c
    return None

def pobj_end_index(prep_tok: Token) -> Optional[int]:
    """
    Exclusive end index of the PP object's subtree.
    Handles nested preps like 'out of time' (prep -> prep -> pobj).
    """
    pobj = first_child(prep_tok, "pobj")
    if pobj is not None:
        return pobj.right_edge.i + 1
    # nested: prep -> prep -> pobj
    for sub in prep_tok.children:
        if sub.dep_ == "prep":
            pobj2 = first_child(sub, "pobj")
            if pobj2 is not None:
                return pobj2.right_edge.i + 1
    return None


# ------------------------------ clause segmentation ------------------------------

def get_clause_spans(sent: Span) -> List[Span]:
    """
    Heuristic clause segmentation inside one sentence.
    Chooses heads at finite verbs with subjects, coordinated verbs, and copula predicates.
    Trims dangling 'and' at edges. Keeps preface fragments as separate clauses.
    """
    doc = sent.doc
    heads: List[Token] = []

    for tok in sent:
        if tok.pos_ in {"VERB", "AUX"} and (has_subject(tok) or tok.dep_ == "conj"):
            heads.append(tok)
        if is_copula_head(tok) and (has_subject(tok) or tok.dep_ == "conj"):
            heads.append(tok)

    if not heads:
        return [sent]

    heads = sorted(set(heads), key=lambda t: t.i)
    spans: List[Span] = []
    for i, head in enumerate(heads):
        start_i = max(head.left_edge.i, sent.start)
        end_i = heads[i + 1].left_edge.i if i + 1 < len(heads) else sent.end
        # trim leading/trailing coordinating conjunctions
        while start_i < end_i and doc[start_i].dep_ == "cc":
            start_i += 1
        while end_i - 1 >= start_i and doc[end_i - 1].dep_ == "cc":
            end_i -= 1
        if start_i < end_i:
            spans.append(doc[start_i:end_i])

    # include a preface fragment before the first clause (e.g., "SO sorry")
    if spans and spans[0].start > sent.start:
        pre = doc[sent.start:spans[0].start]
        if pre.text.strip():
            spans = [pre] + spans

    return spans


# ------------------------------ phrase chunks ------------------------------

def get_phrase_chunks(clause: Span) -> List[Dict[str, Any]]:
    """
    Chunks:
      - DISC: initial discourse fragment (e.g., 'SO sorry')
      - PP: preposition + its object (handles 'out of time')
      - VP_COP: copula + predicate (e.g., 'is swamped')
      - VP_PHV: phrasal/multiword verbs (e.g., 'ran out of time', 'to make sure')
      - VP: simple verb group (verb + aux + particles) as fallback
      - NP: base NPs not fully inside a PP
    """
    doc = clause.doc
    spans: List[Tuple[int,int,str]] = []

    # 0) Discourse fragment at the very start: chain of ADV/INTJ/ADJ
    i = clause.start
    while i < clause.end and doc[i].pos_ in {"ADV", "INTJ", "ADJ"} and doc[i].dep_ not in {"amod"}:
        i += 1
    if i > clause.start:
        add_chunk(spans, clause.start, i, "DISC")

    # 1) PP chunks (prep + pobj, include nested 'out of time')
    pp_ranges: List[Tuple[int,int]] = []
    for tok in clause:
        if tok.dep_ == "prep":
            end = pobj_end_index(tok)
            if end is not None:
                start = tok.i
                add_chunk(spans, start, end, "PP")
                pp_ranges.append((start, end))

    def inside_any_pp(s: int, e: int) -> bool:
        for ps, pe in pp_ranges:
            if s >= ps and e <= pe:
                return True
        return False

    # 2) Copula groups: predicate head with a cop child
    for tok in clause:
        if is_copula_head(tok):
            cop = first_child(tok, "cop")
            if cop is not None:
                start = min(cop.i, tok.left_edge.i)
                end = max(tok.right_edge.i + 1, cop.right_edge.i + 1)
                add_chunk(spans, start, end, "VP_COP")

    # 3) Phrasal/multiword verbs
    for tok in clause:
        if tok.pos_ == "VERB":
            core_left = tok.i
            core_right = tok.i + 1
            # include auxiliaries
            auxs = [c for c in tok.children if c.dep_ == "aux"]
            if auxs:
                core_left = min(core_left, min(a.i for a in auxs))
                core_right = max(core_right, max(a.i for a in auxs) + 1)
            # include particles
            prts = [c for c in tok.children if c.dep_ == "prt"]
            if prts:
                core_left = min(core_left, min(p.i for p in prts))
                core_right = max(core_right, max(p.i for p in prts) + 1)
            # specific fixed construction: "to make sure"
            if tok.lemma_ == "make":
                xcomp_adj = None
                mark_to = None
                for c in tok.children:
                    if c.dep_ == "xcomp" and c.pos_ == "ADJ" and c.lemma_ == "sure":
                        xcomp_adj = c
                    elif c.dep_ == "mark" and c.lower_ == "to":
                        mark_to = c
                if xcomp_adj is not None and mark_to is not None:
                    core_left = min(core_left, mark_to.i)
                    core_right = max(core_right, xcomp_adj.right_edge.i + 1)
            # extend through prepositional complements like "out of time"
            for p in (c for c in tok.children if c.dep_ == "prep"):
                end = pobj_end_index(p)
                if end is not None:
                    core_right = max(core_right, end)
            add_chunk(spans, core_left, core_right, "VP_PHV")

    # 4) Simple VP fallback: verb + auxiliaries + particles only
    covered = set((s, e) for s, e, _ in spans)
    for tok in clause:
        if tok.pos_ in {"VERB", "AUX"}:
            group = [tok]
            group += [c for c in tok.children if c.dep_ == "aux"]
            group += [c for c in tok.children if c.dep_ == "prt"]
            group.sort(key=lambda t: t.i)
            s = group[0].i
            e = group[-1].i + 1
            if not any(s >= cs and e <= ce for cs, ce in covered):
                add_chunk(spans, s, e, "VP")

    # 5) NP chunks (suppress those fully inside a PP)
    for np in clause.noun_chunks:
        s, e = np.start, np.end
        if not inside_any_pp(s, e):
            add_chunk(spans, s, e, "NP")

    # resolve overlaps: prefer longer spans and priority order
    priority = {"VP_PHV": 5, "VP_COP": 5, "PP": 4, "VP": 3, "NP": 2, "DISC": 1}
    spans.sort(key=lambda x: (-(x[1]-x[0]), -priority.get(x[2], 0), x[0], x[1]))

    chosen: List[Tuple[int,int,str]] = []
    used = set()
    for s, e, t in spans:
        if any(i in used for i in range(s, e)):
            continue
        for i in range(s, e):
            used.add(i)
        chosen.append((s, e, t))

    # restore order
    chosen.sort(key=lambda x: (x[0], x[1]))
    return [{"text": span_text(clause.doc, s, e), "type": t} for s, e, t in chosen]


# ------------------------------ main API ------------------------------

def split_text_into_clauses_and_phrases(text: str, nlp_model=None) -> Dict[str, Any]:
    nlp_local = nlp_model or nlp
    doc: Doc = nlp_local(text)
    results = []
    for sent in doc.sents:
        clause_spans = get_clause_spans(sent)
        clauses = []
        for cl in clause_spans:
            clauses.append({
                "clause_text": cl.text.strip(),
                "phrases": get_phrase_chunks(cl)
            })
        results.append({"sentence_text": sent.text.strip(), "clauses": clauses})
    return {"text": text, "sentences": results}

def pretty_print_split(result: Dict[str, Any]) -> None:
    print(f"INPUT: {result['text']}\n")
    for si, s in enumerate(result["sentences"], 1):
        print(f"Sentence {si}: {s['sentence_text']}")
        for ci, c in enumerate(s["clauses"], 1):
            print(f"  Clause {ci}: {c['clause_text']}")
            for p in c["phrases"]:
                print(f"    [{p['type']}] {p['text']}")
        print()


# ------------------------------ demo ------------------------------
if __name__ == "__main__":
    samples = [
        "we ran out of time yesterday afternoon",
        "SO sorry--we ran out of time yesterday afternoon and Dr. Person1 had",
        "The good doctor is swamped in a emails and she wants to make sure he sees it",
    ]
    for s in samples:
        res = split_text_into_clauses_and_phrases(s)
        pretty_print_split(res)
