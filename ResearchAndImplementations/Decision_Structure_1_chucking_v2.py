# ------------------------------------------------------------
# Better clause + phrase chunk splitter (spaCy, rule-enhanced)
# ------------------------------------------------------------
# pip install spacy
# python -m spacy download en_core_web_sm
# Optional: pip install benepar[cpu]; python -m spacy download en_core_web_trf
# ------------------------------------------------------------

from typing import List, Dict, Any, Tuple, Optional
import spacy
from spacy.tokens import Doc, Span, Token

nlp = spacy.load("en_core_web_sm")
Token.set_extension("used_in_chunk", default=False, force=True)

# ------------------------------ helpers ------------------------------

def span_text(doc: Doc, start: int, end: int) -> str:
    return doc.text[doc[start].idx : doc[end-1].idx + len(doc[end-1])]

def add_chunk(spans: List[Tuple[int,int,str]], start: int, end: int, ctype: str):
    if start < end:
        spans.append((start, end, ctype))

def has_subject(tok: Token) -> bool:
    return any(c.dep_ in {"nsubj", "nsubjpass", "csubj"} for c in tok.children)

def is_copula_head(tok: Token) -> bool:
    # adjectival or nominal predicate with a copula child (is/was/etc.)
    if tok.pos_ in {"ADJ", "NOUN", "PROPN"}:
        return any(c.dep_ == "cop" and c.pos_ in {"AUX", "VERB"} for c in tok.children)
    return False

# Add near the top (below imports)
# def first_child(tok: Token, dep_label: str) -> Optional[Token]:
#     """Return the first direct child with a given dependency label, else None."""
#     for c in tok.children:
#         if c.dep_ == dep_label:
#             return c
#     return None
def first_child(tok: Token, dep_label: str) -> Optional[Token]:
    """Return the first direct child with a given dependency label, else None."""
    for c in tok.children:
        if c.dep_ == dep_label:
            return c
    return None

def pobj_right_edge_end(prep_tok: Token) -> Optional[int]:
    """
    Return exclusive end index (int) of the PP object's subtree.
    Handles 'out of time' by looking one level deeper (prep -> prep -> pobj).
    """
    # direct pobj
    pobj = first_child(prep_tok, "pobj")
    if pobj is not None:
        return pobj.right_edge.i + 1
    # nested prep (e.g., 'out' -> 'of' -> pobj)
    for sub in prep_tok.children:
        if sub.dep_ == "prep":
            pobj2 = first_child(sub, "pobj")
            if pobj2 is not None:
                return pobj2.right_edge.i + 1
    return None



# ------------------------------ clause segmentation ------------------------------

def get_clause_spans(sent: Span) -> List[Span]:
    doc = sent.doc
    heads = []

    for tok in sent:
        if tok.pos_ in {"VERB", "AUX"} and (has_subject(tok) or tok.dep_ == "conj"):
            heads.append(tok)
        if is_copula_head(tok) and (has_subject(tok) or tok.dep_ == "conj"):
            # use the predicate token as the head for copular clauses
            heads.append(tok)

    if not heads:
        return [sent]

    heads = sorted(set(heads), key=lambda t: t.i)
    spans: List[Span] = []
    for i, head in enumerate(heads):
        # include left edge up to subject; keep inside sentence bounds
        start_i = max(head.left_edge.i, sent.start)
        # end at the next head's left edge
        end_i = heads[i + 1].left_edge.i if i + 1 < len(heads) else sent.end
        # trim leading/trailing coordinating conjunctions
        while start_i < end_i and doc[start_i].dep_ == "cc":
            start_i += 1
        while end_i - 1 >= start_i and doc[end_i - 1].dep_ == "cc":
            end_i -= 1
        if start_i < end_i:
            spans.append(doc[start_i:end_i])

    # include any preface fragment before the first clause
    if spans and spans[0].start > sent.start:
        pre = doc[sent.start:spans[0].start]
        if pre.text.strip():
            spans = [pre] + spans

    return spans

# ------------------------------ phrase chunks ------------------------------


def get_phrase_chunks(clause: Span) -> List[Dict[str, Any]]:
    """
    Chunks:
      - VP_PHV: phrasal/multiword verb groups (e.g., 'ran out of time', 'to make sure')
      - VP_COP: copula + adjective/noun (e.g., 'is swamped')
      - VP: simple verb groups when nothing fancy
      - PP: preposition + object (kept as one; suppress inner NP duplicates)
      - NP: base NPs not fully inside a PP
      - DISC: discourse fragment at clause start (e.g., 'SO sorry')
    """
    doc = clause.doc
    spans: List[Tuple[int,int,str]] = []

    # 0) Discourse fragment at the very start: consecutive ADV/INTJ/ADJ before a main verb
    i = clause.start
    while i < clause.end and doc[i].pos_ in {"ADV", "INTJ", "ADJ"} and doc[i].dep_ not in {"amod"}:
        i += 1
    if i > clause.start:
        add_chunk(spans, clause.start, i, "DISC")

    # 1) PP chunks (prep + pobj and its subtree)
    # pp_ranges: List[Tuple[int,int]] = []
    # for tok in clause:
    #     if tok.dep_ == "prep":
    #         pobj = first_child(tok, "pobj")        # returns Token or None
    #         if pobj is not None:
    #             start = tok.i
    #             end = pobj.subtree.end             # safe: pobj is a Token
    #             add_chunk(spans, start, end, "PP")
    #             pp_ranges.append((start, end))
    pp_ranges = []
    for tok in clause:
        if tok.dep_ == "prep":
            end = pobj_right_edge_end(tok)
            if end is not None:
                start = tok.i
                add_chunk(spans, start, end, "PP")
                pp_ranges.append((start, end))



    # helper to test if a candidate NP is fully inside any PP span
    def inside_any_pp(s: int, e: int) -> bool:
        for ps, pe in pp_ranges:
            if s >= ps and e <= pe:
                return True
        return False

    # 2) Copula groups: predicate head with a cop child
    for tok in clause:
        if is_copula_head(tok):
            cop = next(c for c in tok.children if c.dep_ == "cop")
            start = min(cop.i, tok.left_edge.i)
            end = max(tok.right_edge.i + 1, cop.right_edge.i + 1)
            add_chunk(spans, start, end, "VP_COP")

    # 3) Phrasal/multiword verbs
    for tok in clause:
        if tok.pos_ in {"VERB"}:
            core_left = tok.i
            core_right = tok.i + 1
            # include auxiliaries to the left
            aux_left = min([a.left_edge.i for a in tok.children if a.dep_ == "aux"], default=tok.i)
            core_left = min(core_left, aux_left)
            # include particles
            prts = [c for c in tok.children if c.dep_ == "prt"]
            if prts:
                core_right = max(core_right, max(p.i for p in prts) + 1)
                core_left = min(core_left, min(p.left_edge.i for p in prts))
            # include specific fixed constructions: "to make sure"
            # if tok.lemma_ == "make":
            #     xcomp_adj = next((c for c in tok.children if c.dep_ in {"xcomp"} and c.pos_ == "ADJ" and c.lemma_ == "sure"), None)
            #     mark_to = next((c for c in tok.children if c.dep_ == "mark" and c.lower_ == "to"), None)
            #     if xcomp_adj and mark_to:
            #         core_left = min(core_left, mark_to.i)
            #         core_right = max(core_right, xcomp_adj.right_edge.i + 1)
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
            # preps = [c for c in tok.children if c.dep_ == "prep"]
            # for p in preps:
            #     pobj = next((c for c in p.children if c.dep_ == "pobj"), None)
            #     if pobj:
            #         core_right = max(core_right, pobj.subtree.end)
            # extend through prepositional complements like "out of time"
            # preps = [c for c in tok.children if c.dep_ == "prep"]
            # for p in preps:
            #     pobj = first_child(p, "pobj")             # <— FIX
            #     if pobj is not None:
            #         core_right = max(core_right, pobj.subtree.end)
            # extend through prepositional complements like "out of time"
            # for p in [c for c in tok.children if c.dep_ == "prep"]:
            #     pobj = first_child(p, "pobj")
            #     if pobj is not None:
            #         core_right = max(core_right, pobj.subtree.end)

            # extend through prepositional complements like "out of time"
            for p in (c for c in tok.children if c.dep_ == "prep"):
                end = pobj_right_edge_end(p)
                if end is not None:
                    core_right = max(core_right, end)                    
                    
            add_chunk(spans, core_left, core_right, "VP_PHV")

    # 4) Simple VP fallback for lone AUX/VERB not covered yet
    covered = set((s, e) for s, e, _ in spans)
    for tok in clause:
        if tok.pos_ in {"VERB", "AUX"}:
            s = tok.left_edge.i
            e = tok.right_edge.i + 1
            if not any(s >= cs and e <= ce for cs, ce in covered):
                add_chunk(spans, s, e, "VP")

    # 5) NP chunks (suppress those fully inside a PP)
    for np in clause.noun_chunks:
        s, e = np.start, np.end
        if not inside_any_pp(s, e):
            add_chunk(spans, s, e, "NP")

    # resolve overlaps: prefer longer spans and priority order
    priority = {"VP_PHV": 4, "VP_COP": 4, "PP": 3, "VP": 2, "NP": 1, "DISC": 1}
    spans.sort(key=lambda x: (-(x[1]-x[0]), -priority.get(x[2],0), x[0], x[1]))

    chosen: List[Tuple[int,int,str]] = []
    used = [False]*(clause.doc.__len__())
    for s, e, t in spans:
        if any(used[i] for i in range(s, e)):
            continue
        for i in range(s, e):
            used[i] = True
        chosen.append((s, e, t))

    # restore document order
    chosen.sort(key=lambda x: (x[0], x[1]))
    out = [{"text": span_text(clause.doc, s, e), "type": t} for s, e, t in chosen]
    return out

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
        'The good doctor is swamped in a emails and she wants to make sure he sees it'
    ]
    for s in samples:
        res = split_text_into_clauses_and_phrases(s)
        pretty_print_split(res)
