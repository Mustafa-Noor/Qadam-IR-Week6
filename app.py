"""Browsing module for the social rehab app.

Provides scenario search (Boolean and TF-IDF vector ranking), category
browsing, breadcrumb data, related scenarios and scenario previews.
Scenario content is a draft until reviewed by a clinician.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "scenarios.json"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "you", "your", "i",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
QUERY_TOKEN_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+")


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_scenarios():
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)["scenarios"]


class SearchIndex:
    def __init__(self, scenarios):
        self.scenarios = {s["id"]: s for s in scenarios}
        self.all_ids = set(self.scenarios)
        self.postings = defaultdict(set)  # term -> scenario ids (Boolean)
        self.tf = {}
        self.df = Counter()

        for s in scenarios:
            text = " ".join([
                s["title"], s["summary"], s["category"],
                s["subcategory"], " ".join(s["tags"]),
            ])
            counts = Counter(tokenize(text))
            self.tf[s["id"]] = counts
            for term in counts:
                self.postings[term].add(s["id"])
                self.df[term] += 1

        n = len(scenarios)
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in self.df.items()}
        self.vectors = {sid: self._vectorize(c) for sid, c in self.tf.items()}

    def _vectorize(self, counts):
        vec = {
            t: (1 + math.log(c)) * self.idf[t]
            for t, c in counts.items() if t in self.idf
        }
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    # Boolean retrieval: AND, OR, NOT (capitals) and parentheses.
    # NOT binds tightest, then AND, then OR.
    def boolean_search(self, query):
        tokens = QUERY_TOKEN_RE.findall(query)
        if not tokens:
            return set(self.all_ids)
        pos = 0

        def peek():
            return tokens[pos] if pos < len(tokens) else None

        def take():
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("the query ends unexpectedly")
            tok = tokens[pos]
            pos += 1
            return tok

        def parse_or():
            result = parse_and()
            while peek() == "OR":
                take()
                result = result | parse_and()
            return result

        def parse_and():
            result = parse_not()
            while peek() == "AND":
                take()
                result = result & parse_not()
            return result

        def parse_not():
            if peek() == "NOT":
                take()
                return self.all_ids - parse_not()
            return parse_atom()

        def parse_atom():
            tok = take()
            if tok == "(":
                result = parse_or()
                if take() != ")":
                    raise ValueError("a closing parenthesis is missing")
                return result
            if tok in {")", "AND", "OR"}:
                raise ValueError(f"'{tok}' is in the wrong place")
            terms = tokenize(tok)
            if not terms:
                return set()
            return set(self.postings.get(terms[0], set()))

        result = parse_or()
        if pos != len(tokens):
            raise ValueError("part of the query could not be read")
        return result

    # Vector retrieval: cosine similarity between TF-IDF vectors.
    def vector_scores(self, query):
        q_vec = self._vectorize(Counter(tokenize(query)))
        if not q_vec:
            return {}
        scores = {}
        for sid, d_vec in self.vectors.items():
            score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            if score > 0:
                scores[sid] = score
        return scores

    def related(self, sid, limit=3):
        base = self.vectors[sid]
        sims = []
        for other, vec in self.vectors.items():
            if other == sid:
                continue
            score = sum(w * vec.get(t, 0.0) for t, w in base.items())
            if score > 0:
                sims.append((score, other))
        sims.sort(reverse=True)
        return [
            {"id": o, "title": self.scenarios[o]["title"], "score": round(s, 4)}
            for s, o in sims[:limit]
        ]

    def search(self, query, mode, category="", subcategory="", difficulty=""):
        if mode == "boolean":
            scored = {sid: None for sid in self.boolean_search(query)}
        elif query.strip():
            scored = self.vector_scores(query)
        else:
            scored = {sid: None for sid in self.all_ids}

        results = []
        for sid, score in scored.items():
            s = self.scenarios[sid]
            if category and s["category"] != category:
                continue
            if subcategory and s["subcategory"] != subcategory:
                continue
            if difficulty and s["difficulty"] != difficulty:
                continue
            results.append({
                "id": sid,
                "title": s["title"],
                "category": s["category"],
                "subcategory": s["subcategory"],
                "difficulty": s["difficulty"],
                "summary": s["summary"],
                "score": round(score, 4) if score is not None else None,
            })
        results.sort(key=lambda r: (-(r["score"] or 0), r["title"]))
        return results


index = SearchIndex(load_scenarios())


@app.get("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/categories")
def categories():
    tree = defaultdict(Counter)
    for s in index.scenarios.values():
        tree[s["category"]][s["subcategory"]] += 1
    return jsonify({"categories": [
        {
            "name": cat,
            "count": sum(subs.values()),
            "subcategories": [
                {"name": name, "count": count}
                for name, count in sorted(subs.items())
            ],
        }
        for cat, subs in sorted(tree.items())
    ]})


@app.get("/api/search")
def search():
    query = request.args.get("q", "")
    mode = request.args.get("mode", "vector")
    if mode not in {"vector", "boolean"}:
        return jsonify({"error": "Mode must be 'vector' or 'boolean'."}), 400
    try:
        results = index.search(
            query, mode,
            category=request.args.get("category", ""),
            subcategory=request.args.get("subcategory", ""),
            difficulty=request.args.get("difficulty", ""),
        )
    except ValueError as exc:
        return jsonify({
            "error": (
                f"Could not read the search: {exc}. Use AND, OR, NOT and "
                "parentheses in capitals, for example: (friend OR neighbour) AND cafe"
            ),
        }), 400
    return jsonify({
        "query": query,
        "mode": mode,
        "count": len(results),
        "results": results,
    })


@app.get("/api/scenarios/<sid>")
def scenario(sid):
    s = index.scenarios.get(sid)
    if s is None:
        abort(404)
    return jsonify({
        "scenario": s,
        "breadcrumb": [
            {"label": "All scenarios"},
            {"label": s["category"], "category": s["category"]},
            {"label": s["subcategory"], "category": s["category"],
             "subcategory": s["subcategory"]},
            {"label": s["title"], "scenario_id": sid},
        ],
        "related": index.related(sid),
    })


if __name__ == "__main__":
    app.run(debug=True)
