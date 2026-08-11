"""Configurable query-understanding strategies for advanced RAG retrieval.

All five requested strategies are implemented:
1. contextual reformulation,
2. query decomposition,
3. multi-query expansion,
4. HyDE (Hypothetical Document Embeddings), and
5. hypothetical-question generation.

The default configuration intentionally activates only reformulation, decomposition,
and multi-query. Running all five on every request usually adds redundant queries,
latency, and model cost. HyDE and hypothetical questions remain switchable for
experiments and difficult queries.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ttl_cache import TTLCache

# Every strategy this module implements. Used to reject an unknown name in `build`
# and, by serving_agent, to report which of them actually ran. One definition so
# the two cannot disagree and silently under-report.
STRATEGIES = frozenset(
    {
        "reformulation",
        "decomposition",
        "multi_query",
        "hyde",
        "hypothetical_questions",
    }
)


@dataclass
class QueryBundle:
    standalone_question: str
    retrieval_queries: list[str]
    strategy_outputs: dict[str, Any] = field(default_factory=dict)


class QueryStrategyEngine:
    def __init__(self, llm, config: dict[str, Any], cache_config: dict[str, Any]):
        self.llm = llm
        self.config = config
        self.active = [
            str(name).strip().lower()
            for name in config.get(
                "active_strategies",
                ["reformulation", "decomposition", "multi_query"],
            )
        ]
        self.max_total_queries = int(config.get("max_total_queries", 10))
        self.cache = TTLCache[str, QueryBundle](
            ttl_seconds=float(cache_config.get("query_plan_ttl_seconds", 600)),
            max_entries=int(cache_config.get("max_query_plans", 256)),
        )

    def build(
        self,
        question: str,
        chat_history: str,
        strategies_override: list[str] | None = None,
    ) -> QueryBundle:
        active = [
            str(name).strip().lower()
            for name in (strategies_override or self.active)
            if str(name).strip()
        ]
        unknown = sorted(set(active) - STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown query strategies: {unknown}")
        key = self._cache_key(question, chat_history, active)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        outputs: dict[str, Any] = {}
        standalone = question.strip()

        if "reformulation" in active:
            try:
                standalone = self.reformulate(question, chat_history)
                outputs["reformulation"] = standalone
            except Exception as exc:
                outputs["reformulation_error"] = str(exc)

        base_queries = [standalone]
        if "decomposition" in active:
            try:
                decomposed = self.decompose(standalone)
                outputs["decomposition"] = decomposed
                base_queries.extend(decomposed)
            except Exception as exc:
                outputs["decomposition_error"] = str(exc)

        expanded = list(base_queries)
        if "multi_query" in active:
            try:
                multi = self.multi_query(standalone)
                outputs["multi_query"] = multi
                expanded.extend(multi)
            except Exception as exc:
                outputs["multi_query_error"] = str(exc)

        if "hyde" in active:
            try:
                hyde = self.hyde(standalone)
                outputs["hyde"] = hyde
                expanded.append(hyde)
            except Exception as exc:
                outputs["hyde_error"] = str(exc)

        if "hypothetical_questions" in active:
            try:
                questions = self.hypothetical_questions(standalone)
                outputs["hypothetical_questions"] = questions
                expanded.extend(questions)
            except Exception as exc:
                outputs["hypothetical_questions_error"] = str(exc)

        queries = self._deduplicate(expanded)[: self.max_total_queries]
        bundle = QueryBundle(
            standalone_question=standalone,
            retrieval_queries=queries or [question],
            strategy_outputs=outputs,
        )
        self.cache.set(key, bundle)
        return bundle

    def reformulate(self, question: str, chat_history: str) -> str:
        if not chat_history.strip():
            return question.strip()
        prompt = PromptTemplate.from_template(
            """Rewrite the latest question as one standalone retrieval query.
Resolve pronouns and references using the conversation history. Preserve patient IDs,
document IDs, dates, medication names, and laboratory names exactly. Do not answer.

Conversation history:
{history}

Latest question:
{question}

Standalone query:"""
        )
        value = (prompt | self.llm | StrOutputParser()).invoke(
            {"history": chat_history[-5000:], "question": question}
        )
        return self._clean_line(value) or question.strip()

    def decompose(self, question: str) -> list[str]:
        prompt = PromptTemplate.from_template(
            """Break the medical-record navigation question into zero to three focused
retrieval subquestions. Decompose only when the request contains multiple facts,
dates, comparisons, or steps. Return a JSON array of strings and nothing else.

Question: {question}"""
        )
        value = (prompt | self.llm | StrOutputParser()).invoke({"question": question})
        return self._parse_list(value, limit=3)

    def multi_query(self, question: str) -> list[str]:
        count = int(self.config.get("multi_query_count", 3))
        prompt = PromptTemplate.from_template(
            """Generate {count} alternative search queries for the same patient-record
question. Use different wording and include likely clinical document terminology.
Do not add facts. Return a JSON array of strings and nothing else.

Question: {question}"""
        )
        value = (prompt | self.llm | StrOutputParser()).invoke(
            {"count": count, "question": question}
        )
        return self._parse_list(value, limit=count)

    def hyde(self, question: str) -> str:
        prompt = PromptTemplate.from_template(
            """Write a short hypothetical clinical-record passage that would contain the
answer to the question. It is only a retrieval probe, not a real patient fact. Use
record-like wording, dates only when the question supplies them, and no diagnosis or
recommendation.

Question: {question}

Hypothetical passage:"""
        )
        value = (prompt | self.llm | StrOutputParser()).invoke({"question": question})
        return value.strip()[:3000]

    def hypothetical_questions(self, question: str) -> list[str]:
        count = int(self.config.get("hypothetical_question_count", 2))
        prompt = PromptTemplate.from_template(
            """Generate {count} concise questions that a relevant patient-record chunk
would naturally answer and that would help retrieve evidence for the original
question. Return a JSON array of strings and nothing else.

Original question: {question}"""
        )
        value = (prompt | self.llm | StrOutputParser()).invoke(
            {"count": count, "question": question}
        )
        return self._parse_list(value, limit=count)

    def _cache_key(
        self, question: str, history: str, active: list[str]
    ) -> str:
        payload = json.dumps(
            {
                "question": question,
                "history": history[-5000:],
                "active": active,
                "config": self.config,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_line(value: str) -> str:
        value = value.strip().strip('"').strip("'")
        return re.sub(r"\s+", " ", value)

    @classmethod
    def _parse_list(cls, value: str, limit: int) -> list[str]:
        text = value.strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return cls._deduplicate(
                        [cls._clean_line(str(item)) for item in parsed if str(item).strip()]
                    )[:limit]
            except json.JSONDecodeError:
                pass
        lines = [
            re.sub(r"^[-*\d.()\s]+", "", line).strip()
            for line in text.splitlines()
        ]
        return cls._deduplicate([line for line in lines if line])[:limit]

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            clean = re.sub(r"\s+", " ", value).strip()
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                result.append(clean)
        return result
