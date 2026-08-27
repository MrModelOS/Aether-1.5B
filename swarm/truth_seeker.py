"""
Swarm-RAG / Truth-Seeker: энтропийный триггер E_thresh (swarm/truth_seeker.py:1)

Зачем: var(phi) = физический сигнал неуверенности волнового поля.
Если фазы гасят друг друга (деструктивная интерференция) -> дисперсия растет -> фактов не хватает -> запускаем поиск.

Пайплайн: Planner | QueryRefiner | FactChecker | Synthesizer (Holographic State Collapse)
"""
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SwarmConfig:
    e_thresh: float = 0.35  # порог энтропии
    top_k: int = 5
    max_queries: int = 3

class EntropyTrigger:
    """Вычисляет E = var(phi) по амплитуде (swarm/truth_seeker.py:20)"""
    def __init__(self, thresh: float = 0.35):
        self.thresh = thresh

    def compute(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: [B, T, D] или [B, D] complex/real
        if torch.is_complex(phi):
            amp = phi.abs()
        else:
            amp = phi
        # var по D
        e = amp.var(dim=-1).mean(dim=-1) if amp.dim() == 3 else amp.var(dim=-1)
        # e: [B]
        return e

    def should_search(self, phi: torch.Tensor) -> torch.Tensor:
        e = self.compute(phi)  # [B]
        return e > self.thresh, e

class QueryRefiner:
    """Генерит поисковые запросы из phi (swarm/truth_seeker.py:40) — прототип"""
    def refine(self, prompt: str, phi: torch.Tensor, n: int = 3) -> List[str]:
        # В реале: LLM генерит queries. Здесь эвристика для прототипа.
        base = prompt[:120].strip()
        return [f"{base} facts {i+1}" for i in range(n)]

class FactChecker:
    """NLI-фильтр фейков (swarm/truth_seeker.py:47) — прототип через косинус"""
    def score(self, claim: str, evidence: str) -> float:
        # Заглушка: в реале cross-encoder/nli модель
        # Здесь Jaccard по токенам
        a, b = set(claim.lower().split()), set(evidence.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def filter(self, evidences: List[str], query: str, thresh: float = 0.2) -> List[str]:
        scored = [(e, self.score(query, e)) for e in evidences]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored if s >= thresh]

class TruthSeekerSwarm:
    """Оркестратор (swarm/truth_seeker.py:62)"""
    def __init__(self, config: SwarmConfig = SwarmConfig(), search_fn=None):
        self.config = config
        self.trigger = EntropyTrigger(config.e_thresh)
        self.refiner = QueryRefiner()
        self.checker = FactChecker()
        self.search_fn = search_fn  # Callable[[query], List[str]] — Brave/Tavily API

    async def maybe_search(self, prompt: str, phi: torch.Tensor) -> Optional[dict]:
        need, e = self.trigger.should_search(phi)
        # need: [B] bool, берем любой True
        if not need.any():
            return None
        queries = self.refiner.refine(prompt, phi, n=self.config.max_queries)
        all_evid = []
        if self.search_fn is None:
            # оффлайн заглушка
            all_evid = [f"Evidence for {q}: ... (stub)" for q in queries]
        else:
            for q in queries:
                try:
                    res = self.search_fn(q)
                    if isinstance(res, list):
                        all_evid.extend(res)
                    else:
                        all_evid.append(str(res))
                except Exception as ex:
                    all_evid.append(f"search error {q}: {ex}")

        filtered = self.checker.filter(all_evid, prompt)
        return {
            "entropy": e.detach().cpu().tolist(),
            "queries": queries,
            "evidence": filtered[: self.config.top_k],
            "triggered": need.detach().cpu().tolist(),
        }

    def collapse(self, base_answer: str, evidence: List[str]) -> str:
        """Holographic State Collapse: синтез ответа с верификацией (swarm/truth_seeker.py:98)"""
        if not evidence:
            return base_answer
        ev_block = "\n".join(f"- {e}" for e in evidence[:3])
        return f"{base_answer}\n\n[Верифицировано по источникам]:\n{ev_block}"

# Синхронный хелпер для train_pstc
def check_entropy_and_search(prompt: str, phi: torch.Tensor, swarm: TruthSeekerSwarm):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(swarm.maybe_search(prompt, phi))
