"""关键结论核验。"""

from __future__ import annotations

from evidence.models import Claim
from evidence.source_ranker import source_score


def verify_claim(claim: Claim) -> Claim:
    if not claim.evidence:
        claim.confidence = 0.0
        return claim
    claim.confidence = max(source_score(item) for item in claim.evidence)
    return claim
