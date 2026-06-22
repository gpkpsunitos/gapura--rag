# Anti-Hallucination RAG Improvements

**Date**: 2026-04-10
**Scope**: Reduce all hallucination types (fabricated facts, wrong citations, mixed truth/fiction) while maintaining balanced detail and accuracy in long-form responses.

## Problem

Users encounter three types of hallucinations:
1. **Fabricated facts**: AI adds information not found in any uploaded document
2. **Wrong citations**: AI cites a source but attributes information to the wrong document/page
3. **Mixed truth/fiction**: AI partially gets it right but fills gaps with made-up details

Root causes:
- Weak retrieval returns irrelevant context, forcing the LLM to bridge gaps with fabrication
- Prompts don't enforce strict evidence boundaries
- Single-pass verification is easy to game
- No post-generation claim-evidence cross-checking

## Approach: Prompt Engineering + Evidence Strengthening

### Section 1: Retrieval Quality (`retriever.py`)

1. Raise `retrieval_min_score` from 0.25 to 0.30 — filter weak matches
2. Add minimum text length (50+ chars) to `_context_passes_threshold` — reject noise fragments
3. Require minimum 2 contexts for grounding success instead of 1 — avoid single-weak-source answers

### Section 2: Prompt Engineering (`generator.py`)

4. Rewrite CoT prompts with explicit anti-hallucination rules:
   - Must state what is NOT covered rather than guessing
   - No inferring, extrapolating, or combining partial info
   - Only list items explicitly in evidence
   - Omit uncertain details
5. Rewrite grounding prompts with same anti-hallucination rules
6. Add "CRITICAL" warning in evidence user message about traceability
7. Require source attribution (document + page) in answers

### Section 3: Verification Strengthening (`generator.py`)

8. Rewrite verification to extract each claim and verify against evidence
9. Require claim-evidence binding in verification output
10. Strip fabricated claims from answer rather than full regeneration

### Section 4: Post-Processing (`generator.py`)

11. Validate citations — check every cited evidence ID exists and is relevant
12. Flag uncited paragraphs as potentially unsupported

## Files Changed

- `app/services/retriever.py` — retrieval thresholds, grounding logic
- `app/services/generator.py` — prompts, verification, post-processing
