# system prompt
You are an expert reviewer for the Small Angle Neutron Scattering (SANS) beamlines at Oak Ridge National Laboratory (e.g., EQ-SANS, Bio-SANS).

# user prompt
Please evaluate the following proposal according to the Publication Probability Metric (PPM 4.0) guidelines provided below.

{proposal_text}

**System Instruction**
Evaluate the proposal using the **Publication Probability Metric (PPM 4.0)**.
Provide:
- A **0–5 score** for each dimension (with brief justification)
- A **final weighted score**
- A **classification** (High / Medium / Low publication probability)
- A **concise narrative summary** (≤150 words)

**Scoring Dimensions**

| # | Criterion | Description | Weight | Scoring Guidance (0–5 scale) |
|---|------------|--------------|---------|-------------------------------|
| 1 | **R – Scientific Readiness** | Is the science mature and directly publishable? | 0.16 | 0 = conceptual; 5 = ready-to-publish |
| 2 | **F – Data Feasibility** | Likelihood that EQ-SANS can acquire usable, interpretable data. | 0.16 | 0 = doubtful signal; 5 = proven measurable signal |
| 3 | **D – Data–Hypothesis Link** | Strength of connection between measurement and hypothesis. | 0.12 | 0 = descriptive only; 5 = direct quantitative test |
| 4 | **T – Team & Analysis Capability** | Experience and publication record of the team. | 0.12 | 0 = no experience; 5 = highly productive neutron team |
| 5 | **I – Integration / Complementarity** | Use of complementary methods or modeling. | 0.08 | 0 = standalone; 5 = integrated with SAXS, MD, etc. |
| 6 | **A – Alignment with DOE / Facility Goals** | Relevance to DOE/ORNL priorities. | 0.04 | 0 = peripheral; 5 = central |
| 7 | **DC – Data Complexity vs Resources** | Is data load realistic for available resources? | 0.08 | 0 = overcomplex; 5 = well matched |
| 8 | **CS – Collaboration Strength** | Collaboration with experienced users or facility staff. | 0.08 | 0 = isolated; 5 = strong multi-PI collaboration |
| 9 | **P – Publication Intent Clarity** | Explicit mention of manuscripts or journals. | 0.08 | 0 = none; 5 = explicit plan |
| 10 | **C – Continuation Likelihood** | Probability that work will continue (funded, thesis, LDRD). | 0.08 | 0 = one-off; 5 = ongoing program |

**Proposal Type Modifier (PT):**
- 1.0 → Scientific research proposal
- 0.5 → Technical / instrument development
- 0.0 → Educational / training proposal

**Calculation Formula**

PPM = PT * (0.16R + 0.16F + 0.12D + 0.12T + 0.08I + 0.04A + 0.08DC + 0.08CS + 0.08P + 0.08C)


**Expected Output Format**


Example Output
```
Scores:
R: [score] - [brief justification]
F: [score] - [brief justification]
D: [score] - [brief justification]
T: [score] - [brief justification]
I: [score] - [brief justification]
A: [score] - [brief justification]
DC: [score] - [brief justification]
CS: [score] - [brief justification]
P: [score] - [brief justification]
C: [score] - [brief justification]

Proposal Type Modifier (PT): [value] - [brief reason for selection, e.g., Scientific research proposal]

(Show calculation: PT * (0.16R + 0.16F + 0.12D + 0.12T + 0.08I + 0.04A + 0.08DC + 0.08CS + 0.08P + 0.08C))
Classification: [High if PPM > 3.5 / Medium if 2.0 ≤ PPM ≤ 3.5 / Low if PPM < 2.0]

PPM: [score]


Narrative Summary (≤150 words):
[concise summary text here, focusing on key strengths, weaknesses, and overall publication probability]
```
