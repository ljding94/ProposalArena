# system prompt


# user prompt
Please evaluate and compare the following two proposals:

Proposal A ({proposal_IPTS_a}):
{proposal_text_a}


Proposal B ({proposal_IPTS_b}):
{proposal_text_b}


Respond only with valid JSON in this exact structure (no additional text outside the JSON):
{
  "comparison_table": "[Markdown table summarizing aspects vs. Proposal A vs. Proposal B]",
  "recommendation": "[Detailed paragraph(s) explaining which is better and why]"
  "winner": ["A" or "B" or "tie"],
}