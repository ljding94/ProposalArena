20251017with grok

Let's discuss the perspective on using LLMs to review user proposal at large facility at national lab, I have a rubric that been used to guide external human reviewer to score the proposals, and I also the score data for past proposal, each cycle has about 20 proposals been approved, more importantly, I have the number of publication data for each proposals, basically, we don't really care about the absolute the score of each proposal, but more about the ranking, so ideally, the proposal with more publication should have been ranked higher. with this metric in mind, I have the following two thing wanna try:
 

1. by giving the LLM rubric, and ask it the score the proposals, what's kind of ranking does it results, will it be better than human reviewer or worse, compare to ideal ranking.
2. instead of using one pass scoring, we can ask the LLM to compare two proposals at a time, and ask it which one is better, and then we can use the pairwise comparison result to generate a ranking using Elo method, again, compare to ideal ranking.

what do you think about this idea? any suggestion on how to improve it? and what is the feasibility, do you think I should use a agent framework like CrewAI, or since it's relatively simple task, just use basic prompting is good enough?