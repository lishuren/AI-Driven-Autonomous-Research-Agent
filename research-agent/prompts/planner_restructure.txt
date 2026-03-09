You are a research planner reviewing a research graph that has gaps.

Current research outline:
{graph_outline}

Identified gaps or issues:
{gaps}
{user_context}

Suggest changes to improve coverage. You can:
- Add new sub-topics under existing parents.
- Note which areas need more depth.

Respond ONLY with valid JSON:
[
  {{"action": "add", "parent_name": "<existing parent topic name>", "name": "<new sub-topic>", "query": "<2-6 word query>", "priority": <1-10>}},
  ...
]
