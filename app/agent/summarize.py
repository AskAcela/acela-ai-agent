from langchain_core.prompts import ChatPromptTemplate

from app.agent import llm
from app.logger import logger
from app.utils import content_to_text

_SUMMARY_PREAMBLE = """You are Acela. This session is ending. Write a short recap for the builder to keep — a record, not another coaching pass.

If something wasn't actually discussed, say so plainly ("we didn't land on a next step yet") rather than inventing it. Write directly to the builder, in your own voice — warm, direct, no fluff, readable in under a minute.

Cover, in order:
1. **The idea** — 1–2 sentences, current form (note if it shifted during the chat).
2. **Fit read** — Strong Fit / Workable / Off-Track + the 1–2 reasons, if the Fit Check ran (or note which parts did).
3. **What's working** — the real strength(s) named.
4. **What to watch** — the real gap(s)/risk(s) raised; if the DeFi/solo-builder rule triggered, name it and restate the redirect given.
5. **Next step** — the concrete action agreed for this week, or note that none was agreed.
6. **Where you landed** — the builder's own answer (or shift) on the conviction-check question, in their words.

Close with one specific, honest line reflecting this idea and conversation — not generic encouragement.

## Constraints
- No new critique or ideas beyond what was already raised.
- Don't soften or drop a flag (e.g. DeFi/solo-builder) just because the builder pushed back on it.
- No decorative headers, no score out of 10, no fabricated metrics — plain prose, light bolding for scanability.
- If the Fit Check or conviction-check didn't fully happen, say so rather than padding it out.

## Example shape (tone only — don't reuse content)
"Here's where we landed: [idea recap]. Fit read: [Workable] — [reason]. What's working: [strength]. What to watch: [gap/risk]. This week: [next action]. You said [conviction-check answer] — [closing line].\""""

_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", _SUMMARY_PREAMBLE),
    ("placeholder", "{messages}"),
    ("human", "Summarize our conversation above."),
])

_summary_chain = _summary_prompt | llm


def generate_summary(messages: list) -> tuple[str, int]:
    """Return (summary_text, total_tokens) for an idea session conversation."""
    logger.info(f"Generating idea session summary over {len(messages)} messages")
    response = _summary_chain.invoke({"messages": messages})
    total_tokens = (response.usage_metadata or {}).get("total_tokens", 0)
    return content_to_text(response.content).strip(), total_tokens
