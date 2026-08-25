import logging

from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError
from pygments.formatters import terminal

from app.models import ParsedQueryIntent, IntentTerm, IntentType
from app.services.query_term_resolver import QueryTermResolver
from app.utils.string_formatter_util import StringFormatterUtil


class IntentParser:
    prompt = """
        Your task is to parse queries in a book recommendation system. Your job is to:
        1. Identify specific references to book titles, author names or genres.
        2. Determine if each reference is something the user wants, doesn't want or is using as a similarity anchor
        3. Extract the remaining descriptive query (what kind of book, setting, mood, etc.)
        4. Rewrite that remaining descriptive query into a longer, expanded description
        
        Extract the anchor terms from parts of the query where the user lists things he likes or doesn't like,
        or if he's listing things he has or hasn't read. If the user provides additional context besides the anchor terms,
        like the qualities of the book he wants, return it under the query_context_residue field.
        Skip adding generic terms like 'give me something similar', 'recommend me something like this', etc. into
        query_context_residue
        In the terms property return a list of all the anchor terms you extracted. The terms should be separate JSON
        objects with intent, item_type and item keys. 
        The intent has three possible values: 'include', 'exclude' and 'similar'. These are the use cases for those three:
        
        - 'include': the user wants results containing the term they named
            cues: "something by X", "something in the Y genre", "more Z"
        - 'similar': the user is naming reference points, not requirements:
            cues: "something like X", "i liked Y", "i want more books like Z"
        - 'exclude': the user doesn't want these terms showing up in results
            cues: "i didn't like X", "not like Y", "no Z"
        
        If the anchor term is an author name, item_type should be 'author', if it's a book title it should be
        'book', and if it's a genre it should be 'genre'. Also fix any obvious typos in the anchor terms (Sanderosn -> Sanderson, Mistbornn -> Mistborn).
        Do NOT change names that could be real authors, even if they're unknown to you. When in doubt, preserve the original.
        Return the user's full query in the query_full field.
        
        If there is no extra context after extracting the anchor terms, leave query_context_residue as an
        empty string. If there are no anchor terms in the query, leave terms as an empty array.

        In the query_expanded field return an expanded version of the descriptive query. This text gets embedded
        and matched against book descriptions by vector similarity, so write it the way a book blurb is written:
        "Title by Author. Description. Genres: ...". Follow these rules:

        - Expand ONLY the descriptive content, meaning whatever you put in query_context_residue when you
          extracted anchor terms, or the whole query when you extracted none.
        - Start by restating the user's descriptive request in their own terms, then elaborate on it with themes,
          tone, setting, character types and other wording a matching book's description would plausibly use.
        - Never mention the anchor terms you extracted into the terms array. Author names, book titles and genres
          are handled separately, so they must not appear in query_expanded.
        - Never name a concept the user excluded, not even to negate it. This text is matched by vector similarity
          and similarity has no notion of negation, so writing "no dystopian elements" pulls the results towards
          dystopias. Describe what the user does want instead, and leave the excluded words out entirely.
        - Keep it to 2-3 sentences and 60 words at most. Longer expansions dilute the match.
        - Do not invent specific plot points, character names, titles or authors. Describe qualities, not one book.
        - If the query has no descriptive content at all, because it is only anchor terms or because it says
          nothing concrete like "what should I read", return an empty string.

        Examples:
        Query: "Recommend me something like Brandon Sanderson's books"
        Result:
        {
            "query_full": "Recommend me something like Brandon Sanderson's books",
            "query_context_residue": "",
            "terms": [
                {"item": "Brandon Sanderson", "item_type": "author", "intent": "similar"}
            ],
            "query_expanded": ""
        }
        
        Query: "Recommend me more Brandon Sanderson books"
        Result:
        {
            "query_full": "Recommend me more Brandon Sanderson books",
            "query_context_residue": "",
            "terms": [
                {"item": "Brandon Sanderson", "item_type": "author", "intent": "include"}
            ],
            "query_expanded": ""
        }
        
        Query: "I liked Mistborn and Stormlight, recommend something similar"
        Result:
        {
            "query_full": "I liked Mistborn and Stormlight, recommend something similar",
            "query_context_residue": "",
            "terms": [
                {"item": "Mistborn", "item_type": "book", "intent": "similar"},
                {"item": "Stormlight", "item_type": "book", "intent": "similar"}
            ],
            "query_expanded": ""
        }
        
        Query: "Recommend me something by Brandon Sanderson but not Mistborn since I already read it"
        Result:
        {
            "query_full": "Recommend me something by Brandon Sanderson but not Mistborn since I already read it",
            "query_context_residue": "",
            "terms": [
                {"item": "Brandon Sanderson", "item_type": "author", "intent": "include"},
                {"item": "Mistborn", "item_type": "book", "intent": "exclude"}
            ],
            "query_expanded": ""
        }
        
        Query: "cozy fantasy with a witch protagonist, no grimdark"
        Result:
        {
            "query_full": "cozy fantasy with a witch protagonist, no grimdark",
            "query_context_residue": "cozy with a witch protagonist",
            "terms": [
                {"item": "fantasy", "item_type": "genre", "intent": "include"},
                {"item": "grimdark", "item_type": "genre", "intent": "exclude"}
            ],
            "query_expanded": "A cozy, low-stakes story following a witch as she goes about her days. Warm, gentle and character-driven, with everyday magic, herbalism and spellcraft woven into small-town village life. Comforting and hopeful in tone, full of found family, quiet friendships and slow personal growth."
        }
        
        Query: "a book about grief set on a spaceship"
        Result:
        {
            "query_full": "a book about grief set on a spaceship",
            "query_context_residue": "",
            "terms": [],
            "query_expanded": "A story about grief and loss set aboard a spaceship. Quiet, introspective and character-driven, following a small crew in the confines of a long voyage as they process mourning, memory and isolation far from home. Meditative and emotional rather than action-heavy."
        }
        
        Query: "something like Mistborn but with a heist plot and lots of political scheming"
        Result:
        {
            "query_full": "something like Mistborn but with a heist plot and lots of political scheming",
            "query_context_residue": "with a heist plot and lots of political scheming",
            "terms": [
                {"item": "Mistborn", "item_type": "book", "intent": "similar"}
            ],
            "query_expanded": "A heist story driven by political scheming. A crew of specialists plans an elaborate job against a powerful ruling class, with double-crosses, court intrigue, shifting alliances and carefully laid plans unravelling. Clever, twisty and full of conspiracy."
        }
        
        Query: "sci fi but nothing dystopian"
        Result:
        {
            "query_full": "sci fi but nothing dystopian",
            "query_context_residue": "",
            "terms": [
                {"item": "sci fi", "item_type": "genre", "intent": "include"},
                {"item": "dystopian", "item_type": "genre", "intent": "exclude"}
            ],
            "query_expanded": "A hopeful, forward-looking story of exploration and discovery, set among the stars or in a bright imagined future. Curious characters, wondrous technology and optimism about where humanity is headed."
        }
        
        Query: "what should I read"
        Result:
        {
            "query_full": "what should I read",
            "query_context_residue": "",
            "terms": [],
            "query_expanded": ""
        }
    """

    def __init__(self, client: OpenAI, term_resolver: QueryTermResolver):
        self.client = client
        self.term_resolver = term_resolver

    def parse(self, query_text: str)-> ParsedQueryIntent|None:
        messages = [
            ChatCompletionSystemMessageParam(content=self.prompt, role="system"),
            ChatCompletionUserMessageParam(content=query_text, role="user"),
        ]

        response_format = ResponseFormatJSONSchema(
            type="json_schema",
            json_schema={
                "name": "ParsedQueryIntent",
                "schema": ParsedQueryIntent.model_json_schema(),
                "strict": True,
            }
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                response_format=response_format,
                # extra_body={
                #     "models": [
                #         "google/gemma-4-26b-a4b-it:free",
                #         "qwen/qwen3-next-80b-a3b-instruct:free",
                #         "openai/gpt-oss-120b:free"
                #     ],
                #     "reasoning": {"enabled": False}
                # }
            )
        except APIError as e:
            logging.error(f"Error parsing query intent: {e.message}")
            return None

        response_content = response.choices[0].message.content
        if response_content is None:
            logging.error(f"Got empty response from chat completion call, skipping intent parsing")
            return None

        try:
            parsed_intent = ParsedQueryIntent.model_validate_json(
                StringFormatterUtil.remove_json_markdown(response_content)
            )
        except ValidationError as e:
            logging.error(f"Error validating parsed query intent format: {str(e)}")
            return None

        resolved_terms, unresolved_terms = [], []
        for term in parsed_intent.terms:
            resolved = self._resolve_query_intent_term(term)

            if resolved is not None:
                resolved_terms.append(resolved)
            elif term.intent is not IntentType.EXCLUDE:
                # unresolved terms fall back to being embedded as text, so excluded ones have to be dropped
                # instead, otherwise "no grimdark" ends up pulling the query vector towards grimdark
                unresolved_terms.append(term.item)

        return parsed_intent.model_copy(update={
            "query_context_residue": self._append_terms(parsed_intent.query_context_residue, unresolved_terms),
            "query_expanded": self._append_terms(parsed_intent.query_expanded, unresolved_terms),
            "terms": resolved_terms
        })

    def _append_terms(self, text: str, terms: list[str]) -> str:
        return " ".join(part for part in [text, *terms] if part)

    def _resolve_query_intent_term(self, term: IntentTerm) -> IntentTerm|None:
        resolved_term = self.term_resolver.resolve(term.item_normalized, term.item_type)

        if resolved_term is None:
            return None

        return term.model_copy(update={"item": resolved_term})
