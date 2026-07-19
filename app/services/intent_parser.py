import logging

from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError

from app.models import ParsedQueryIntent
from app.utils.string_formatter_util import StringFormatterUtil


class IntentParser:
    prompt = """
        Your task is to parse queries in a book recommendation system. Your job is to:
        1. Identify specific references to book titles, author names or genres.
        2. Determine if each reference is something the user wants, doesn't want or is using as a similarity anchor
        3. Extract the remaining descriptive query (what kind of book, setting, mood, etc.)
        
        Extract the anchor terms from parts of the query where the user lists things he likes or doesn't like,
        or if he's listing things he has or hasn't read. If the user provides additional context besides the anchor terms,
        like the qualities of the book he wants, return it under the query_context_residue field.
        Skip adding generic terms like 'give me something similar', 'recommend me something like this', etc. into
        query_context_residue
        In the terms property return a list of all the anchor terms you extracted. The terms should be separate JSON
        objects with intent, item_type and item keys. 
        The intent has three possible values: 'include', 'exclude' and 'similar'.
        Use 'similar' when a user expresses positive sentiment toward a book or author (I liked books X and Y... or
        I like author A..., I read X books by author A, etc.) and use 'include' only when the user expresses
        positive intent toward future reading (I want more X... Show me Y).
        Use 'exclude' for any term with a negative connotation (I didn't like X and Y...
        or It shouldn't be like X or Y..., etc.).
        If the anchor term is an author name, item_type should be 'author', if it's a book title it should be
        'book', and if it's a genre it should be 'genre'. Also fix any obvious typos in the anchor terms (Sanderosn -> Sanderson, Mistbornn -> Mistborn).
        Do NOT change names that could be real authors, even if they're unknown to you. When in doubt, preserve the original.
        Return the user's full query in the query_full field.
        
        If there is no extra context after extracting the anchor terms, leave query_context_residue as an
        empty string. If there are no anchor terms in the query, leave terms as an empty array.
        
        Examples:
        Query: "I liked Mistborn and Stormlight, recommend something similar"
        Result:
        {
            "query_full": "I liked Mistborn and Stormlight, recommend something similar"
            "query_context_residue": "",
            "terms": [
                {"item": "Mistborn", "item_type": "book", "intent": "similar"},
                {"item": "Stormlight", "item_type": "book", "intent": "similar"}
            ]
        }
        
        Query: "Recommend me something by Brandon Sanderson but not Mistborn since I already read it"
        Result:
        {
            "query_full": "Recommend me something by Brandon Sanderson but not Mistborn since I already read it"
            "query_context_residue": "",
            "terms": [
                {"item": "Brandon Sanderson", "item_type": "author", "intent": "include"},
                {"item": "Mistborn", "item_type": "book", "intent": "exclude"}
            ]
        }
        
        Query: "cozy fantasy with a witch protagonist, no grimdark"
        Result:
        {
            "query_full": "cozy fantasy with a witch protagonist, no grimdark"
            "query_context_residue": "cozy fantasy with a witch protagonist, no grimdark",
            "terms": [
                {"item": "fantasy", "item_type": "genre", "intent": "include"},
                {"item": "grimdark", "item_type": "genre", "intent": "exclude"}
            ]
        }
        
        Query: "what should I read"
        Result:
        {
            "query_full": "what should I read
            "query_context_residue": "",
            "terms": []
        }
    """

    def __init__(self, client: OpenAI):
        self.client = client

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

        try:
            return ParsedQueryIntent.model_validate_json(
                StringFormatterUtil.remove_json_markdown(response.choices[0].message.content)
            )
        except ValidationError as e:
            logging.error(f"Error validating parsed query intent format: {e.message}")
            return None
