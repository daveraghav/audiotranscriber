import os
from langchain_together import Together
import together
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

# load env vars
from dotenv import load_dotenv
load_dotenv()

DEFAULT_LLM = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LLM_MODEL=os.environ.get("LLM_MODEL", DEFAULT_LLM)
    
llm = Together(model=LLM_MODEL, max_tokens=5000)

# Pydantic
class TranscriptAnalysis(BaseModel):
    title: str = Field(description="A short, descriptive title for the discussion topic.")
    summary: str = Field(description="A concise, 2-3 line summary of the main discussion points.")
    keywords: List[str] = Field(description="3-5 relevant keywords that summarise the discussion topics.")
    speaker_enriched_transcript: str = Field(
        description=(
            "A markdown formatted string of the full chronological conversation after speaker diarization. "
            "**Absolutely no modification, summarisation, or rephrasing of the original transcript content is allowed.** "
            "Each speaker's turn must follow this precise pattern:\n"
            "**<Speaker Name>:** <Their exact, verbatim utterance from the transcript>\n\n"
            "**Crucially, every speaker's full utterance must be followed by two newline characters (\\n\\n) before the start of the next speaker's name.** "
            "This creates a distinct paragraph break between consecutive speaker turns. "
            "If a speaker's utterance itself contains newlines, these should be preserved within their utterance, but the segment between speaker turns *must* be `\\n\\n`."
            "\n\nExample transcript for schema understanding only:\n**Speaker_1:** Hi there this is your host Speaker_1, I'm here with Speaker_2. Hi Speaker_2 welcome to the show, how are you?\n\n**Speaker_2:** I'm doing good, how are you?\n\n**Speaker_1:** Great, thanks for asking! So, tell us about your latest project.\n\n"
            "The content of the utterance must be the raw, verbatim text from the original transcript, exactly as provided."
            "Hence, the output of speaker enriched transcript and raw transcript must be like-for-like, except for the speaker enrichment and specified paragraph breaks."
            "If there is only one speaker detected, follow this format: **<Speaker Name>:** <Their exact, verbatim utterance from the transcript>\n\n"
        )
    )

parser = PydanticOutputParser(pydantic_object=TranscriptAnalysis)

template = """
You are an expert speaker diarization model and transcript analyst. Your task is to process the given transcript, with focus on speaker change detection based on conversational style and context (act as a replacement of pyannote) and extract requested transcript information as a JSON object.

Strictly adhere to the following output format. Your response MUST be a JSON object, and ONLY the JSON object. Do NOT wrap it in markdown code blocks (e.g., ```json ... ```) or any other conversational text.

{format_instructions}

Transcript Text:
{transcript_text}
"""

prompt = ChatPromptTemplate.from_template(template).partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | llm | parser

def enrich_with_llm(transcript):
    try:
        print(f"Attempting AI Enrichment using {LLM_MODEL}")
        return chain.invoke({"transcript_text": transcript}).model_dump()
    except Exception as e:
        print(f"Failed AI Enrichment with the following Exception {e}")
        return None
    
client = together.Together(timeout=20)
def enrich_with_llm_together(transcript):
    try:
        print(f"Attempting AI Enrichment using {LLM_MODEL}")
        extract = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "The following is a transcript. Only answer in JSON.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ],
        model=LLM_MODEL,
        response_format={
            "type": "json_object",
            "schema": TranscriptAnalysis.model_json_schema(),
        },
        )

        output = json.loads(extract.choices[0].message.content)
        return output
    except Exception as e:
        print(f"Failed AI Enrichment with the following Exception {e}")
        return None
