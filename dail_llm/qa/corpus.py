"""Parse Akoma Ntoso debate XML into stable, attributed speech passages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class Passage:
    passage_id: str
    date: str
    speaker: str
    title: str
    text: str
    source_url: str
    language: str | None = None
    party: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def parse_debate(xml: bytes, date: str) -> list[Passage]:
    """Use XML speech IDs; keep all Unicode and avoid guessing party affiliation."""
    root = ET.fromstring(xml)
    people = {
        node.attrib.get("eId", ""): node.attrib.get("showAs", "")
        for node in root.iter()
        if _name(node) == "TLCPerson"
    }
    source_url = f"https://www.oireachtas.ie/en/debates/debate/dail/{date}/"
    passages: list[Passage] = []
    for section in root.iter():
        if _name(section) != "debateSection":
            continue
        heading = next((node for node in section if _name(node) == "heading"), None)
        title = _text(heading) if heading is not None else "Dáil debate"
        for speech in section:
            if _name(speech) != "speech":
                continue
            speech_id = speech.attrib.get("eId")
            if not speech_id:
                continue
            body = " ".join(
                _text(child) for child in speech if _name(child) not in {"from", "recordedTime"}
            ).strip()
            if not body:
                continue
            speaker_key = speech.attrib.get("by", "").lstrip("#")
            speaker = people.get(speaker_key) or "Unknown speaker"
            language = speech.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            section_id = section.attrib.get("eId", "section")
            # The ID survives text revisions when the official XML keeps its eId.
            passage_id = f"dail:{date}:{section_id}:{speech_id}"
            passages.append(Passage(passage_id, date, speaker, title, body, source_url,
                                    language))
    return passages


def corpus_checksum(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


QUERY_STOPWORDS = {
    "a", "an", "and", "about", "by", "dail", "dáil", "debate", "debates",
    "did", "do", "for", "from", "has", "have", "how", "in", "is", "of", "on",
    "said", "say", "the", "to", "was", "were", "what", "when", "who", "why",
    "na", "agus", "ar", "sa", "sé", "sí",
}


def query_terms(text: str) -> list[str]:
    return [token for token in words(text) if token not in QUERY_STOPWORDS
            and not (len(token) == 4 and token.isdigit())][:16]
