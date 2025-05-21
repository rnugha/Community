from enum import Enum
from dataclasses import dataclass
from typing import List
from langchain.schema import BaseMessage

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

from typing import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str