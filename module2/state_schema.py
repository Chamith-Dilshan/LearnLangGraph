
""" When we define a LangGraph StateGraph, we use a state schema.
The state schema represents the structure and types of data that our graph will use.
All nodes are expected to communicate with that schema.

You can use TypedDict, Dataclass, or Pydantic to define the state schema.

TypedDict -> For type hints.They can be used by static type checkers or IDEs to catch potential
type-related errors before the code is run. But they are not enforced at runtime!
We can access the state values like this : state['name']

Dataclass-> A dataclass is a special kind of class that is designed to hold data.
It provides a convenient way to define classes that are primarily used to store data,
and it automatically generates special methods like __init__, __repr__, and __eq__ for you.
We use state.name for the dataclass state rather than state["name"]

Pydantic -> TypedDict and dataclasses provide type hints, but they don't enforce types at runtime.
Pydantic is a data validation and settings management library using Python type annotations.
It's particularly well-suited for defining state schemas in LangGraph due to its validation capabilities.
Pydantic can perform validation to check whether data conforms to the specified types and constraints at runtime.

"""
from dataclasses import dataclass
from typing import TypedDict, Literal

from pydantic import BaseModel, field_validator, ValidationError


class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy","sad"]


@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]


class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad"

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

if __name__ == "__main__":
    try:
        state = PydanticState(name="John Doe", mood="mad")
    except ValidationError as e:
        print("Validation Error:", e)