"""AgentCard <-> A2A JSON wire format.

The official ``a2a.types.AgentCard`` is a protobuf message; its JSON wire
format (camelCase, per A2A v1.0) is produced/consumed with the protobuf JSON
mapping. ``google.protobuf`` ships with ``a2a-sdk`` — no extra dependency.
"""

from __future__ import annotations

from typing import cast

from a2a.types import AgentCard
from google.protobuf.json_format import MessageToDict, ParseDict

from agentplane_core.types import JsonObject


def agent_card_to_json_dict(card: AgentCard) -> JsonObject:
    """Serialize an AgentCard to its A2A JSON object form."""
    return cast(JsonObject, MessageToDict(card))


def agent_card_from_dict(data: JsonObject) -> AgentCard:
    """Parse an A2A JSON card into the official protobuf AgentCard.

    Unknown fields (e.g. v0.3 compatibility fields some servers still emit)
    are ignored.
    """
    card = AgentCard()
    ParseDict(data, card, ignore_unknown_fields=True)
    return card


__all__ = ["agent_card_from_dict", "agent_card_to_json_dict"]
