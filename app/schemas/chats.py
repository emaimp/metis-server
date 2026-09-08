from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CreateChatRequest(BaseModel):
    """Payload to create a new chat. The title defaults to 'New chat'."""
    title: str | None = None


class ChatBase(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatSummary(ChatBase):
    """Summary of a chat, as returned by create/update endpoints."""


class ChatListItem(ChatBase):
    """List item, with the number of persisted messages."""
    message_count: int


class AttachmentMeta(BaseModel):
    """Metadata of a file attached to a message (the raw bytes are served separately)."""
    id: str
    filename: str
    content_type: str
    size: int


class MessageModel(BaseModel):
    id: str
    chat_id: str
    role: Literal["user", "assistant"]
    content: str
    model: str | None = None
    created_at: str
    response_time_ms: int | None = None
    attachments: list[AttachmentMeta] = []


class ChatDetailResponse(ChatBase):
    """Full chat, including its messages in chronological order."""
    messages: list[MessageModel]


class UpdateTitleRequest(BaseModel):
    title: str


class CreateMessageResponse(BaseModel):
    user_message: MessageModel
    assistant_message: MessageModel
    chat: ChatDetailResponse
