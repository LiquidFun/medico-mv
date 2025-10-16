from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List
from datetime import datetime

from app.models import User, Conversation, ChatMessage, get_db
from app.services import get_current_user

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse]


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: ConversationCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation."""
    new_conversation = Conversation(
        user_id=user.id,
        title=conversation_data.title,
    )

    db.add(new_conversation)
    await db.commit()
    await db.refresh(new_conversation)

    return ConversationResponse.model_validate(new_conversation)


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all conversations for the current user (excludes empty conversations)."""
    # Get all conversations for the user
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(desc(Conversation.updated_at))
    )
    conversations = result.scalars().all()

    # Filter out empty conversations (those with no messages)
    non_empty_conversations = []
    for conv in conversations:
        msg_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conv.id)
            .limit(1)
        )
        if msg_result.scalar_one_or_none():
            non_empty_conversations.append(conv)

    return [ConversationResponse.model_validate(conv) for conv in non_empty_conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific conversation with all messages."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Get messages
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
    )
    messages = messages_result.scalars().all()

    return {
        **ConversationResponse.model_validate(conversation).model_dump(),
        "messages": [MessageResponse.model_validate(msg) for msg in messages],
    }


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Delete all messages in the conversation
    messages_result = await db.execute(
        select(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
    )
    messages = messages_result.scalars().all()
    for msg in messages:
        await db.delete(msg)

    # Delete the conversation
    await db.delete(conversation)
    await db.commit()
