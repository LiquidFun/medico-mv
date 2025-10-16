from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
import json
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.models import User, Conversation, ChatMessage, get_db
from app.services import LLMService
from app.services.auth import SECRET_KEY, ALGORITHM
from app.services.rag_client import RAGClient
from app.services.tools import ToolRegistry

router = APIRouter()

llm_service = LLMService()
rag_client = RAGClient()
tool_registry = ToolRegistry()
executor = ThreadPoolExecutor(max_workers=4)


async def get_user_from_token(token: str, db: AsyncSession) -> User:
    """Authenticate user from WebSocket token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None

    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


@router.websocket("/ws/chat/{conversation_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    conversation_id: int,
    token: str = Query(...),
):
    """WebSocket endpoint for streaming chat messages."""
    await websocket.accept()

    # Get database session
    async for db in get_db():
        try:
            # Authenticate user
            user = await get_user_from_token(token, db)
            if not user:
                await websocket.send_json({"error": "Authentication failed"})
                await websocket.close()
                return

            # Verify conversation belongs to user
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user.id,
                )
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                await websocket.send_json({"error": "Conversation not found"})
                await websocket.close()
                return

            # Main chat loop
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    user_message = message_data.get("message", "")

                    if not user_message.strip():
                        continue

                    # Save user message to database
                    user_chat_message = ChatMessage(
                        conversation_id=conversation_id,
                        role="user",
                        content=user_message,
                    )
                    db.add(user_chat_message)
                    await db.commit()

                    # Get conversation history
                    history_result = await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.conversation_id == conversation_id)
                        .order_by(ChatMessage.created_at)
                    )
                    messages = history_result.scalars().all()

                    # Update conversation title based on first message
                    if len(messages) == 1 and conversation.title == "New Conversation":
                        # Generate title from first user message
                        title = user_message[:50]
                        if len(user_message) > 50:
                            title += "..."
                        conversation.title = title
                        await db.commit()

                    # Search for relevant context from RAG
                    context_chunks = await rag_client.search(user_message, top_k=3)

                    # Format messages for LLM
                    llm_messages = [
                        {"role": msg.role, "content": msg.content}
                        for msg in messages
                    ]

                    # Inject RAG context if available
                    if context_chunks:
                        context_text = "\n\n".join([
                            f"[Source: {chunk['metadata'].get('filename', 'Unknown')}]\n{chunk['text']}"
                            for chunk in context_chunks
                        ])
                        system_message = {
                            "role": "system",
                            "content": f"Use the following context to help answer the user's question. If the context is relevant, reference it in your answer:\n\n{context_text}"
                        }
                        llm_messages.insert(0, system_message)

                    # Stream response from LLM with tool support
                    assistant_response = ""
                    tool_execution_task = None
                    await websocket.send_json({"type": "start"})

                    async for chunk in llm_service.stream_chat_completion(
                        llm_messages,
                        tools=tool_registry.get_tool_definitions()
                    ):
                        print(f"DEBUG: Received chunk: {chunk}")  # Debug logging
                        if chunk["type"] == "content":
                            assistant_response += chunk["content"]
                            await websocket.send_json({
                                "type": "chunk",
                                "content": chunk["content"],
                            })
                        elif chunk["type"] == "tool_start":
                            # Forward tool_start immediately to frontend
                            print(f"DEBUG: Tool start detected: {chunk['tool_name']}")
                            await websocket.send_json({
                                "type": "tool_start",
                                "tool_name": chunk["tool_name"],
                            })
                            print(f"DEBUG: Sent tool_start message to frontend")
                        elif chunk["type"] == "tool_call":
                            # Now we have complete arguments, execute the tool
                            tool_name = chunk["tool_name"]
                            arguments = chunk["arguments"]
                            print(f"DEBUG: Tool call with complete args: {tool_name}")

                            # Start tool execution as a background task
                            async def execute_tool_async():
                                print(f"DEBUG: Executing tool {tool_name} with args: {arguments}")
                                try:
                                    loop = asyncio.get_event_loop()
                                    tool_result = await loop.run_in_executor(
                                        executor,
                                        tool_registry.execute_tool,
                                        tool_name,
                                        arguments
                                    )
                                    print(f"DEBUG: Tool result length: {len(tool_result)}")

                                    # Send the tool result as HTML
                                    await websocket.send_json({
                                        "type": "tool_result",
                                        "tool_name": tool_name,
                                        "html": tool_result,
                                    })

                                    # Append to assistant response for storage
                                    return f"\n\n[Tool: {tool_name}]\n{json.dumps(arguments)}"

                                except Exception as e:
                                    print(f"DEBUG: Tool execution error: {str(e)}")
                                    await websocket.send_json({
                                        "type": "chunk",
                                        "content": f"\n\n[Error executing tool: {str(e)}]"
                                    })
                                    return ""

                            tool_execution_task = asyncio.create_task(execute_tool_async())

                    # Wait for any pending tool execution to complete
                    if tool_execution_task:
                        tool_response = await tool_execution_task
                        assistant_response += tool_response

                    print(f"DEBUG: Final assistant_response: '{assistant_response}'")
                    await websocket.send_json({"type": "end"})

                    # Save assistant response to database
                    assistant_message = ChatMessage(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_response,
                    )
                    db.add(assistant_message)

                    # Update conversation timestamp
                    conversation.updated_at = datetime.now(timezone.utc)
                    await db.commit()

                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    break

        except Exception as e:
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass
