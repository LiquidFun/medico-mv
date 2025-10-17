from dotenv import load_dotenv
load_dotenv()  # noqa

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
from sqlalchemy import select

from app.models import init_db, User, get_db
from app.routes import auth_router, chat_router, ws_router, tools_router, documents_router
from app.services.auth import get_password_hash



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_db()

    # Create default admin user if it doesn't exist
    async for db in get_db():
        try:
            result = await db.execute(select(User).where(User.username == "admin"))
            admin_user = result.scalar_one_or_none()

            if not admin_user:
                admin_user = User(
                    username="admin",
                    email="admin@example.com",
                    hashed_password=get_password_hash(os.getenv("ADMIN_PASSWORD", "password")),
                    display_name="Ulrike Schlüter"
                )
                db.add(admin_user)
                await db.commit()
                print("✓ Default admin user created (username: admin, password: password)")
            else:
                # Update display name if not set
                if not admin_user.display_name:
                    admin_user.display_name = "Ulrike Schlüter"
                    await db.commit()
                print("✓ Admin user already exists")
        except Exception as e:
            print(f"Error creating admin user: {e}")
        break

    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="AI Chat Application",
    description="A modern AI chat application with authentication and chat history",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(ws_router)
app.include_router(tools_router)
app.include_router(documents_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("templates/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOSTNAME", "0.0.0.0"),
        port=8000,
        reload=True,
    )
