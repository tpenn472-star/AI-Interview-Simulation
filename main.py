from fastapi import FastAPI

from routes.health_routes import router as health_router
from routes.interview_routes import router as interview_router


app = FastAPI(
    title="Evalify AI Interview Service",
    description="Final AI backend flow for interview simulation.",
    version="1.0",
)

app.include_router(health_router)
app.include_router(interview_router)
