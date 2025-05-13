from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from squat_analysis import process_squat_video
import os

app = FastAPI()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    report_path = process_squat_video(file_path)
    return FileResponse(report_path, media_type='application/pdf', filename="Squat_Report.pdf")
