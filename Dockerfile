FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow==12.3.0

COPY backend/app ./backend/app
COPY outputs/checkpoints/mobilenet_v3_large.pth ./outputs/checkpoints/mobilenet_v3_large.pth

EXPOSE 8080

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
