# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all source files to the container
COPY ./app ./app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r app/requirements.txt

# Expose the default FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
