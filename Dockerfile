# --- Stage 1: Build Stage ---
# Use a slim, secure version of Python
FROM python:3.11-slim-bullseye AS builder

# Set environment variables for a clean build
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
# Copying requirements.txt first leverages Docker's layer caching for faster builds
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# --- Stage 2: Final Production Stage ---
FROM python:3.11-slim-bullseye

# Create a non-root user for better security
RUN addgroup --system app && adduser --system --group app

# Set the working directory
WORKDIR /home/app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /wheels /wheels

# Install the dependencies from the wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy the application source code
COPY ./app ./app
COPY ./run.py .

# Create directories needed by the application (e.g., for logs, sessions)
RUN mkdir -p logs sessions && chown -R app:app logs sessions

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We use the run.py script as the entrypoint
CMD ["python", "run.py"]