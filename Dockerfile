# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any needed for pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Remove scipy if it's still in there, but assuming it was removed.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV MODULE_NAME="api.main"
ENV VARIABLE_NAME="app"
ENV PORT=8000

# Run the application
# Using shell form to allow variable expansion
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
