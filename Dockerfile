# Use a base image with Python
FROM python:3.12.6

# Install ffmpeg and other required packages
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Create and set the working directory
WORKDIR /app

# Copy your app code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port expected by Hugging Face (7860)
EXPOSE 7860

# Run the Flask app
CMD ["python", "demo/app.py"]
