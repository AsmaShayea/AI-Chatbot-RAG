FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the Flask port
EXPOSE 8000

# Run the Flask application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "run:app"]
