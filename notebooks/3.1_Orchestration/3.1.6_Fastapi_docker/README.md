# Train the model
python train_model.py

# Build the Docker image
docker build -t wine-classification-api .

# Run the Docker container
docker run -p 8000:8000 wine-classification-api

# Test the API
curl -Method Post -Uri "http://localhost:8000/predict" -Headers @{ "Content-Type" = "application/json" } -Body '{"features": [13.2, 2.7, 2.36, 21, 100, 2.98, 3.15, 0.22, 2.26, 6.5, 1.05, 3.33, 820]}'

