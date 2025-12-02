#!/bin/bash

# Stop on error
set -e

# ---- User settings ----
PROJECT_ID="streamlit-tigusa"
REGION="us-central1"
REPO="streamlit-repo"
SERVICE="stock-and-flow"
IMAGE="stock-and-flow"
# ------------------------

TAG=$(date +%s)
IMAGE_PATH="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG"

echo "-----------------------------------------"
echo " Building Docker image:"
echo "   $IMAGE_PATH"
echo "-----------------------------------------"

gcloud builds submit \
  --project=$PROJECT_ID \
  --tag $IMAGE_PATH .

echo "-----------------------------------------"
echo " Deploying Cloud Run service:"
echo "   Service: $SERVICE"
echo "   Image:   $IMAGE_PATH"
echo "-----------------------------------------"

gcloud run deploy $SERVICE \
  --image $IMAGE_PATH \
  --region=$REGION \
  --project=$PROJECT_ID \
  --platform=managed \
  --allow-unauthenticated

echo "-----------------------------------------"
echo " Deployment complete!"
echo " Cloud Run service URL:"
gcloud run services describe $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)'
echo "-----------------------------------------"