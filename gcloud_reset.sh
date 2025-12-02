#!/bin/bash
set -e

# ------------------------------------------------------------
# CLEAN RESET + BUILD + DEPLOY + CONFIGURE
# For Streamlit on Google Cloud Run
# ------------------------------------------------------------

# ----------- USER CONFIGURE THESE -----------
PROJECT_ID="streamlit-tigusa"
REGION="us-central1"
REPO="streamlit-repo"
SERVICE="stock-and-flow"
IMAGE="stock-and-flow"
# -------------------------------------------


echo "------------------------------------------------------------"
echo " STEP 1: Deleting existing Cloud Run service: $SERVICE"
echo "------------------------------------------------------------"

gcloud run services delete $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --quiet || true

echo "Service deleted (or did not exist)."


echo "------------------------------------------------------------"
echo " STEP 2: Building new Docker image"
echo "------------------------------------------------------------"

TAG=$(date +%s)

gcloud builds submit \
  --project=$PROJECT_ID \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG .

echo "Build complete. Image tag: $TAG"


echo "------------------------------------------------------------"
echo " STEP 3: Deploying fresh Cloud Run service"
echo "------------------------------------------------------------"

gcloud run deploy $SERVICE \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:$TAG \
  --region=$REGION \
  --project=$PROJECT_ID \
  --allow-unauthenticated \
  --platform=managed

echo "Service deployed."


echo "------------------------------------------------------------"
echo " STEP 4: Applying stable configuration"
echo "   min-instances=1"
echo "   concurrency=10"
echo "   timeout=900"
echo "   cpu-boost=ON"
echo "------------------------------------------------------------"

gcloud run services update $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --min-instances=1 \
  --concurrency=10 \
  --timeout=900 \
  --cpu-boost

echo "Configuration applied."


echo "------------------------------------------------------------"
echo " STEP 5: Final service URL"
echo "------------------------------------------------------------"

gcloud run services describe $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format="value(status.url)"

echo "------------------------------------------------------------"
echo " CLEAN RESET COMPLETE!"
echo "------------------------------------------------------------"