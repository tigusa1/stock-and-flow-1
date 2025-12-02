#!/bin/bash
# ------------------------------------------------------------
# Apply stable Streamlit-friendly Cloud Run configuration
# Reusable for ANY Cloud Run service
# ------------------------------------------------------------

# --------- USER SETTINGS (EDIT THESE FOR EACH PROJECT) ---------
PROJECT_ID="streamlit-tigusa"
REGION="us-central1"
SERVICE="stock-and-flow"
# ---------------------------------------------------------------

echo "---------------------------------------------------------"
echo "Applying stable configuration to Cloud Run service: $SERVICE"
echo "Project: $PROJECT_ID | Region: $REGION"
echo "---------------------------------------------------------"

gcloud run services update $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --min-instances=1 \
  --concurrency=100 \
  --timeout=3600 \
  --cpu-boost

echo "---------------------------------------------------------"
echo "Done!"
echo "Current configuration:"
gcloud run services describe $SERVICE --region=$REGION --project=$PROJECT_ID \
  --format="table(spec.template.spec.containerConcurrency, spec.template.scaling.minInstanceCount, spec.template.timeoutSeconds, spec.template.spec.containers[].resources.limits.cpu, spec.template.spec.containers[].resources.limits.memory)"
echo "---------------------------------------------------------"