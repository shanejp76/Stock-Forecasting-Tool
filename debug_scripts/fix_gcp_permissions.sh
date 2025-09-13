#!/bin/bash
# Fix Service Account Permissions for Docker Push
# Run these commands in Google Cloud Shell or with gcloud CLI

echo "Adding Artifact Registry Writer permissions..."
gcloud projects add-iam-policy-binding stock-forecasting-tool-2025 \
    --member="serviceAccount:stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

echo "Adding Storage Admin permissions for GCR legacy support..."
gcloud projects add-iam-policy-binding stock-forecasting-tool-2025 \
    --member="serviceAccount:stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

echo "Listing current permissions..."
gcloud projects get-iam-policy stock-forecasting-tool-2025 \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com"

echo "Service account permissions updated successfully!"
