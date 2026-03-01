#!/bin/bash
# VaaniDub One-Click Deploy Script
# Run: bash deploy.sh

set -e
echo "🚀 Deploying VaaniDub to Vercel..."
cd "$(dirname "$0")/frontend"

# Check if vercel is available
if ! command -v vercel &>/dev/null && ! command -v npx &>/dev/null; then
    echo "❌ Install Node.js first: https://nodejs.org"
    exit 1
fi

# Deploy
npx --yes vercel --yes --prod

echo ""
echo "✅ VaaniDub is LIVE!"
