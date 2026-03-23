#!/bin/bash
# Decode the Kalshi private key from base64 env variable and write to file
if [ -n "$KALSHI_PRIVATE_KEY_BASE64" ]; then
  echo "$KALSHI_PRIVATE_KEY_BASE64" | base64 -d > kalshi_private_key.pem
    echo "Private key written from env variable."
    fi
    # Initialize database
    python -m src.utils.database
    # Run the bot in paper trading mode by default
    python cli.py run --paper
