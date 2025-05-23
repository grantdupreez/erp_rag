# Streamlit Secrets Configuration

Add this exact configuration to your Streamlit Cloud app secrets:

```toml
# Qdrant Cloud Configuration
QDRANT_URL = "https://your-cluster-id.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key-here"

# OpenAI Configuration
OPENAI_API_KEY = "sk-your-openai-api-key-here"

# User Authentication
[passwords]
admin = "admin123"  # ⚠️ Change this password! Admin has special privileges
user1 = "password1"
user2 = "password2"
```

## Important Notes:

1. **QDRANT_URL Format**: Make sure your URL includes:
   - The protocol: `https://`
   - The full domain including region
   - The port: `:6333`

2. **QDRANT_API_KEY**: Get this from your Qdrant Cloud dashboard
   - Go to your cluster
   - Click on "API Keys"
   - Copy the key

3. **Admin User**: The user with username "admin" has special privileges:
   - Can view all users' document counts
   - Can delete and recreate the entire collection
   - Can view detailed collection statistics
   - Can test the Qdrant connection

4. **No quotes around values in Streamlit Cloud**: When adding secrets in the Streamlit Cloud UI, don't add extra quotes around the values.

5. **Test Connection**: After updating secrets, restart your app and check if it connects properly.
