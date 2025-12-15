from api.main import app
# Vercel needs this variable to be named 'app'
# Set root_path to match the Vercel rewrite rule
app.root_path = "/api"
