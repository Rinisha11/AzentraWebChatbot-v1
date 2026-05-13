# Client Deployment Template

Use this folder as the starting point for one isolated client deployment.

## How to use it

1. Copy `client-template` to a new folder, for example `deploy/client-acme`.
2. Copy `.env.example` to `.env`.
3. Update `.env` with the client's OpenAI key, branding, origins, and paths.
4. Update `sites.json` with the client's site config.
5. Put that client's source documents into `data/default-site/`.
6. Build the vector store for that client.
7. Start the backend with Docker Compose.

## Folder purpose

- `.env`: runtime secrets and deployment settings for this client only
- `sites.json`: backend site config for this client deployment
- `data/`: raw documents for ingestion
- `chroma/`: persisted vector store for this client
- `logs/`: backend logs for this client

## Ingest this client's documents

Run from `chatbot-backend/`:

```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:DATA_DIR="../deploy/client-acme/data/default-site"
$env:CHROMA_DIR="../deploy/client-acme/chroma/default-site"
python ingest.py
```

## Start this client backend

Run from the client deployment folder:

```powershell
docker compose up --build -d
```

## Package files for a server

Run from the repo root:

```powershell
pwsh ./scripts/package-deploy.ps1 -ClientName client-template
```

That creates:

- `release/client-template-package/`
- `release/client-template-deploy.zip`

Upload:

- `chatbot-backend/`
- `deploy/client-template/`

Host these website assets from the generated package:

- `widget/chatbot-widget.iife.js`
- `widget/chatbot-widget.css`

## Point the website widget to this backend

Use the client's public backend URL in the widget config:

```html
<link rel="stylesheet" href="/assets/chatbot-widget.css" />
<div
  data-chatbot-widget
  data-socket-base-url="wss://chat.client-acme.com/ws"
  data-site-id="default-site"
></div>
<script src="/assets/chatbot-widget.iife.js"></script>
```

## Isolation model

This template is intentionally one client per deployment:

- separate env
- separate vector store
- separate logs
- separate origin allowlist
- separate backend process/container

That keeps one client's data and failures from affecting another client's chatbot.
