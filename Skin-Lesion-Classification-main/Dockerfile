# Backend container: Node 18 + Python 3 + PyTorch (CPU)
FROM node:18-bullseye

# Install Python & pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
# Use PyTorch CPU wheels to avoid CUDA
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir -r requirements.txt

# --- Node deps ---
COPY package*.json ./
RUN npm ci --omit=dev

# --- App code ---
COPY . .

ENV NODE_ENV=production
ENV PORT=3001
EXPOSE 3001

# Healthcheck (optional): uncomment if you add /health route
# HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
#   CMD curl -fsS http://localhost:3001/health || exit 1

CMD ["node", "server.js"]
