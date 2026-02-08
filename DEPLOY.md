# Deploying Food Recognition API to DigitalOcean

## Prerequisites
- DigitalOcean account
- Domain name (optional but recommended)

## Step 1: Create a Droplet

1. Go to [DigitalOcean](https://cloud.digitalocean.com)
2. Create a new Droplet:
   - **Image**: Ubuntu 24.04 LTS
   - **Size**: **Basic** → **Regular** → **$24/mo (4GB RAM, 2 vCPUs)** minimum
     - ML models need ~2-3GB RAM
   - **Region**: Choose closest to your users
   - **Authentication**: SSH keys (recommended) or password

## Step 2: Initial Server Setup

SSH into your droplet:
```bash
ssh root@YOUR_DROPLET_IP
```

Run initial setup:
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Create app directory
mkdir -p /opt/food-api
cd /opt/food-api
```

## Step 3: Deploy the Application

### Option A: Clone from Git (Recommended)
```bash
# If your repo is on GitHub
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .
cd plate-party-mobile/backend

# Build and run
docker compose up -d --build
```

### Option B: Copy files directly
From your local machine:
```bash
# Copy backend folder to droplet
scp -r plate-party-mobile/backend/* root@YOUR_DROPLET_IP:/opt/food-api/
```

Then on the droplet:
```bash
cd /opt/food-api
docker compose up -d --build
```

## Step 4: Configure Firewall

```bash
# Allow SSH and HTTP
ufw allow 22
ufw allow 8000
ufw enable
```

## Step 5: Verify Deployment

```bash
# Check if container is running
docker ps

# Check logs
docker compose logs -f

# Test the API
curl http://localhost:8000/health
```

## Step 6: Set Up Domain & HTTPS (Recommended)

### Install Nginx & Certbot
```bash
apt install nginx certbot python3-certbot-nginx -y
```

### Configure Nginx
```bash
cat > /etc/nginx/sites-available/food-api << 'EOF'
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeout for ML inference
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;

        # Allow large image uploads
        client_max_body_size 10M;
    }
}
EOF

ln -s /etc/nginx/sites-available/food-api /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### Get SSL Certificate
```bash
certbot --nginx -d api.yourdomain.com
```

## Step 7: Update Your App

Update your mobile app config to use the new URL:

```typescript
// plate-party-mobile/utils/ai/core/config.ts
export const BACKEND_URL = process.env.BACKEND_URL || 'https://api.yourdomain.com';
```

Or without a domain:
```typescript
export const BACKEND_URL = process.env.BACKEND_URL || 'http://YOUR_DROPLET_IP:8000';
```

## Maintenance

### View logs
```bash
docker compose logs -f
```

### Restart service
```bash
docker compose restart
```

### Update deployment
```bash
git pull
docker compose up -d --build
```

### Monitor resources
```bash
docker stats
htop
```

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| Droplet (4GB) | $24 |
| Domain (optional) | ~$12/year |
| **Total** | **~$24-26/mo** |

## Scaling Tips

- **More users?** Upgrade to 8GB RAM droplet ($48/mo)
- **Global users?** Add a CDN or deploy to multiple regions
- **High traffic?** Consider DigitalOcean App Platform or Kubernetes
