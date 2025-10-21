# Vercel Deployment Guide for Thumbscore.io

## 🚀 Quick Deploy Steps

### 1. Deploy to Vercel
```bash
cd "/Users/nicvenettacci/Desktop/Thumbnail Lab/thumbnail-lab"
npx vercel
```

### 2. Environment Variables to Set in Vercel Dashboard

Go to your Vercel project → Settings → Environment Variables and add:

```
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key_here
STRIPE_WEBHOOK_SECRET=whsec_...
NEXT_PUBLIC_APP_URL=https://your-app-name.vercel.app
JWT_SECRET=your-super-secret-jwt-key-change-in-production-with-random-32-chars-minimum
RESEND_API_KEY=re_your_resend_api_key_here
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-python-backend.railway.app
```

### 3. Python Backend Deployment Options

**Option A: Railway (Recommended)**
- Go to railway.app
- Connect GitHub repo
- Deploy python-service folder
- Get the Railway URL

**Option B: Render**
- Go to render.com
- Create new Web Service
- Connect GitHub repo
- Deploy python-service folder

### 4. Update Environment Variables
After getting your Python backend URL, update:
```
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-actual-backend-url
NEXT_PUBLIC_APP_URL=https://your-actual-vercel-url
```

### 5. Test Production Deployment
1. Visit your Vercel URL
2. Test thumbnail upload
3. Test Stripe checkout
4. Verify all features work

## 🎯 Ready for Launch!
