# Railway Deployment Guide

## 🚀 Deploy Python Backend to Railway

### 1. Environment Variables for Railway Dashboard

Go to your Railway project → Variables and add these:

```
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.your-service-role-key-here

RESEND_API_KEY=re_your_resend_api_key_here

JWT_SECRET=your-super-secret-jwt-key-change-in-production-with-random-32-chars-minimum
```

### 2. Files Created for Railway

- ✅ `Procfile` - Tells Railway how to start the app
- ✅ `runtime.txt` - Specifies Python 3.11
- ✅ `requirements-railway.txt` - Fixed NumPy version conflict
- ✅ `railway.json` - Railway configuration

### 3. Next Steps

1. **Connect Railway to GitHub**:
   - Go to railway.app
   - Create new project
   - Connect GitHub repository
   - Select `python-service` folder as root

2. **Add Environment Variables**:
   - Copy the variables above to Railway dashboard

3. **Deploy**:
   - Railway will automatically deploy when you push to GitHub

### 4. Test Deployment

Once deployed, test the health endpoint:
```
https://your-railway-url.railway.app/health
```

### 5. Update Vercel

After getting Railway URL, update Vercel environment variables:
```
NEXT_PUBLIC_PYTHON_BACKEND_URL=https://your-railway-url.railway.app
```
