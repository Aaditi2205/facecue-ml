# 🚀 FaceCue ML - Deployment Guide

## Streamlit Cloud Deployment

### 1. **Prepare Your Repository**
- Ensure all files are committed and pushed to GitHub
- Make sure `requirements.txt` is in the root directory
- Verify model files are included in the repository

### 2. **Deploy to Streamlit Cloud**

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Repository**: `Aaditi2205/facecue-ml`
5. **Branch**: `main`
6. **Main file path**: `scripts/deployment_ui.py`
7. **App URL**: Choose a custom URL (e.g., `facecue-ml-health-assessment`)
8. **Click "Deploy"**

### 3. **Alternative: Use Professional UI**

For the full professional healthcare interface:
- **Main file path**: `scripts/professional_healthcare_ui.py`

### 4. **Troubleshooting**

If you see "Model not found" error:
- The deployment UI includes a fallback model for demonstration
- For full functionality, ensure model files are uploaded to the repository
- The fallback model provides basic predictions for demonstration

### 5. **Model Files Required**

For full functionality, ensure these files are in your repository:
- `data/integrated_nutritional_model.pkl`
- `data/nutritional_deficiency_model.pkl`

### 6. **Deployment Features**

✅ **Fallback Model** - Works even without model files  
✅ **Professional UI** - Healthcare-grade interface  
✅ **Error Handling** - Graceful degradation  
✅ **Mobile Responsive** - Works on all devices  
✅ **Fast Loading** - Optimized for cloud deployment  

## 🌐 Your Deployed App

Once deployed, your app will be available at:
`https://[your-app-name].streamlit.app`

## 📊 Features Available

- **AI-Powered Health Assessment**
- **Professional Medical Recommendations**
- **Real-time Deficiency Detection**
- **Healthcare-Grade Interface**
- **Mobile Responsive Design**

## 🔧 Customization

To customize your deployment:
1. Edit `scripts/deployment_ui.py`
2. Push changes to GitHub
3. Streamlit Cloud will auto-update your app

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all dependencies in `requirements.txt`
3. Ensure model files are properly uploaded
4. Test locally before deploying
