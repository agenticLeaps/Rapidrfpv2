#!/usr/bin/env python3
"""
Check available Gemini models and their quotas for production use
"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_gemini_models():
    """Check available Gemini models and their quota limits"""
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        print("🔍 Available Gemini Models for Production:")
        print("=" * 60)
        
        # List available models
        models = genai.list_models()
        
        production_models = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-2.0-flash-exp',
            'gemini-1.0-pro',
            'gemini-pro'
        ]
        
        for model in models:
            model_name = model.name.split('/')[-1]
            if any(prod_model in model_name for prod_model in production_models):
                print(f"\n📋 {model_name}")
                print(f"   Display Name: {model.display_name}")
                print(f"   Description: {model.description[:100]}...")
                
                # Known quota info for production planning
                if 'flash' in model_name.lower():
                    if '2.0' in model_name:
                        quota_info = "⚠️  Limited quota: 10 RPM (Experimental)"
                        recommendation = "❌ Not suitable for 5 users"
                    else:
                        quota_info = "✅ Higher quota: 15 RPM (Free tier), 1000 RPM (Paid)"
                        recommendation = "✅ Good for 5 users with paid plan"
                elif 'pro' in model_name.lower():
                    quota_info = "✅ Production quota: 60 RPM (Free tier), 1000 RPM (Paid)"
                    recommendation = "✅ Excellent for 5 users"
                else:
                    quota_info = "❓ Check Google AI documentation"
                    recommendation = "❓ Verify quota limits"
                
                print(f"   Quota: {quota_info}")
                print(f"   Recommendation: {recommendation}")
        
        print("\n" + "=" * 60)
        print("💡 PRODUCTION RECOMMENDATIONS:")
        print("=" * 60)
        print("🥇 BEST: gemini-1.5-pro")
        print("   • 60 RPM free tier, 1000 RPM paid")
        print("   • Most capable model")
        print("   • Excellent for 5 concurrent users")
        print()
        print("🥈 GOOD: gemini-1.5-flash") 
        print("   • 15 RPM free tier, 1000 RPM paid")
        print("   • Fast and efficient")
        print("   • Suitable for 5 users with rate limiting")
        print()
        print("💳 COST OPTIMIZATION:")
        print("   • Free tier: 15-60 requests/minute")
        print("   • Paid tier: 1000+ requests/minute")  
        print("   • Consider upgrading to paid for production")
        print()
        print("🔧 RATE LIMITING STRATEGIES:")
        print("   • Reduce parallel workers: 4-6 workers max")
        print("   • Add delays between requests: 0.2s")
        print("   • Implement request queuing for peak usage")
        print("   • Use batch processing when possible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking Gemini Models for 5-User Production Setup")
    check_gemini_models()