# Gemini 2.5 Flash-Lite Integration Setup

## Overview
The system has been successfully migrated from OpenAI GPT-3.5-turbo to Google Gemini 2.5 Flash-Lite. This provides improved performance, cost efficiency, and faster response times.

## Changes Made

### 1. Configuration Updates
- Added `LLM_PROVIDER` setting (default: "gemini")
- Added Gemini-specific configuration options:
  - `GEMINI_API_KEY`
  - `GEMINI_MODEL` (default: "gemini-2.5-flash-lite")
  - `GEMINI_PROJECT_ID`
  - `GEMINI_LOCATION` (default: "global")

### 2. LLM Service Updates
- Modified `src/llm/llm_service.py` to support both OpenAI and Gemini providers
- Added provider-specific initialization methods
- Maintained backward compatibility with OpenAI
- Added JSON parsing improvements for Gemini responses

### 3. Dependencies
- Added `google-genai>=0.5.0` to requirements.txt

## Setup Instructions

### 1. Install Dependencies
```bash
pip install google-genai>=0.5.0
```

### 2. Get Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Generate an API key

### 3. Update Environment Configuration

#### Option A: Use Gemini (Recommended)
Create/update your `.env` file:
```bash
# LLM Provider
LLM_PROVIDER=gemini

# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# Keep OpenAI as fallback (optional)
OPENAI_API_KEY=your_openai_api_key_here
```

#### Option B: Continue with OpenAI
```bash
# LLM Provider
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Test the Integration
Run the test script to verify everything is working:
```bash
python test_gemini_integration.py
```

## Benefits of Gemini 2.5 Flash-Lite

1. **Cost Efficiency**: Significantly lower cost per token compared to GPT-3.5-turbo
2. **Speed**: Faster response times optimized for low latency
3. **Performance**: Improved reasoning, multimodal, math and factuality benchmarks
4. **Context Window**: 1 million token context window
5. **Multimodal Support**: Native support for text, images, and other modalities

## API Endpoints Affected

The following endpoints now use Gemini 2.5 Flash-Lite:

### `/process-document` Process
- Entity extraction from document chunks
- Relationship detection
- Semantic unit generation
- High-level summarization

### `/generate-response` Function
- Query understanding and response generation
- Conversation history integration
- Agentic knowledge discovery
- Context-aware answer synthesis

## Fallback Behavior

The system includes robust fallback mechanisms:
1. If Gemini is unavailable, it falls back to OpenAI (if configured)
2. If JSON parsing fails with Gemini, it attempts multiple parsing strategies
3. Configuration validation prevents startup with missing API keys

## Monitoring and Debugging

- All LLM calls are logged with provider information
- Error messages include provider-specific details
- Test script helps validate configuration

## Next Steps

1. Install the google-genai library: `pip install google-genai`
2. Get your Gemini API key from Google AI Studio
3. Update your .env file with the new configuration
4. Run the test script to verify everything works
5. Start the NodeRAG service and test document processing

The migration is complete and ready for production use!