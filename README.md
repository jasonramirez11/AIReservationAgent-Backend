# AI Restaurant Reservation Agent

This FastAPI application provides an AI-powered system that calls real restaurants to make reservations on behalf of users.

## Architecture

- **FastAPI Backend**: REST API for handling reservation requests
- **LangChain Workflow**: Orchestrates decision-making using LLMs
- **Perplexity AI**: Retrieves restaurant information (phone numbers, opening hours)
- **Twilio API**: Makes outbound calls to restaurants and handles call flow

## How It Works

1. A user submits a reservation request through the API with details like restaurant name, date, time, and party size
2. The system uses Perplexity AI to look up the restaurant's information, including phone number
3. LangChain workflows determine if the reservation is feasible and generate a script for calling the restaurant
4. The AI agent makes an outbound call to the restaurant using Twilio
5. During the call, the AI agent speaks with the restaurant staff:
   - Explaining the reservation request
   - Answering questions about the reservation details
   - Handling confirmation or rejection responses
   - Negotiating alternative times if needed
6. The system updates the reservation status based on the call outcome
7. The user can check the status of their reservation through the API

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-reservation-agent.git
cd ai-reservation-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a .env file with required API keys

```
# OpenAI API Key for LLM
OPENAI_API_KEY=your_openai_api_key

# Twilio for phone calls
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number

# Perplexity AI for restaurant info
PERPLEXITY_API_KEY=your_perplexity_api_key

# Public URL for webhooks (use ngrok during development)
SERVER_URL=your_public_server_url
```

### 5. Set up public URL for Twilio webhooks

For Twilio to communicate with your application during calls, your server needs to be accessible on the internet. During development, you can use ngrok:

```bash
ngrok http 8000
```

Update the SERVER_URL in your .env file with the ngrok URL.

### 6. Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

## API Endpoints

### Create a Reservation

```
POST /reservations/
```

Request body:
```json
{
  "restaurant_name": "Example Restaurant",
  "date": "2025-04-20",
  "time": "18:30",
  "party_size": 4,
  "special_requests": "Window table preferred",
  "user_id": "user123"
}
```

Response:
```json
{
  "reservation_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Your reservation request is being processed. Check the status endpoint for updates.",
  "details": null
}
```

### Check Reservation Status

```
GET /reservations/{reservation_id}
```

Response:
```json
{
  "reservation_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "reservation_confirmed",
  "message": "Reservation status: reservation_confirmed",
  "details": {
    "restaurant_name": "Example Restaurant",
    "date": "2025-04-20",
    "time": "18:30",
    "party_size": 4,
    "status": "reservation_confirmed",
    "last_restaurant_response": "Yes, we can accommodate your reservation",
    "last_ai_response": "Thank you for confirming. We look forward to our visit."
  }
}
```

## Call Flow Design

The system handles phone calls with restaurants using a stateless design:

1. **Call Initiation**: When the AI calls a restaurant, it uses TwiML to set up the call flow
2. **Conversation State**: Instead of using a database, conversation state is passed through URL parameters between requests
3. **Speech Recognition**: The Twilio Gather verb collects spoken responses from restaurant staff
4. **AI Response Generation**: LangChain processes staff responses and generates appropriate AI replies
5. **Call Stages**: The system tracks call stages (initial, gathering_info, considering_alternative, confirmed, rejected)
6. **Call Completion**: When a reservation is confirmed or rejected, the call ends and status is saved

## Requirements

- Python 3.9+
- OpenAI API key
- Twilio account with phone number
- Perplexity AI API key
- Public URL for webhooks