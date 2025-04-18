import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import requests

# Initialize the router
router = APIRouter()

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4.1",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# For storing call state (in a production app, use Redis or a database)
call_states: Dict[str, Dict[str, Any]] = {}

# Optional: Initialize ElevenLabs for TTS (Text-to-Speech)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") # Default voice

def text_to_speech(text: str) -> Optional[bytes]:
    """Convert text to speech using ElevenLabs API"""
    if not ELEVENLABS_API_KEY:
        return None
        
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content
    return None

def create_llm_conversation_chain():
    """Create a LLM chain for handling the conversation with the restaurant"""
    template = """
    You are an AI assistant making a restaurant reservation call. 
    You need to have a natural conversation with the restaurant staff to book a table.
    
    Reservation details:
    - Restaurant: {restaurant_name}
    - Date: {date}
    - Time: {time}
    - Party size: {party_size}
    - Special requests: {special_requests}
    
    Current conversation history:
    {conversation_history}
    
    The restaurant staff just said: "{current_input}"
    
    Please provide your response. Keep it conversational, polite, and focused on making the reservation.
    If they ask for information you don't have, politely explain you don't have that information but can proceed with the basic reservation.
    If the reservation is confirmed, express gratitude and confirm the details.
    If they say they're fully booked, ask if there are any alternative times available.
    
    Your response:
    """
    
    prompt = PromptTemplate(
        input_variables=["restaurant_name", "date", "time", "party_size", "special_requests", 
                         "conversation_history", "current_input"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def initialize_call_state(call_sid: str, reservation_data: Dict[str, Any]):
    """Initialize the state for a new call"""
    call_states[call_sid] = {
        "reservation_data": reservation_data,
        "conversation_history": [],
        "turns": 0
    }

def update_call_state(call_sid: str, speaker: str, message: str):
    """Update the conversation history in the call state"""
    if call_sid not in call_states:
        raise ValueError(f"Call {call_sid} not found in call states")
    
    call_states[call_sid]["conversation_history"].append(f"{speaker}: {message}")
    call_states[call_sid]["turns"] += 1

def get_ai_response(call_sid: str, user_input: str) -> str:
    """Get the AI response based on the current conversation"""
    if call_sid not in call_states:
        return "I'm sorry, there was an error with this call. Please try again later."
    
    state = call_states[call_sid]
    reservation_data = state["reservation_data"]
    
    # Create the conversation chain
    chain = create_llm_conversation_chain()
    
    # Get AI response
    response = chain.run(
        restaurant_name=reservation_data.get("restaurant_name", ""),
        date=reservation_data.get("date", ""),
        time=reservation_data.get("time", ""),
        party_size=reservation_data.get("party_size", ""),
        special_requests=reservation_data.get("special_requests", "None"),
        conversation_history="\n".join(state["conversation_history"]),
        current_input=user_input
    )
    
    # Update the conversation history
    update_call_state(call_sid, "Restaurant", user_input)
    update_call_state(call_sid, "AI", response)
    
    return response

@router.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    
    # Get reservation data from the query params (in production, fetch from database)
    query_params = {k: v for k, v in request.query_params.items()}
    reservation_data = json.loads(query_params.get("reservation_data", "{}"))
    
    # Initialize call state if this is a new call
    if call_sid not in call_states:
        initialize_call_state(call_sid, reservation_data)
    
    # Create TwiML response
    response = VoiceResponse()
    
    # If we've had too many conversation turns, end the call
    if call_states[call_sid]["turns"] >= 10:
        response.say("Thank you for your time. Have a great day!")
        response.hangup()
        return str(response)
    
    # Generate initial message if this is the first interaction
    if call_states[call_sid]["turns"] == 0:
        initial_message = (
            f"Hello, I'm calling to make a reservation at {reservation_data.get('restaurant_name', '')} "
            f"for {reservation_data.get('party_size', '')} people on {reservation_data.get('date', '')} "
            f"at {reservation_data.get('time', '')}."
        )
        if reservation_data.get("special_requests"):
            initial_message += f" We have a special request: {reservation_data.get('special_requests')}."
        
        update_call_state(call_sid, "AI", initial_message)
        
        # Use ElevenLabs TTS if available, otherwise use Twilio's TTS
        audio_data = text_to_speech(initial_message) if ELEVENLABS_API_KEY else None
        
        if audio_data:
            # TODO: In production, you would store this audio file and provide a URL for Twilio to play
            # For now, we'll use Twilio's built-in TTS
            response.say(initial_message)
        else:
            response.say(initial_message)
    
    # Gather speech input from the restaurant staff
    gather = Gather(input="speech", action="/twilio/gather", timeout=5, speech_timeout="auto")
    response.append(gather)
    
    # If no input is received, ask if they're still there
    response.say("I'm waiting for your response. Are you still there?")
    response.redirect("/twilio/voice")
    
    return str(response)

@router.post("/twilio/gather", response_class=PlainTextResponse)
async def twilio_gather_webhook(request: Request):
    """Handle speech input from the restaurant staff"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    speech_result = form_data.get("SpeechResult", "")
    
    # Get AI response
    ai_response = get_ai_response(call_sid, speech_result)
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Use ElevenLabs TTS if available, otherwise use Twilio's TTS
    audio_data = text_to_speech(ai_response) if ELEVENLABS_API_KEY else None
    
    if audio_data:
        # TODO: In production, you would store this audio file and provide a URL for Twilio to play
        # For now, we'll use Twilio's built-in TTS
        response.say(ai_response)
    else:
        response.say(ai_response)
    
    # Redirect back to the voice webhook to continue the conversation
    response.redirect("/twilio/voice")
    
    return str(response)

@router.post("/twilio/status", response_class=PlainTextResponse)
async def twilio_status_webhook(request: Request):
    """Handle Twilio call status callbacks"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    
    # In production, update your database with the call status
    print(f"Call {call_sid} status: {call_status}")
    
    # Clean up call state when the call is completed
    if call_status in ["completed", "busy", "no-answer", "failed", "canceled"]:
        if call_sid in call_states:
            # In production, save the conversation history to your database
            conversation_history = call_states[call_sid]["conversation_history"]
            print(f"Call {call_sid} conversation history: {conversation_history}")
            
            # Clean up
            del call_states[call_sid]
    
    return ""