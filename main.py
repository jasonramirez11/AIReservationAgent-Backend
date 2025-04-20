import os
import json
import re
import uuid
import urllib.parse
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Literal
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Say, Gather, Pause
import httpx
from fastapi.responses import Response, JSONResponse
import logging  # Add at top with other imports
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Restaurant Reservation Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-reservation-agent-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize LLM using OpenAI
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.4,
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_base_url():
    """Get the base URL for webhook callbacks"""
    # Get from environment variable with fallback
    return os.getenv("SERVER_URL", "http://localhost:8000")

# Store active reservations in memory (for demo purposes)
# In production, we'd need a proper database
active_reservations = {}

# Pydantic models for data validation
class ReservationRequest(BaseModel):
    restaurant_name: str
    date: str
    time: str
    party_size: int
    special_requests: Optional[str] = None
    user_id: str = Field(..., description="Unique identifier for the user")
    phone_number: Optional[str] = Field(None, description="User's phone number for confirmation"),
    opening_hours: Optional[str] = Field(None, description="Restaurant's opening hours")

class ReservationResponse(BaseModel):
    reservation_id: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class LLMResponse(BaseModel):
    response: str = Field(description="The spoken response to the restaurant staff.")
    next_stage: Literal[
        "confirmed", "rejected", "gathering_info", "considering_alternative",
        "initial", "waiting", "finalizing", "end_call"
    ] = Field(description="The next stage of the call.")

# Helper function to fetch restaurant information using Perplexity API
async def fetch_restaurant_info(restaurant_name: str):
    """Fetch restaurant information including phone number using Perplexity API"""
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {perplexity_api_key}"
            },
            json={
                "model": "sonar-medium-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate information about restaurants."
                    },
                    {
                        "role": "user",
                        "content": f"Find the phone number, address, and opening hours for the restaurant named '{restaurant_name}'. Return only the information in JSON format."
                    }
                ]
            },
            timeout=10.0
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch restaurant information")
        
        result = response.json()
        return result

def create_langchain_workflow():
    """Creates a LangChain workflow for making reservation decisions"""
    template = """
     You are a warm, friendly, and professional AI restaurant reservation agent. Your goal is to help make restaurant reservations.
    
    Restaurant Information:
    {restaurant_info}
    
    Reservation Request:
    - Restaurant: {restaurant_name}
    - Date: {date}
    - Time: {time}
    - Party Size: {party_size}
    - Special Requests: {special_requests}
    
    Based on this information, determine:
    1. Is the restaurant likely open at the requested time?
    2. What should be said when calling the restaurant to make this reservation?
    3. What follow-up questions might the restaurant staff ask?
    
    Your call script should:
    - Keep it brief and natural - just a short opening like a real person would use
    - Start with a simple greeting and state your purpose (making a reservation)
    - Only mention the basic details (party size, date, time) in the initial greeting
    - Do NOT include all special requests, contact details, or excessive information in the opening
    - Do NOT use formal language like "I would like to inquire about" or "on behalf of"
    - Sound like a real person making a casual but polite call
    - The script should be 1-3 short sentences maximum
    - Remember the AI will handle follow-up questions during the conversation
    
    Output your decision in the following format:
    Decision: [proceed with call/need more information/suggest alternative]
    Reasoning: [explanation of your decision]
    Call Script: [exact script of what the AI should say during the call to the restaurant]
    Expected Questions: [what the restaurant staff might ask during the reservation call]
    Conversation Strategy: [brief notes on how to adapt if the conversation doesn't go as expected]
    """
    
    prompt = PromptTemplate(
        input_variables=["restaurant_info", "restaurant_name", "date", "time", "party_size", "special_requests"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# Create a LangChain for handling restaurant staff responses
def create_response_handler_chain():
    """Creates a LangChain to handle responses from restaurant staff (without memory)"""
    template = """
    You are a warm, friendly, and professional AI assistant making a restaurant reservation by phone. You need to respond appropriately to what the restaurant staff says.
    
    Reservation details:
    - Restaurant: {restaurant_name}
    - Date: {date}
    - Time: {time}
    - Party Size: {party_size}
    - Special Requests: {special_requests}
    
    The restaurant staff just said: "{staff_response}"
    
    Current call stage: {call_stage}
    
    Determine the appropriate response. Keep it conversational, natural, and focused on successfully making the reservation.
    If they ask for information you have (like contact details, name, etc.), provide it naturally.
    If they ask for information you don't have, politely explain you don't have that information currently.
    If they confirm the reservation, express gratitude and confirm the details.
    If they say they're fully booked, ask if there are any alternative times available.
    
    CAREFULLY ANALYZE THE FULL MEANING OF THE RESTAURANT STAFF'S RESPONSE:
    - Don't just look for keywords - understand the complete meaning and context
    - Consider context and tone, not just individual words
    - Pay attention to negations, qualifications, and nuances
    - Understand that affirmative words might be part of negative responses
      (e.g., "Yes, sorry, we're fully booked" is a rejection, not a confirmation)
    
    Based on your complete understanding, determine the next stage:
    - If they clearly confirmed the reservation is possible or booked: set stage to "confirmed"
    - If you're in the "confirmed" or "finalizing" stage and are still discussing details: set stage to "finalizing"
    - If all details are confirmed and the conversation has reached its natural conclusion: set stage to "end_call"
    - If they clearly rejected or stated no availability: set stage to "rejected"
    - If they're asking for information or details: set stage to "gathering_info"
    - If they suggested a different time or date: set stage to "considering_alternative"
    - If this is the initial greeting and they seem ready to help: set stage to "gathering_info"
    - Otherwise, keep the current stage as is
    
    Your response should be formatted exactly as follows:
    Response: [your spoken response to the restaurant staff]
    Next Stage: [confirmed/rejected/gathering_info/considering_alternative/initial/finalizing/end_call]
    
    The Next Stage MUST be one of the exact values listed above, with no additional text.
    
    Note: The "finalizing" stage is for handling any final details after the initial confirmation. This allows the conversation to continue naturally even after basic confirmation.
    """
    
    prompt = PromptTemplate(
        input_variables=["restaurant_name", "date", "time", "party_size", "special_requests", "staff_response", "call_stage"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def create_conversation_graph():
    """Creates a LangGraph for handling restaurant conversation with persistent memory"""
    from langgraph.graph import START, StateGraph

    class RestaurantConversationState(MessagesState):
        restaurant_name: str
        date: str
        time: str
        party_size: str
        special_requests: str
        call_stage: str
        reservation_id: str

    workflow = StateGraph(state_schema=RestaurantConversationState)

    # Define the function that calls the model
    def process_restaurant_response(state):
        # Valid stages list for prompt and validation
        valid_stages_list = [
            "confirmed", "rejected", "gathering_info", "considering_alternative",
            "initial", "waiting", "finalizing", "end_call"
        ]

        # Create system message with guidelines, asking for JSON output
        system_content = f"""
        You are a warm, friendly, and professional AI assistant making a restaurant reservation by phone. You need to respond appropriately to what the restaurant staff says.

        Reservation details:
        - Restaurant: {state["restaurant_name"]}
        - Date: {state["date"]}
        - Time: {state["time"]}
        - Party Size: {state["party_size"]}
        - Special Requests: {state["special_requests"]}

        Current call stage: {state["call_stage"]}

        Conversation guidelines:
        - Reference and acknowledge previous parts of the conversation when appropriate
        - Listen carefully to the staff and respond to their specific questions/concerns
        - If they ask for information you have (like contact details, name, etc.), provide it naturally
        - If they ask for information you don't have, politely explain you don't have that information
        - Always maintain a friendly, professional tone
        - Adapt to unexpected turns in the conversation (if they suggest different times, discuss specials, etc.)
        - Stay focused on the goal (making a reservation) but be flexible in how you get there
        - If they confirm the reservation, express gratitude and confirm all details
        - If they say they're fully booked, ask about alternative times/dates or thank them for their time
        - If they put you on hold or need to check something, acknowledge this when they return
        
        Determine the next stage of the call based on the conversation:
        - If they initially confirmed the reservation, set stage to "confirmed"
        - If you're in the "confirmed" or "finalizing" stage and are still discussing details, set stage to "finalizing"
        - If the conversation has reached its natural conclusion and all details are confirmed, set stage to "end_call"
        - If they rejected the reservation, set stage to "rejected"
        - If they need more information, set stage to "gathering_info"
        - If they suggested an alternative, set stage to "considering_alternative"
        - If they asked you to wait, set stage to "waiting"
        - Otherwise, keep the stage as is

        The next stage MUST be one of the following exact strings: {json.dumps(valid_stages_list)}.
        
        Note: 
        - The "finalizing" stage is for handling any final details after the initial confirmation
        - The "end_call" stage indicates the conversation has reached its natural conclusion and it's appropriate to end the call
        - Only use "end_call" when all details are confirmed and there's nothing more to discuss
    
        
        Your output MUST be a valid JSON object containing exactly two fields: "response" (string) and "next_stage" (string, one of the valid stages above).
        Provide ONLY the JSON object in your response, with no introductory text, explanations, or markdown formatting.

        Example JSON Output:
        {{
          "response": "Okay, great! So that's a table for 2 on Tuesday at 7 PM. Can I get a name for the reservation?",
          "next_stage": "gathering_info"
        }}
        """

        messages = [SystemMessage(content=system_content)] + state["messages"]

        # Initialize default values in case of error
        ai_response = "I apologize, I encountered a technical issue. Could you please repeat that?"
        next_stage = state["call_stage"] # Default to current stage on error

        for _ in range(3):
            try:
                # Call the model (consider adding retries specifically for JSON parsing if needed)
                response = llm.invoke(messages)
                response_text = response.content.strip()
                logger.info(f"Raw model response: {response_text}")

                # Attempt to parse the JSON output and validate with Pydantic
                # Handle potential ```json ... ``` markdown fences
                if response_text.startswith("```json"):
                    response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.DOTALL)
                    logger.info(f"Stripped markdown fences, attempting parse: {response_text}")

                llm_output_data = json.loads(response_text)
                llm_response_obj = LLMResponse(**llm_output_data) # Validate structure and next_stage enum

                ai_response = llm_response_obj.response
                next_stage = llm_response_obj.next_stage # Already validated by Pydantic
                
                if not ai_response or not next_stage:
                    raise ValueError("LLM response missing ai_response or next_stage")
                
                logger.info(f"Successfully parsed JSON: response='{ai_response[:30]}...', next_stage='{next_stage}'")
                break
            except (json.JSONDecodeError, ValidationError, Exception) as e: # Catch JSON, Pydantic, and other errors
                logger.error(f"Failed to parse/validate LLM JSON or other error: {e}", exc_info=True) # Log traceback
                logger.error(f"LLM Raw Output causing error: {response_text}")

        # Log the stage transition
        logger.info(f"Stage transition: '{state['call_stage']}' -> '{next_stage}'")

        # Update the call stage in the new state
        new_state = state.copy()
        new_state["call_stage"] = next_stage

        # Add AI message to the conversation history
        new_state["messages"].append(AIMessage(content=ai_response))

        # Log the final response and stage being used
        logger.info(f"Final AI response being used: {ai_response}")
        logger.info(f"Final next stage being used: {next_stage}")

        return new_state

    # Define the node and edge
    workflow.add_node("process_response", process_restaurant_response)
    workflow.add_edge(START, "process_response")

    # Compile the graph with memory
    return workflow.compile(checkpointer=conversation_memory)


def create_response_handler_chain_with_memory(memory):
    """Creates a LangChain to handle responses from restaurant staff with conversation memory
    (Legacy version maintained for compatibility)"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
            You are an AI assistant making a restaurant reservation by phone. You need to respond appropriately to what the restaurant staff says.
            
            Conversation guidelines:
            - Reference and acknowledge previous parts of the conversation when appropriate
            - Listen carefully to the staff and respond to their specific questions/concerns
            - If they ask for information you have (contact details, name, etc.), provide it naturally
            - If they ask for information you don't have, politely explain you don't have that information
            - Always maintain a friendly, professional tone
            - Adapt to unexpected turns in the conversation (if they suggest different times, discuss specials, etc.)
            - Stay focused on the goal (making a reservation) but be flexible in how you get there
            - If they confirm the reservation, express gratitude and confirm all details
            - If they say they're fully booked, ask about alternative times/dates or thank them for their time
            - If they put you on hold or need to check something, acknowledge this when they return
            
            Determine the next stage of the call based on the conversation:
            - If they confirmed the reservation, set stage to "confirmed"
            - If you're in the "confirmed" or "finalizing" stage and are still discussing details, set stage to "finalizing"
            - If all details are confirmed and the conversation has reached its natural conclusion, set stage to "end_call"
            - If they rejected the reservation, set stage to "rejected"
            - If they need more information, set stage to "gathering_info"
            - If they suggested an alternative, set stage to "considering_alternative"
            - If they asked you to wait, set stage to "waiting"
            - Otherwise, keep the stage as is
            
            Note: 
            - The "finalizing" stage is for handling any final details after the initial confirmation
            - The "end_call" stage indicates the conversation has reached its natural conclusion
            - Only use "end_call" when all details are confirmed and there's nothing more to discuss
            
            Your response should be formatted as text that can be spoken by the AI phone system.
            Output in the format:
            Response: [your spoken response]
            Next Stage: [next call stage]
        """),
        HumanMessage(content="""
            Reservation details:
            - Restaurant: {restaurant_name}
            - Date: {date}
            - Time: {time}
            - Party Size: {party_size}
            - Special Requests: {special_requests}
            
            Current call stage: {call_stage}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="The restaurant staff just said: \"{staff_response}\"")
    ])
    
    # Create chain without memory parameter to avoid KeyError
    chain = prompt | llm
    return chain

# Helper function to make a call to the restaurant using Twilio
def call_restaurant(phone_number: str, call_script: str, reservation_data: Dict[str, Any], reservation_id: str):
    """Makes a call to the restaurant using Twilio to place the reservation"""
    try:
        base_url = os.getenv("SERVER_URL", "https://your-server-url.com")

        logging.info(f"Making call to {phone_number} for reservation ID {reservation_id}")

        response = VoiceResponse()

        # Use only the call script generated by LangChain for a more natural introduction
        response.say(call_script, voice="alice")

        # URL encode reservation data for webhook
        url_params = {
            "restaurant": reservation_data["restaurant_name"],
            "date": reservation_data["date"],
            "time": reservation_data["time"],
            "party_size": str(reservation_data["party_size"]),
            "special_requests": reservation_data.get("special_requests", ""),
            "reservation_id": reservation_id,
            "call_stage": "initial"
        }
        param_str = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in url_params.items()])

        # Set up Gather to collect the restaurant's response
        gather = Gather(
            input="speech",
            action=f"{base_url}/call-response?{param_str}",
            method="POST",
            timeout=5,
            speechTimeout="auto"
        )
        response.append(gather)

        # If no input is received
        response.say("I'm sorry, I didn't catch that. I'll call back at a better time. Thank you.", voice="alice")

        # Make the call
        call = twilio_client.calls.create(
            twiml=str(response),
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            status_callback=f"{base_url}/call-status?reservation_id={reservation_id}",
            status_callback_method="POST",
            record=True
        )

        # Update reservation status in memory
        active_reservations[reservation_id] = {
            **reservation_data,
            "status": "call_initiated",
            "call_sid": call.sid
        }

        return call.sid
    except Exception as e:
        logger.error(f"Error making call: {str(e)}")
        # Save error to memory
        active_reservations[reservation_id] = {
            **reservation_data,
            "status": "call_failed",
            "error_message": str(e)
        }
        raise HTTPException(status_code=500, detail="Failed to make the call")

# Process the reservation in the background
async def process_reservation(reservation_data: Dict[str, Any], reservation_id: str):
    """Process the reservation request in the background"""
    try:

        logging.info(f"Processing reservation {reservation_id} for {reservation_data['restaurant_name']}")
        logging.info(f"Reservation data: {reservation_data}")

        restaurant_info = {
            "address": reservation_data.get("address"),
            "opening_hours": reservation_data.get("opening_hours"),
            "phone": reservation_data.get("phone_number"),
        }

        if not restaurant_info["phone"]:
            # 1. Fetch restaurant information
            restaurant_info_response = await fetch_restaurant_info(reservation_data["restaurant_name"])
            # Extract the restaurant info from the response
            restaurant_info = restaurant_info_response['choices'][0]['message']['content']
        
        # 2. Use LangChain workflow to make decisions
        workflow = create_langchain_workflow()

        # Log the full prompt being sent to the LLM
        prompt_inputs = {
            "restaurant_info": restaurant_info,
            "restaurant_name": reservation_data["restaurant_name"],
            "date": reservation_data["date"],
            "time": reservation_data["time"],
            "party_size": str(reservation_data["party_size"]),
            "special_requests": reservation_data.get("special_requests") or "None"
        }
        
        # Format the actual prompt template with inputs
        full_prompt = workflow.prompt.format(**prompt_inputs)
        logger.info(f"Decision-making prompt:\n{full_prompt}")

        decision_result = workflow.run(**prompt_inputs)
        logger.info(f"LLM decision result:\n{decision_result}")

        #logging.info(f"Decision result: {decision_result}")

        # Extract the call script from the decision result
        call_script_match = re.search(r'Call Script:(.*?)(?:Expected Questions:|Conversation Strategy:|$)', decision_result, re.DOTALL)
        call_script = call_script_match.group(1).strip() if call_script_match else "I'd like to make a reservation."
        
        # Extract conversation strategy if available
        strategy_match = re.search(r'Conversation Strategy:(.*?)$', decision_result, re.DOTALL)
        conversation_strategy = strategy_match.group(1).strip() if strategy_match else ""
        
        # Save the strategy to the reservation data for reference during the conversation
        if conversation_strategy:
            reservation_data["conversation_strategy"] = conversation_strategy
        
        # 3. Make the call if the decision is to proceed
        if "proceed with call" in decision_result.lower():
            conversation_strategy = strategy_match.group(1).strip() if strategy_match else ""
        
            # Save the strategy to the reservation data for reference during the conversation
            if conversation_strategy:
                reservation_data["conversation_strategy"] = conversation_strategy
        
        # 3. Make the call if the decision is to proceed
        if "proceed with call" in decision_result.lower():
            logging.info(f"Proceeding with call to {reservation_data['restaurant_name']} at {reservation_data['time']}")

            phone_number = restaurant_info.get("phone")

            # Extract phone number from restaurant info
            #phone_match = re.search(r'"phone":\s*"([^"]+)"', restaurant_info)
            #phone_number = phone_match.group(1) if phone_match else None
            
            if not phone_number:
                # Save failure to memory
                active_reservations[reservation_id] = {
                    **reservation_data,
                    "status": "failed",
                    "message": "Could not find restaurant phone number"
                }
                return
            
            # Make the AI-powered call to the restaurant
            call_sid = call_restaurant(phone_number, call_script, reservation_data, reservation_id)
            
            # Status is updated in the call_restaurant function
        else:
            logging.info(f"Decision not to call: {decision_result}")

            # Save the decision not to call
            active_reservations[reservation_id] = {
                **reservation_data,
                "status": "decision_no_call",
                "decision": decision_result
            }
            
    except Exception as e:
        # Log the error
        logger.error(f"Error processing reservation: {str(e)}")
        
        # Save error to memory
        active_reservations[reservation_id] = {
            **reservation_data,
            "status": "error",
            "error_message": str(e)
        }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint to verify the API is running"""
    return {"message": "AI Restaurant Reservation Agent API is running"}

@app.post("/reservations/", response_model=ReservationResponse)
async def create_reservation(reservation_request: ReservationRequest, background_tasks: BackgroundTasks):
    """
    Create a new restaurant reservation.
    The system will:
    1. Lookup the restaurant information
    2. Make a decision about the reservation
    3. Call the restaurant if appropriate
    4. Return the status of the reservation
    """
    reservation_id = str(uuid.uuid4())
    
    # Log new reservation request
    logger.info(f"Creating reservation {reservation_id} for {reservation_request.restaurant_name} "
                f"on {reservation_request.date} at {reservation_request.time} "
                f"for {reservation_request.party_size} people")
    
    # Initialize reservation in memory
    active_reservations[reservation_id] = {
        **reservation_request.dict(),
        "status": "processing"
    }
    
    # Start background processing
    background_tasks.add_task(process_reservation, reservation_request.dict(), reservation_id)
    
    # Return immediate response
    return ReservationResponse(
        reservation_id=reservation_id,
        status="processing",
        message="Your reservation request is being processed. Check the status endpoint for updates."
    )

@app.get("/reservations/{reservation_id}", response_model=ReservationResponse)
async def get_reservation_status(reservation_id: str):
    """Get the status of a reservation by its ID"""
    logger.info(f"Checking status of reservation {reservation_id}")
    
    if reservation_id not in active_reservations:
        logger.warning(f"Reservation {reservation_id} not found")
        raise HTTPException(status_code=404, detail="Reservation not found")
    
    reservation_data = active_reservations[reservation_id]
    logger.debug(f"Reservation {reservation_id} status: {reservation_data.get('status')}")
    
    # Check if this reservation has an active conversation in LangGraph memory
    thread_id = f"reservation_{reservation_id}"
    
    # Add conversation status to the details
    details = {
        **reservation_data,
    }
    
    # Try to get conversation information if available
    try:
        # Check if a conversation is stored in memory
        conversation_data = conversation_memory.get(thread_id)
        if conversation_data and isinstance(conversation_data, dict):
            details["has_active_conversation"] = True
            
            # Extract messages safely
            messages = conversation_data.get("messages", [])
            if isinstance(messages, list):
                details["conversation_messages_count"] = len(messages)
                # Count AI messages safely
                ai_messages = 0
                for m in messages:
                    if hasattr(m, "__class__") and m.__class__.__name__ == "AIMessage":
                        ai_messages += 1
                details["conversation_turns"] = ai_messages
            else:
                details["conversation_messages_count"] = 0
                details["conversation_turns"] = 0
                
            # Extract call stage safely
            if isinstance(conversation_data.get("call_stage"), str):
                details["last_stage"] = conversation_data.get("call_stage")
            else:
                details["last_stage"] = "unknown"
        else:
            details["has_active_conversation"] = False
    except Exception as e:
        logger.error(f"Error accessing conversation memory: {str(e)}, type: {type(conversation_memory)}")
        details["has_active_conversation"] = False
        details["conversation_error"] = str(e)
    
    return ReservationResponse(
        reservation_id=reservation_id,
        status=reservation_data.get("status", "unknown"),
        message=f"Reservation status: {reservation_data.get('status', 'unknown')}",
        details=details
    )

# Create memory saver for persistent conversations
conversation_memory = MemorySaver()

# Twilio call webhook endpoint
@app.post("/call-response")
async def call_response(request: Request):
    """Handle the restaurant's response to the reservation call and use conversation memory."""
    # Get base URL from request
    base_url = get_base_url()
    logger.info(f"Using base URL: {base_url}")
    
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    
    logger.info(f"Received restaurant response: '{speech_result}'")

    # Extract reservation data from URL params
    params = request.query_params
    restaurant = params.get("restaurant", "")
    date = params.get("date", "")
    time_ = params.get("time", "")
    party_size = params.get("party_size", "")
    special_requests = params.get("special_requests", "")
    reservation_id = params.get("reservation_id", "")
    call_stage = params.get("call_stage", "initial")
    
    logger.info(f"Call response for reservation ID {reservation_id}, restaurant {restaurant}, stage: {call_stage}")

    reservation = active_reservations.get(reservation_id)
    if not reservation:
        logger.warning(f"Reservation {reservation_id} not found in /call-response")
        return Response(content="<Response><Say>Reservation not found.</Say></Response>", media_type="application/xml")
    
    logger.info(f"Reservation data: {reservation}")

    thread_id = f"reservation_{reservation_id}"
    logger.info(f"Thread ID for conversation: {thread_id}")
    
    conversation_app = create_conversation_graph()
    
    # Prepare input for the conversation
    # First, check if this is continuing a conversation or starting a new one
    if call_stage == "initial":
        # For first-time conversation, create a new state with just the human message
        logger.info(f"Starting new conversation for reservation {reservation_id}")
        
        # For initial messages, add any conversation strategy as a system message
        initial_messages = []
        if "conversation_strategy" in reservation:
            initial_messages.append(
                SystemMessage(content=f"Conversation Strategy: {reservation['conversation_strategy']}")
            )
        
        initial_messages.append(HumanMessage(content=speech_result))
        
        # Create the initial state
        input_state = {
            "messages": initial_messages,
            "restaurant_name": restaurant,
            "date": date,
            "time": time_,
            "party_size": party_size,
            "special_requests": special_requests,
            "call_stage": call_stage,
            "reservation_id": reservation_id
        }
    else:
        logger.info(f"Continuing conversation for reservation {reservation_id}, stage: {call_stage}")
        
        # Create input with just the new human message
        input_state = {
            "messages": [HumanMessage(content=speech_result)],
            "restaurant_name": restaurant,
            "date": date,
            "time": time_,
            "party_size": party_size,
            "special_requests": special_requests,
            "call_stage": call_stage,
            "reservation_id": reservation_id
        }
    
    # Invoke the conversation app
    try:
        logger.info(f"Invoking conversation graph for thread {thread_id}")
        result = conversation_app.invoke(
            input_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the AI response from the conversation state
        messages = result["messages"]
        ai_message = messages[-1]  # Last message should be the AI's response
        
        if isinstance(ai_message, AIMessage):
            ai_response = ai_message.content
            
            # Extract and validate the call stage
            raw_next_stage = result["call_stage"]
            
            # Validate that the next_stage is one of the expected values
            valid_stages = ["confirmed", "rejected", "gathering_info", "considering_alternative", "initial", "waiting", "finalizing", "end_call"]
            next_stage = raw_next_stage if raw_next_stage in valid_stages else call_stage
            
            logger.info(f"AI Response: {ai_response[:30]}...")
            logger.info(f"Previous stage: '{call_stage}', transitioned to: '{next_stage}'")
            logger.info(f"Total conversation turns: {len(messages)}")
        else:
            # Fallback if something unexpected happens
            ai_response = "I apologize, could you please repeat that? I want to make sure we get your reservation details right."
            next_stage = call_stage
            logger.error(f"Expected AI message, got: {type(ai_message)}")
    except Exception as e:
        # Fallback to original chain if there's an error with LangGraph
        logger.error(f"Error using conversation graph: {str(e)}")
        logger.info("Falling back to legacy response handler chain without memory")
        
        # Use the legacy chain without memory to avoid reference errors
        response_chain = create_response_handler_chain()  # Use non-memory version for fallback
        legacy_result = response_chain.invoke(
            restaurant_name=restaurant,
            date=date,
            time=time_,
            party_size=party_size,
            special_requests=special_requests,
            staff_response=speech_result,
            call_stage=call_stage
        )
        
        # Parse result with improved regex to handle the enhanced format
        response_match = re.search(r'Response:(.*?)(?:Next Stage:|$)', legacy_result.content, re.DOTALL)
        stage_match = re.search(r'Next Stage:\s*(\w+)', legacy_result.content, re.DOTALL)
        
        ai_response = response_match.group(1).strip() if response_match else "I apologize, could you repeat that? I want to make sure I get your reservation details correctly."
        
        # Validate that the next_stage is one of the expected values
        valid_stages = ["confirmed", "rejected", "gathering_info", "considering_alternative", "initial", "waiting", "finalizing", "end_call"]
        extracted_stage = stage_match.group(1).strip() if stage_match else ""
        next_stage = extracted_stage if extracted_stage in valid_stages else call_stage
        
        logger.info(f"Parsed response format - Response: '{ai_response[:30]}...', Next Stage: '{next_stage}'")
        logger.info(f"Previous stage was: '{call_stage}', transitioned to: '{next_stage}'")

    # ai_response and next_stage are already set by this point from either the LangGraph or the fallback
    
    # Create TwiML response
    response = VoiceResponse()
    response.say(ai_response, voice="alice")
    
    # If the reservation is confirmed, transition to finalizing stage for additional details
    # but only if we don't already have an explicit end_call or finalizing stage
    if next_stage == "confirmed" and call_stage not in ["finalizing", "end_call"]:
        next_stage = "finalizing"
        logger.info(f"Transitioning from confirmed to finalizing stage to complete reservation details")
    
    # End the call if in end_call stage, or rejected
    if next_stage == "rejected" or next_stage == "end_call":
        response.say("Thank you so much for your help today. We look forward to dining with you! Have a wonderful day.", voice="alice")
        response.hangup()
        
        # Update reservation status in memory
        if reservation_id in active_reservations:
            # For end_call or finalizing stage that's ending, we mark it as confirmed unless it's rejected
            final_status = "confirmed" if next_stage in ["finalizing", "end_call"] else next_stage
            active_reservations[reservation_id]["status"] = "reservation_" + final_status
            active_reservations[reservation_id]["last_restaurant_response"] = speech_result
            active_reservations[reservation_id]["last_ai_response"] = ai_response
            
            # Final conversation is already saved in LangGraph memory
            logger.info(f"Call completed with status: {final_status}. Final conversation saved to LangGraph.")
    else:
        # Continue the conversation
        # URL encode reservation data for the next webhook
        url_params = {
            "restaurant": restaurant,
            "date": date,
            "time": time_,
            "party_size": party_size,
            "special_requests": special_requests,
            "reservation_id": reservation_id,
            "call_stage": next_stage
        }
        param_str = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in url_params.items()])
        
        # Set up another gather for continuing the conversation
        gather = Gather(
            input="speech", 
            action=f"{base_url}/call-response?{param_str}",
            method="POST",
            timeout=5,
            speechTimeout="auto"
        )
        response.append(gather)
        
        # If no further input
        response.say("I'm sorry, I didn't catch that. Thank you for your time today! Goodbye.", voice="alice")
        response.hangup()
        
        # Update reservation status in memory
        if reservation_id in active_reservations:
            # Save this exchange to both systems
            active_reservations[reservation_id]["status"] = "call_in_progress"
            active_reservations[reservation_id]["call_stage"] = next_stage
            active_reservations[reservation_id]["last_restaurant_response"] = speech_result
            active_reservations[reservation_id]["last_ai_response"] = ai_response
            
            # Update conversation with the new message in LangGraph memory
            # (No need to manually save context since LangGraph already handled this)
            logger.info(f"Conversation continuing. Stage: {next_stage}. Added to LangGraph memory.")
            
            # Also keep the history array for backward compatibility
            if "history" not in active_reservations[reservation_id]:
                active_reservations[reservation_id]["history"] = []
            
            active_reservations[reservation_id]["history"].append({
                "stage": next_stage,
                "restaurant_response": speech_result,
                "ai_response": ai_response,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    return Response(content=str(response), media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request):
    """Handle Twilio call status callbacks"""
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "")
    call_status = form_data.get("CallStatus", "")
    reservation_id = request.query_params.get("reservation_id")
    
    logger.info(f"Call status update for reservation {reservation_id}: {call_status}")
    
    # Status mapping
    status_mapping = {
        "completed": "call_completed",
        "busy": "restaurant_busy",
        "no-answer": "no_answer",
        "failed": "call_failed",
        "canceled": "call_canceled"
    }
    
    # Update reservation status in memory
    if reservation_id in active_reservations:
        active_reservations[reservation_id]["call_status"] = call_status
        
        # Only update the overall status if it hasn't been confirmed or rejected already
        current_status = active_reservations[reservation_id].get("status", "")
        if "reservation_confirmed" not in current_status and "reservation_rejected" not in current_status:
            active_reservations[reservation_id]["status"] = status_mapping.get(call_status, current_status)
        
        # If the call is ended for any reason, archive the conversation data
        if call_status in ["completed", "busy", "no-answer", "failed", "canceled"]:
            thread_id = f"reservation_{reservation_id}"
            try:
                # Get conversation data from LangGraph memory
                conversation_data = conversation_memory.get(thread_id)
                
                if conversation_data and isinstance(conversation_data, dict):
                    # Extract message count safely
                    messages = conversation_data.get("messages", [])
                    message_count = len(messages) if isinstance(messages, list) else 0
                    
                    # Extract call stage safely
                    call_stage = "unknown"
                    if isinstance(conversation_data.get("call_stage"), str):
                        call_stage = conversation_data.get("call_stage")
                    
                    # Save conversation data to reservation record
                    active_reservations[reservation_id]["conversation_data"] = {
                        "messages_count": message_count,
                        "last_stage": call_stage,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Conversation data retrieved from thread {thread_id}: {message_count} messages, stage: {call_stage}")
                    logger.info(f"Archived conversation for reservation {reservation_id}")
                else:
                    logger.warning(f"Conversation data not found or not in expected format for thread {thread_id}")
                    active_reservations[reservation_id]["conversation_data"] = {
                        "messages_count": 0,
                        "last_stage": "unknown",
                        "completed_at": datetime.utcnow().isoformat(),
                        "error": "Data not in expected format"
                    }
                    logger.info(f"No valid conversation data available for thread {thread_id}")
            except Exception as e:
                logger.error(f"Error archiving conversation: {str(e)}")
                # Save error to reservation data
                active_reservations[reservation_id]["conversation_data"] = {
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat()
                }
    
    return {"status": "received"}

# Run the app with Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)