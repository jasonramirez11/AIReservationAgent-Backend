import requests
import json

# For local development testing

# Create a reservation
response = requests.post(
    "http://localhost:8080/reservations/",
    json={
        "restaurant_name": "The Italian Restaurant",
        "date": "2025-04-20",
        "time": "19:00",
        "party_size": 4,
        "special_requests": "Window table if possible",
        "user_id": "user123",
        "phone_number": "+15108475240",
        "opening_hours": "24/7"
    }
)

# Print the response
reservation_data = response.json()
print(json.dumps(reservation_data, indent=2))

# Get the reservation ID
reservation_id = reservation_data["reservation_id"]

# Check reservation status
status_response = requests.get(f"http://localhost:8000/reservations/{reservation_id}")
print(json.dumps(status_response.json(), indent=2))