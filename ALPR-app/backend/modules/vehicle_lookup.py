# Import regex module to validate license plate formats
import re
# Import requests module to execute HTTP calls to external APIs
import requests
# Import ElementTree to parse XML responses specifically used by SOAP APIs
import xml.etree.ElementTree as ET

def fetch_vehicle_data(plate: str, username: str = "udishxarora2") -> dict:
    """
    Cleans a license plate string, validates it, and fetches vehicle data
    from the RegCheck SOAP API.
    """
    # Preprocessing: Remove all empty spaces and convert the string to uppercase
    clean_plate = plate.replace(" ", "").upper()
    
    # Define a regex pattern validating typical Indian License Plates
    # e.g., MH12AB1234 -> 2 letters, 2 numbers, 1-2 letters, 4 numbers
    pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
    # If the provided plate doesn't match the regex pattern
    if not pattern.match(clean_plate):
        # Decline the API call locally saving bandwidth and time
        return {
            "plate": clean_plate,
            "valid": False,
            "data": None,
            "error": "Invalid Indian License Plate Format."
        }

    # Configuration for the external RegCheck SOAP API Call
    url = "http://www.regcheck.org.uk/api/reg.asmx"
    # Specify the Content-Type and SOAPAction as required by the regcheck documentation
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": '"http://regcheck.org.uk/CheckIndia"'
    }
    
    # Construct the strictly formatted SOAP XML payload required for the POST request
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <CheckIndia xmlns="http://regcheck.org.uk">
      <RegistrationNumber>{clean_plate}</RegistrationNumber>
      <username>{username}</username>
    </CheckIndia>
  </soap:Body>
</soap:Envelope>"""

    try:
        # Fire the POST request. Set a 10-second timeout to avoid indefinite hanging
        response = requests.post(url, data=soap_body, headers=headers, timeout=10)
        # Inherently raise an exception for HTTP failure/bad statuses (e.g. 500, 404)
        response.raise_for_status()
        
        # Parse the raw XML string response into an ElementTree object
        root = ET.fromstring(response.content)
        
        # We need the JSON library because the SOAP API wraps a JSON string inside the XML node
        import json
        
        # Helper function to extract text content dynamically from the XML tree
        def get_text(tag_name):
            # ElementTree does not support the local-name() XPath function out of the box.
            # We iterate through all elements and match the tag name, ignoring namespaces.
            for elem in root.iter():
                # Elements frequently include namespace URLs in '{}' braces prepended to their tag name
                if elem.tag == tag_name or elem.tag.endswith(f"}}{tag_name}"):
                    return elem.text
            return None

        # The CheckIndia API actually returns a stringified JSON blob within the 'vehicleJson' element
        vehicle_json_str = get_text("vehicleJson")
        
        # If the expected JSON payload exists in the response
        if vehicle_json_str:
            try:
                # Convert the JSON string back into a Python dictionary
                v_data = json.loads(vehicle_json_str)
                
                # Helper function because some fields might be plain strings, and others dictionaries
                def extract_val(field):
                    val = v_data.get(field)
                    # For complex fields dictating nested 'CurrentTextValue' hierarchies
                    if isinstance(val, dict):
                        return val.get("CurrentTextValue", "")
                    # Otherwise, just return the value or an empty string
                    return val if val else ""
                
                # Extract all targeted metadata fields from the unwrapped JSON
                make = extract_val("CarMake")
                model = extract_val("CarModel")
                fuel = extract_val("FuelType")
                engine = extract_val("EngineSize")
                year = extract_val("RegistrationYear")
                owner = extract_val("Owner")
                registration_date = extract_val("RegistrationDate")
                insurance = extract_val("Insurance")
                location = extract_val("Location")
                
                # Package everything cleanly into our application's response blueprint
                return {
                    "plate": clean_plate,
                    "valid": True,
                    "data": {
                        "make": make,
                        "model": model,
                        "fuel": fuel,
                        "engine": engine,
                        "year": year,
                        "owner": owner,
                        "registration_date": registration_date,
                        "insurance": insurance,
                        "location": location
                    }
                }
            except json.JSONDecodeError as e:
                # Error handler for unexpected or corrupt JSON strings
                return {
                    "plate": clean_plate,
                    "valid": False,
                    "error": f"Failed to parse inner JSON: {str(e)}",
                    "data": None
                }
                
        # If 'vehicleJson' isn't present, the API may have returned an error string instead
        # Check standard SOAP fault locations
        error_msg = get_text("Description") or get_text("faultstring")
        return {
            "plate": clean_plate,
            "valid": False,
            "error": error_msg or "No vehicle data returned.",
            "data": None
        }

    except requests.exceptions.RequestException as e:
        # HTTP specific exceptions like Timeouts or ConnectionRefused
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"API request failed: {str(e)}",
            "data": None
        }
    except ET.ParseError as e:
        # Exceptions caused if the RegCheck servers respond with unreadable XML
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"XML parsing failed: {str(e)}",
            "data": None
        }
    except Exception as e:
        # Global catch-all fallback for any other uncaught runtime issues
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"Unexpected error: {str(e)}",
            "data": None
        }
