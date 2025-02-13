import json
import subprocess
import requests
import re
from fastapi import FastAPI, HTTPException, Query
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import os
import glob
from pathlib import Path
from dateutil.parser import parse
from PIL import Image
import pytesseract
import numpy as np
import sqlite3


# Define the FastAPI app
app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST","GET"],
    allow_headers=["*"],
)

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
OPENAI_API_KEY = os.getenv("AIPROXY_TOKEN")

OPENAI_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_file",
            "description": "run python file with email",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "email id of the user"
                    },
                    "file_url": {
                        "type": "string",
                        "description": "url of the python file"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "python file name"
                    },
                },
                "required": ["email_id","file_name","file_url"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "format file using prettier",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "file name to be formatted"
                    },
                },
                "required": ["file_name"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "day_count",
            "description": "week day count",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_name": {
                        "type": "string",
                        "description": "text file name containing dates"
                    },
                    "day_name": {
                        "type": "string",
                        "description": "which day of the week to be counted"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "file to which output needs to be written"
                    },
                },
                "required": ["input_file_name","output_file_name","day_name"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort the contacts using last name and first name",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_name": {
                        "type": "string",
                        "description": "input Json file containing contacts"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "input Json file containing contacts"
                    },
                    "keys": {
                     "type": "array",
                     "items": {
                       "type": "string"
                     },
                     "description": "List of keys in the order to sort by."
                   },
                 
                },
                "required": ["input_file_name","output_file_name","keys"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_log",
            "description": "write specified line of specified number of files",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ext": {
                        "type": "string",
                        "description": "file extension"
                    },
                    "input_file_path": {
                        "type": "string",
                        "description": "Input File Path"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "Output File Name"
                    },
                    "number_of_files": {
                        "type": "string",
                        "description": "number of files to be analyzed"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "latest or the oldest"
                    },
                    "line_number": {
                        "type": "number",
                        "description": "which row to be extracted"
                    },
                },
                "required": ["input_file_path","file_ext","output_file_name","number_of_files","timeframe","line_number"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_style_markdown",
            "description": "Find occurance of style in markdown",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ext": {
                        "type": "string",
                        "description": "file extension"
                    },
                    "input_file_path": {
                        "type": "string",
                        "description": "Input File Path"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "Output File Name"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "pattern to search"
                    },
                },
                "required": ["input_file_path","file_ext","output_file_name","symbol"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_sender",
            "description": "Extract sender email id ",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_name": {
                        "type": "string",
                        "description": "file containing email"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "file to write output"
                    },
                    "email_field_name": {
                        "type": "string",
                        "description": "email field to extract"
                    },
                },
                "required": ["input_file_name","output_file_name","email_field_name"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_creditcard_details",
            "description": "Get Creditcard Details",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_name": {
                        "type": "string",
                        "description": "credit card image file"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "file to which the output needs to be written"
                    },
                },
                "required": ["input_file_name","output_file_name"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": "Find similar comments",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_name": {
                        "type": "string",
                        "description": "file name containing symbol"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "pattern to search"
                    },
                },
                "required": ["input_file_name","output_file_name"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_sales",
            "description": "Get total sales of items",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_file_name": {
                        "type": "string",
                        "description": "database file"
                    },
                    "column_list": {
                        "type": "string",
                        "description": "comma separated list of columns"
                    },
                    "condition": {
                        "type": "string",
                        "description": "condition to calculate sales"
                    },
                    "output_file_name": {
                        "type": "string",
                        "description": "file to which the output needs to be written"
                    },
                },
                "required": ["database_file_name","column_list","output_file_name","condition"],
                "additionalProperties": False
            },
            "strict": True
        }
    },  
    ]

# Define the functions that will be executed based on the task description
def install_uv():
    try:
        subprocess.run(["pip", "install", "uv"], check=True)
    except subprocess.CalledProcessError:
        return {"error": str(e)}

# Define the functions that will be executed based on the task description
def install_npx():
    try:
        subprocess.run(["pip", "install", "npx"], check=True)
    except subprocess.CalledProcessError:
        return {"error": str(e)}

def download_file(url, filename):
    try:
        dest = url
        response = requests.get(dest)
        subfolder = 'files'
        path = os.path.join(subfolder, filename)

        # Ensure the subfolder exists
        os.makedirs(subfolder, exist_ok=True)
        
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
        return path
    except Exception as e:
        return {"error": str(e)}

def run_python_file(filename,email_id):
    try:
        subprocess.run(["python", filename , email_id], check=True)
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

# Define the run_datagen function that automates the described tasks
async def run_python(email_id: str,file_name: str,file_url: str):
    try:
        install_uv()
        local_file = download_file(file_url,file_name)
        run_python_file(local_file,email_id)

        # Get the current working directory
        current_dir = Path().resolve()

        # Get the root path
        root_path = current_dir.anchor
        full_path = os.path.join(root_path, "data")
        return {"success": True, "email_id": email_id}

    except Exception as e:
        return {"error": str(e)}

def get_path(file_name: str):
    current_dir = Path().resolve()
    # Get the root path
    root_path = current_dir.anchor
    subfolder = 'data'

    # Combine the root folder, subfolder, and filename
    file_path = os.path.join(root_path, subfolder, file_name)
    return file_path


async def format_file(file_name: str):
    try:
        # Format the file using prettier@3.4.2
        # npx prettier --write .           # Format code or docs
        install_npx()
        file_path = get_path(file_name)
        subprocess.run(["npx", "prettier@3.4.2", "--write",file_path],capture_output=True,text=True,check=True)
    except Exception as e:
        return {"error": str(e)}

async def day_count(input_file_name: str,day_name: str,output_file_name:str):
    try:
        file_path = get_path(input_file_name)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_index = weekdays.index(day_name)
        count = 0
        with open(file_path, "r") as file:
            for line in file:
                date_str = line.strip()
                if not date_str:
                    continue  # Skip empty lines
                try:
                    parsed_date = parse(date_str)  # Auto-detect format
                    if parsed_date.weekday() == weekday_index:
                        count += 1
                except ValueError:
                    return {"error": str(e)}

        file_path = get_path(output_file_name)
        # Write the result to the output file
        with open(file_path, "w") as file:
            file.write(str(count))
    except Exception as e:
        return {"error": str(e)}

async def sort_contacts(input_file_name: str,output_file_name: str,keys:list):
    try:
        input_file_path = get_path(input_file_name)
        with open(input_file_path, "r") as file:
            data = json.load(file)
        # Ensure all keys exist in each contact
        for contact in data:
            for key in keys:
                if key not in contact:
                    contact[key] = ""

        # Sort the data based on the provided keys
        sorted_data = sorted(data, key=lambda x: (x['last_name'], x['first_name']))
       
        output_file_path = get_path(output_file_name)
        with open(output_file_path, "w") as file:
            json.dump(sorted_data, file)
    except Exception as e:
        return {"error": str(e)}

async def write_log(input_file_path: str,file_ext:str,output_file_name:str,number_of_files:str,timeframe:str,line_number:int):
    try:
        # Get all .log files in the directory
        input_file_path = get_path(input_file_path)
        output_file_path = get_path(output_file_name) 
        log_files = glob.glob(os.path.join(input_file_path, "*.log"))
        
        # Sort files by modification time, most recent first
        if timeframe == "latest":
            log_files.sort(key=os.path.getmtime, reverse=True)
        else:
            log_files.sort(key=os.path.getmtime, reverse=False)
        
        # Take the top `num_logs` files
        recent_logs = log_files[:int(number_of_files)]
        
        # Write the first line of each file to the output file
        with open(output_file_path, "w") as outfile:
            for log_file in recent_logs:
                with open(log_file, "r") as infile:
                    lines = infile.readlines()
                    if line_number <= len(lines):
                        outfile.write(lines[line_number - 1])
                    else:
                        outfile.write(f"Line {line_number} does not exist in {log_file}\n")
 
    except Exception as e:
        return {"error": str(e)}
    

#write_log
async def find_style_markdown(input_file_path: str,file_ext:str,output_file_name:str,style: str):
    try:
        input_file_path = get_path(input_file_path)
        output_file_path = get_path(output_file_name)

        file_selection = "*."+file_ext
        files = glob.glob(os.path.join(input_file_path, "**", file_selection), recursive=True)
        
        index = {}

        for file in files:
            title = None
            with open(file, "r", encoding="utf-8") as file1:
                for line in file1:
                    if line.startswith(style):
                        title = line.lstrip(style).strip()
                        break  # Stop reading after first H1

            # Compute relative path
            relative_path = os.path.relpath(file, input_file_path)

            # Store in index (even if no H1 is found)
            index[relative_path] = title if title else ""

        # Write to JSON file
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(index, json_file, indent=2, sort_keys=True)
    except Exception as e:
        return {"error": str(e)}
    
async def get_llm_response(function: str, inputs: list):
    try:
        if function == "extract_sender":
            field_name = inputs[0]
            email_body = inputs[1]
            prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Extract the {field_name} from the following email body:\n\n{email_body}"},
                        {"role": "user", "content": "Respond with only the field value. It should not contain any other content"}
                    ]
            payload={"model": "gpt-4o-mini", "messages" :prompt,}
        elif function == "get_creditcard_details":
            text = inputs[0]
            prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Verify if the following text contains a valid credit card number:\n\n{text}"}
                        #{"role": "user", "content": f"Extract credit card number from the image :\n\n{img}"},
                        #{"role": "user", "content": "Remove any unnecessarty spaces and characters in the card number and return only the card number"},
                    ]
            payload={"model": "gpt-4o-mini", "messages" :prompt,}
        elif function == "find_similar":
            text = inputs[0]
            payload={"model": "text-embedding-3-small", "input" :text,}
            OPENAI_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENAI_URL,
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            response_data = response.json()
            await write_cache(response_data)
        return response_data            
    except Exception as e:
        return {"error": str(e)}
    
async def write_cache(data: str,cache_file: str):
    with open(cache_file, "w") as f:
        json.dump(data, f)


async def extract_sender(input_file_name: str,output_file_name: str,email_field_name:str):
    try:
        # Write the extracted email address to the output file
        email_content_path = get_path(input_file_name)
        with open(email_content_path, "r") as file:
            email_info = file.read() #readlines gives list, this gives string

        output_file_path = get_path(output_file_name)
        inputs=[]
        inputs.append(email_field_name)
        inputs.append(email_info)
        response = await get_llm_response("extract_sender",inputs)

        with open(output_file_path, "w") as file:
            file.write(response['choices'][0]['message']['content'])
    except Exception as e:
        return {"error": str(e)}
    
def extract_text_from_png(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image,config='--psm 11')
    match = re.search(r'^\d{8} \d{4} \d{4}', text)
    if match:
        credit_card_number = match.group(0)
    else:
        credit_card_number = None

    credit_card_number = ''.join(credit_card_number.split())
    return credit_card_number

async def get_creditcard_details(input_file_name: str,output_file_name: str):
    try:
        input_file_path = get_path(input_file_name)
        output_file_path = get_path(output_file_name)
        text = extract_text_from_png(input_file_path)
        with open(output_file_path, 'w') as f:
            f.write(text)
    except Exception as e:
        return {"error": str(e)}

async def find_similar(input_file_name: str,output_file_name: str):
    try:
        input_file_path = get_path(input_file_name)
        output_file_path = get_path(output_file_name)
        # Read comments from file
        with open(input_file_path, 'r') as f:
            comments = f.readlines()
        inputs=[]
        inputs.append(comments)
        
        response = await get_llm_response("find_similar",inputs)
        embeddings = [data['embedding'] for data in response['data']]
        embeddings = np.array(embeddings)
        cosine_similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(cosine_similarities, -1)  # Exclude the diagonal
        most_similar_pair_idx = np.unravel_index(np.argmax(cosine_similarities), cosine_similarities.shape)
        most_similar_comments = [comments[most_similar_pair_idx[0]], comments[most_similar_pair_idx[1]]]
                
        with open(output_file_path, "w") as f:
            for comment in most_similar_comments:
                f.write(comment + "\n")
    except Exception as e:
        return {"error": str(e)}


async def get_ticket_sales(database_file_name:str,column_list:str,output_file_name:str,condition:str):
    try:
        input_file_path = get_path(database_file_name)
        output_file_path = get_path(output_file_name)

        # Connect to the SQLite database
        conn = sqlite3.connect(input_file_path)
        cursor = conn.cursor()

        query = f"""
        SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE {condition}
        """
        cursor.execute(query)
        result = cursor.fetchone()
        total_sales = result[0] if result[0] else 0

        # Write the total sales to a text file
        with open(output_file_path, 'w') as f:
            f.write(str(total_sales))

        # Close the database connection
        conn.close()

    except Exception as e:
        return {"error": str(e)}
                    
# Define the OpenAI call function
async def call_openai(task_description: str):
    # Define the JSON payload for the OpenAI API
    try:
        payload={"model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": task_description}],
                "tools": tools,
                "tool_choice": "auto",}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENAI_URL,
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data
    except Exception as e:
        return {"error": str(e)} 

# Define the endpoint to automate the tasks
@app.post("/run")
async def automate_tasks(task_description: str = Query(None,alias='task')):
    try:
        response_data = await call_openai(task_description)
            
        if response_data:
            # Example of extracting function and arguments (adjust based on actual OpenAI response format)
            function_name = response_data['choices'][0]['message']['tool_calls'][0]['function']['name']
            arguments = json.loads(response_data['choices'][0]['message']['tool_calls'][0]['function']['arguments'])

            # Map the function name to the actual function and call it with the arguments
            if function_name == "run_file":
                email_id = arguments['email_id']
                file_name = arguments['file_name']
                file_url = arguments['file_url']
                result = await run_python(email_id,file_name,file_url)
                return result
            elif function_name == "format_file":
                file_name = arguments['file_name']
                result = await format_file(file_name)
                return result
            elif function_name == "day_count":
                input_file_name = arguments['input_file_name']
                day_name = arguments["day_name"]
                output_file_name = arguments['output_file_name']
                result = await day_count(input_file_name,day_name,output_file_name)
                return result
            elif function_name == "sort_contacts":
                input_file_name = arguments['input_file_name']
                output_file_name = arguments['output_file_name']
                keys = arguments["keys"]
                result = await sort_contacts(input_file_name,output_file_name,keys)
                return result
            elif function_name == "write_log":
                input_file_path = arguments['input_file_path']
                output_file_name = arguments['output_file_name']
                file_ext = arguments['file_ext']
                number_of_files = arguments['number_of_files']
                timeframe = arguments['timeframe']
                line_number = arguments['line_number']
                result = await write_log(input_file_path,file_ext,output_file_name,number_of_files,timeframe,line_number)
                return result
            elif function_name == "find_style_markdown":
                input_file_path = arguments['input_file_path']
                output_file_name = arguments['output_file_name']
                file_ext = arguments['file_ext']
                symbol = arguments['symbol']
                result = await find_style_markdown(input_file_path,file_ext,output_file_name,symbol)
                return result
            elif function_name == "extract_sender":
                output_file_name = arguments['output_file_name']
                input_file_name = arguments['input_file_name']
                email_field_name = arguments['email_field_name']
                result = await extract_sender(input_file_name,output_file_name,email_field_name)
                return result
            elif function_name == "get_creditcard_details":
                input_file_name = arguments['input_file_name']
                output_file_name = arguments['output_file_name']
                result = await get_creditcard_details(input_file_name,output_file_name)
                return result
            elif function_name == "find_similar":
                input_file_name = arguments['input_file_name']
                output_file_name = arguments['output_file_name']
                result = await find_similar(input_file_name,output_file_name)
                return result
            elif function_name == "get_ticket_sales":
                database_file_name = arguments['database_file_name']
                column_list = arguments['column_list']
                output_file_name = arguments['output_file_name']
                condition = arguments['condition']
                result = await get_ticket_sales(database_file_name,column_list,output_file_name,condition)
                return result
            else:
                return {"error": "Function not recognized"}
        else:
            return {"error": "Failed to extract functions and arguments from OpenAI response"}

    except Exception as e:
        return {"error": str(e)}

# Usage example (run the FastAPI server)
if __name__ == "__main__":
     import uvicorn
     uvicorn.run(app, host="0.0.0.0", port=8000)
