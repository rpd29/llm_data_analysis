import argparse # import argparse module to enable user to pass labeled arguments to program through command line
import os # import os to enable retrieval of API key from environment variable
import pandas as pd # Import pandas to use to read data from files
import openai # import openai module to use for api calls
import tiktoken # import openai tokenizer to use to determine how many tokens are in a text string and to decode tokenized strings


# The code below parses arguments provided by user. 
parser = argparse.ArgumentParser() #Create parser object

parser.add_argument("--data_filepath", type=str, required = True, help = "filepath of file containing data user wants analyzed.")
parser.add_argument("--log_filepath", type=str, required = True, help = "filepath of directory where user wishes log file containing sequences of prompts and model responses to be stored.")
parser.add_argument("--prompt", type=str, required = True, help = "prompt specifying what analysis user wants model to do.")


args = parser.parse_args() # get object containing values of all variables specified by user


# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") # Set value of OpenAI API key by retrieving value from environment variable. 


def analyze_data(data_filepath, log_filepath, prompt):

	# Load Excel file as a pandas dataframe
	data = pd.read_excel(data_filepath) 

	# Convert the dataframe to a string
	data_str = data.to_string()

	# Determine if data is too long to fit into a single prompt
	num_tokens_response = 1000 # Define amount of token space to leave for model's response
	model_context_length = 4000 # Define context length of model being used
	encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # Create encoding object to use for tokenization and detokenization
	num_tokens_prompt = len(encoding.encode(prompt)) # Compute number of tokens in prompt and assign to a variable
	num_tokens_data = len(encoding.encode(data_str)) # Compute number of tokens in data and assign to a variable
	space_for_data = model_context_length - num_tokens_response - num_tokens_prompt # Calculate space for data in prompt 

	if num_tokens_data <= space_for_data: # If there's enough space for all the data, submit prompt and data to OpenAI API
		prompt_and_data = prompt + data_str
		response = openai.ChatCompletion.create(
			model="gpt-3.5-turbo", 
			temperature = 0,
			messages=[
				{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
				{"role": "user", "content": prompt_and_data}], 
		)['choices'][0]['message']['content']

		log = "PROMPT:\n\n" + prompt_and_data + "\n\nRESPONSE:\n\n" + response

		with open(log_filepath, "w") as log_file: # Write prompt and response to log file
			log_file.write(log)


	else: # If there's not enough space for all the data, have model analyze data incrementally by breaking it into chunks and making a sequence of API calls
		data_chunk = data_str[0:space_for_data] # Extract first chunk of data small enough to fit in context window
		truncate_index = data_chunk.rfind('\n') # Find index where last complete tuple/row of data in data_chunk ends
		data_chunk = data_chunk[0:truncate_index] # Truncate data chunk so that it contains only complete tuples/rows
		prompt_and_data = prompt + data_chunk # Append data_chunk to prompt
		response = openai.ChatCompletion.create( # Submit prompt along with first chunk of data to model API
			model="gpt-3.5-turbo", 
			temperature = 0,
			messages=[
				{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
				{"role": "user", "content": prompt_and_data}], 
		)['choices'][0]['message']['content']

		log = "PROMPT:\n\n" + prompt_and_data + "\n\nRESPONSE:\n\n" + response + "\n\n" # Start a log of prompts and responses
		
		attribute_names = data_str[0:data_str.find('\n')] # Extract the attribute names/column headers from the data string
		continuation_prompt = "You are analyzing a table of data that is too large to fit in your context window. Below is your analysis of the data you've seen so far. Below that is the next chunk of data (including the column/attribute names), which you haven't yet seen. Please continue analyzing this dataset. Please make sure your response describes interesting features of all parts of the dataset that you have seen so far, not just interesting features of the chunk of data below.\n\nYour analysis of the data you've seen so far:\n\n" # Prompt to tell model it is in the process of analyzing a large dataset in multiple steps
		next_chunk_prompt = "\n\nThe next chunk of data (which you haven't seen yet):\n\n"
		old_truncate_index = truncate_index
		reached_end_of_data = False # Define variable used to determine when there are no more chunks of data left to submit to model API
		
		while reached_end_of_data == False: # Continue feeding chunks of data into model and update analysis until reaching end of data
			space_for_data = model_context_length - num_tokens_response - len(continuation_prompt) - len(response) - len(next_chunk_prompt) - len(attribute_names)
			new_truncate_index = old_truncate_index + data_str[old_truncate_index : old_truncate_index + space_for_data].rfind('\n') # Calculate where next chunk of data should end so that it fits in context window and contains only complete tuples/rows
			
			if old_truncate_index + space_for_data >= len(data_str): 
				data_chunk = data_str[old_truncate_index:] # Include remainder of data in data_chunk if there's enough space
				reached_end_of_data = True # Set loop variable to true so that loop exits after last iteration
			
			else: # Extract next chunk of data if there's not enough space for rest of data
				data_chunk = data_str[old_truncate_index:new_truncate_index] # Extract next chunk of data
			
			prompt_and_data = continuation_prompt + response + next_chunk_prompt + attribute_names + data_chunk
			
			response = openai.ChatCompletion.create( # Submit prompt along with first chunk of data to model API
				model="gpt-3.5-turbo", 
				temperature = 0,
				messages=[
					{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, 
					{"role": "user", "content": prompt_and_data}], 
			)['choices'][0]['message']['content']

			log = log + "PROMPT:\n\n" + prompt_and_data + "\n\nRESPONSE:\n\n" + response + "\n\n" # Append prompt and response to log

			old_truncate_index = new_truncate_index # Update old_truncate_index variable for next iteration

			if reached_end_of_data == True: # Write prompt and response to log file
				with open(log_filepath, "w") as log_file: 
					log_file.write(log)


analyze_data(data_filepath = args.data_filepath, log_filepath = args.log_filepath, prompt = args.prompt)

