# llm_data_analysis

This repository contains code I wrote to try to use OpenAI's language models to analyze/look for patterns in data. At the time I wrote the code, the context windows of the models I had API access to were so small that they couldn't fit even moderate size datasets. As a result, for datasets that are too large to fit in the model context window, the program breaks them up into chunks that are small enough to fit into the model’s context window and then submits a sequence of prompts containing these data chunks to the OpenAI API. Basically, the program works by: 

- Submitting the user prompt and an initial chunk of the data to the OpenAI API;
- Submitting the next chunk of data to the OpenAI API along with the model’s response to the previous prompt and an instruction to analyze the new data and integrate this analysis with the prior analysis; and
- Terminating if the end of the dataset has been reached, otherwise going back to the step above.

I tested out the program using a mock dataset (a sample sequence of prompts and model responses is in the sales_data_analysis_3.txt file in this repository). My impression from this and a few other tests I did is that while this approach can result in decent analyses of datasets that don’t fit in current models’ context windows, it’s maybe not reliable enough to use for most applications. This is arguably not that surprising, since it's perhaps unlikely that the data used to train OpenAI's language models contains a lot of raw datasets immediately followed by high quality analysis of those datasets.  
