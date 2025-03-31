import openai
import tiktoken
import os
import math
import time
import logging
from typing import List, Dict, Optional, Literal

# --- Configuration for Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The Azure Summarizer Class ---

class AzureOpenAISummarizer:
    """
    A class to summarize large texts using the Azure OpenAI API by chunking
    and applying different summarization strategies.

    Attributes:
        azure_deployment_name (str): The name of your deployed model in Azure OpenAI.
        base_model_name (str): The underlying base model name (e.g., "gpt-3.5-turbo")
                                used primarily for tokenization compatibility.
        max_model_tokens (int): The maximum token limit for the deployed model.
        max_output_tokens (int): Reserved token space for the summary output.
        temperature (float): Sampling temperature for the API call.
        client (openai.AzureOpenAI): The Azure OpenAI API client instance.
        tokenizer (tiktoken.Encoding): Tokenizer compatible with the base model.
        chunk_overlap_ratio (float): Percentage of overlap between chunks.
    """

    DEFAULT_SUMMARIZATION_PROMPT = "Please provide a concise summary of the following text:\n\n{text}\n\nConcise Summary:"
    REFINE_INITIAL_PROMPT = "Please provide a concise summary of the following text:\n\n{text}\n\nConcise Summary:"
    REFINE_STEP_PROMPT = (
        "You are given an existing summary up to a certain point:\n{existing_summary}\n"
        "You are also given the next segment of the original text:\n{text}\n"
        "Refine the existing summary by adding the key information from the new text segment.\n"
        "The refined summary should be concise and integrate the new information smoothly.\n\n"
        "Refined Summary:"
    )

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        base_model_name: Optional[str] = None, # Important for tiktoken
        max_model_tokens: Optional[int] = None,
        max_output_tokens: int = 500,
        temperature: float = 0.5,
        chunk_overlap_ratio: float = 0.1, # e.g., 10% overlap
    ):
        """
        Initializes the AzureOpenAISummarizer.

        Args:
            azure_endpoint (Optional[str]): Your Azure OpenAI resource endpoint URL.
                                           Defaults to AZURE_OPENAI_ENDPOINT env var.
            azure_api_key (Optional[str]): Your Azure OpenAI API key.
                                           Defaults to AZURE_OPENAI_API_KEY env var.
            azure_deployment_name (Optional[str]): The name of your model deployment in Azure.
                                                Defaults to AZURE_OPENAI_DEPLOYMENT_NAME env var.
            api_version (Optional[str]): The API version required by Azure OpenAI (e.g., "2024-02-15-preview").
                                          Defaults to OPENAI_API_VERSION env var.
            base_model_name (Optional[str]): The underlying base model name (e.g., "gpt-3.5-turbo", "gpt-4")
                                             needed for accurate token counting with tiktoken.
                                             If None, attempts to infer crudely from deployment name or defaults.
            max_model_tokens (Optional[int]): Override the default max tokens for the deployed model.
                                              If None, attempts to fetch a default based on base_model_name.
            max_output_tokens (int): Max tokens reserved for the API response (summary).
            temperature (float): Sampling temperature for generation (0.0 to 1.0).
            chunk_overlap_ratio (float): Proportion of overlap between consecutive chunks (0.0 to 0.5).
        """
        # --- Get Azure Credentials ---
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = azure_deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION") # Often set globally too

        if not all([self.azure_endpoint, self.azure_api_key, self.azure_deployment_name, self.api_version]):
            raise ValueError(
                "Azure endpoint, API key, deployment name, and API version are required. "
                "Provide them as arguments or set environment variables: "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_DEPLOYMENT_NAME, OPENAI_API_VERSION"
            )

        # --- Base Model Name for Tokenizer ---
        # Try to infer if not provided (this is a guess, better to provide it)
        if not base_model_name:
            if "gpt-4" in self.azure_deployment_name.lower():
                 self.base_model_name = "gpt-4" # Guess
            elif "gpt-35-turbo" in self.azure_deployment_name.lower() or "gpt-3.5-turbo" in self.azure_deployment_name.lower():
                 self.base_model_name = "gpt-3.5-turbo" # Guess
            else:
                 self.base_model_name = "gpt-3.5-turbo" # Default guess
            logging.warning(f"base_model_name not provided, inferred as '{self.base_model_name}' for tokenizer. Provide explicitly for accuracy.")
        else:
            self.base_model_name = base_model_name

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if not 0.0 <= chunk_overlap_ratio <= 0.5:
            raise ValueError("chunk_overlap_ratio must be between 0.0 and 0.5")
        self.chunk_overlap_ratio = chunk_overlap_ratio

        # --- Initialize Azure OpenAI Client ---
        try:
            self.client = openai.AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.api_version,
                # azure_deployment=self.azure_deployment_name # Can be set here or per-call
            )
            logging.info(f"Azure OpenAI client initialized for endpoint: {self.azure_endpoint}")
        except Exception as e: # Catch broader exceptions during init
            logging.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

        # --- Initialize Tokenizer (using base_model_name) ---
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.base_model_name)
            logging.info(f"Tokenizer loaded based on model: {self.base_model_name}")
        except KeyError:
            logging.warning(f"Base model '{self.base_model_name}' not found in tiktoken. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # --- Determine Max Tokens (using base_model_name for defaults) ---
        DEFAULT_MODEL_TOKEN_LIMITS = {
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo": 4096, # Alias often seen in Azure
            "gpt-3.5-turbo-16k": 16384,
            "gpt-35-turbo-16k": 16384, # Alias
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            # Azure often has specific preview names, add if known
            # Use base model name for lookup
        }
        # Use provided max_model_tokens if available, otherwise lookup default based on base_model_name
        resolved_max_tokens = max_model_tokens or DEFAULT_MODEL_TOKEN_LIMITS.get(self.base_model_name, 4096)

        # Check against known Azure limits if possible (example, may need updates)
        # This part is tricky as limits depend on the *specific* Azure deployment/region/quota
        # It's often best for the user to provide max_model_tokens if unsure.
        self.max_model_tokens = resolved_max_tokens
        logging.info(f"Using max_model_tokens: {self.max_model_tokens} (based on base model '{self.base_model_name}' or user input)")


    def _count_tokens(self, text: str) -> int:
        """Counts tokens using the instance's tokenizer."""
        return len(self.tokenizer.encode(text))

    # _get_max_chunk_size remains the same as it depends on max_model_tokens and tokenizer
    def _get_max_chunk_size(self, prompt_template: str) -> int:
        """Calculates the maximum token size for a text chunk based on prompt and output limits."""
        prompt_structure_tokens = self._count_tokens(prompt_template.format(text="", existing_summary="")) # Include refine placeholder
        max_chunk_size = self.max_model_tokens - prompt_structure_tokens - self.max_output_tokens - 15 # Extra buffer for Azure
        if max_chunk_size <= 0:
            raise ValueError("max_output_tokens and prompt structure exceed max_model_tokens.")
        logging.debug(f"Calculated max_chunk_size: {max_chunk_size} tokens")
        return max_chunk_size

    # _split_text remains the same as it depends on max_chunk_size and tokenizer
    def _split_text(self, text: str, max_chunk_size: int) -> List[str]:
        """
        Splits text into chunks respecting max_chunk_size with overlap.
        (Implementation is identical to the previous version, relying on _count_tokens and max_chunk_size)
        """
        # --- Using the same robust splitting logic from the previous example ---
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk_parts = []
        current_chunk_tokens = 0
        overlap_tokens = int(max_chunk_size * self.chunk_overlap_ratio)
        last_part_for_overlap = ""

        for part in paragraphs:
            part_tokens = self._count_tokens(part)

            if part_tokens > max_chunk_size:
                logging.warning(f"A single paragraph ({part_tokens} tokens) exceeds max_chunk_size ({max_chunk_size}). Splitting further.")
                # Fallback: Split the oversized part by tokens
                logging.info(f"Attempting token-based split for oversized part.")
                encoded_part = self.tokenizer.encode(part)
                sub_chunks = []
                start_idx = 0
                current_overlap = 0 # Overlap calculated based on actual token split

                while start_idx < len(encoded_part):
                    # Adjust end_idx considering potential overlap from previous sub-chunk
                    effective_start_idx = max(0, start_idx - current_overlap)
                    end_idx = min(effective_start_idx + max_chunk_size, len(encoded_part))

                    # Ensure we don't create zero-length chunks
                    if effective_start_idx >= end_idx:
                        logging.error(f"Sub-chunk calculation error: start {effective_start_idx}, end {end_idx}. Breaking split.")
                        break

                    sub_chunk_tokens = encoded_part[effective_start_idx:end_idx]
                    sub_chunk_text = self.tokenizer.decode(sub_chunk_tokens)
                    sub_chunks.append(sub_chunk_text)

                    # Calculate overlap for the *next* iteration based on the current sub-chunk
                    current_overlap = 0
                    if end_idx < len(encoded_part) and overlap_tokens > 0:
                         overlap_point = max(effective_start_idx, end_idx - overlap_tokens)
                         # Ensure overlap doesn't exceed the actual remaining tokens
                         current_overlap = min(overlap_tokens, end_idx - overlap_point)


                    start_idx = end_idx # Move start for the next chunk based on the end of the current one

                logging.info(f"Split oversized part into {len(sub_chunks)} sub-chunks.")
                for sub_chunk in sub_chunks:
                    chunks.append(sub_chunk)
                continue

            # Check if adding the current part exceeds the limit
            joiner_tokens = self._count_tokens("\n\n") if current_chunk_parts else 0
            if current_chunk_tokens + part_tokens + joiner_tokens <= max_chunk_size:
                current_chunk_parts.append(part)
                current_chunk_tokens += part_tokens + joiner_tokens
                last_part_for_overlap = part
            else:
                 # Finalize the current chunk
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))

                # Start new chunk, potentially with overlap
                current_chunk_parts = []
                current_chunk_tokens = 0
                overlap_text = ""
                last_part_tokens = self._count_tokens(last_part_for_overlap)
                if last_part_for_overlap and last_part_tokens < overlap_tokens and last_part_tokens < max_chunk_size:
                    overlap_text = last_part_for_overlap
                    logging.debug(f"Adding overlap: '{overlap_text[:50]}...' ({last_part_tokens} tokens)")

                new_chunk_parts = [overlap_text, part] if overlap_text else [part]
                current_chunk_parts.extend(p for p in new_chunk_parts if p)
                current_chunk_tokens = self._count_tokens("\n\n".join(current_chunk_parts))
                last_part_for_overlap = part

                if current_chunk_tokens > max_chunk_size:
                    logging.warning("Single part + overlap exceeds limit. Adding part as its own chunk without overlap.")
                    # Finalize the chunk *before* this problematic part (if it wasn't already)
                    if overlap_text and chunks[-1] != overlap_text : # Check if overlap text itself formed the last chunk
                        # This logic gets complex, assume previous finalize worked. Re-evaluate if issues arise.
                        pass

                    # Start new chunk with just the current part
                    current_chunk_parts = [part]
                    current_chunk_tokens = self._count_tokens(part)
                    last_part_for_overlap = part


        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        logging.info(f"Split text into {len(chunks)} chunks.")
        return chunks


    def _call_openai_api(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Makes a single call to the Azure OpenAI ChatCompletion API."""
        logging.debug(f"Making API call to Azure deployment: {self.azure_deployment_name}. Prompt length: {self._count_tokens(prompt)} tokens. Max output: {max_tokens} tokens.")
        try:
            response = self.client.chat.completions.create(
                # *** Use deployment name for the model parameter in Azure ***
                model=self.azure_deployment_name,
                messages=[
                    # {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                # top_p=1,
                # frequency_penalty=0,
                # presence_penalty=0
            )
            summary = response.choices[0].message.content.strip()
            completion_tokens = response.usage.completion_tokens if response.usage else -1
            prompt_tokens = response.usage.prompt_tokens if response.usage else -1
            logging.debug(f"API call successful. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            logging.debug(f"Received summary snippet: '{summary[:100]}...'")
            return summary
        # More specific Azure error handling might be needed depending on common issues
        except openai.APIError as e:
            logging.error(f"Azure OpenAI API Error: {e}")
        except openai.RateLimitError as e:
            logging.warning(f"Azure Rate limit exceeded: {e}. Waiting and retrying...")
            # Implement exponential backoff or check headers for retry-after time
            time.sleep(20)
            return None # Or re-raise/retry
        except openai.AuthenticationError as e:
            logging.error(f"Azure Authentication Error: {e}. Check API key and endpoint.")
            raise # Authentication errors are critical
        except Exception as e:
            logging.error(f"An unexpected error occurred during Azure API call: {e}")

        return None # Return None on failure

    # summarize method remains the same structure, calling the updated _call_openai_api
    def summarize(
        self,
        text: str,
        strategy: Literal["map_reduce", "refine"] = "map_reduce",
        summarization_prompt: Optional[str] = None,
        refine_prompt: Optional[str] = None,
        initial_prompt: Optional[str] = None, # Only used for refine
        max_retries: int = 2,
        retry_delay: int = 10, # seconds
    ) -> Optional[str]:
        """
        Summarizes the given text using the specified strategy via Azure OpenAI.

        (Implementation is identical to the previous version, but it now calls
         the Azure-configured _call_openai_api method)

        Args:
            text (str): The large text to summarize.
            strategy (Literal["map_reduce", "refine"]): The summarization strategy.
            summarization_prompt (Optional[str]): Custom prompt for standard/map_reduce.
            refine_prompt (Optional[str]): Custom prompt for the refine step.
            initial_prompt (Optional[str]): Custom prompt for the first step of refine.
            max_retries (int): Maximum number of retries for failed API calls per chunk.
            retry_delay (int): Delay in seconds between retries.

        Returns:
            Optional[str]: The final summary, or None if summarization failed.
        """
        logging.info(f"Starting summarization with strategy: {strategy} using Azure deployment: {self.azure_deployment_name}")

        # --- Map Reduce Strategy ---
        if strategy == "map_reduce":
            prompt_template = summarization_prompt or self.DEFAULT_SUMMARIZATION_PROMPT
            max_chunk_size = self._get_max_chunk_size(prompt_template)
            chunks = self._split_text(text, max_chunk_size)
            individual_summaries: List[str] = []

            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i+1}/{len(chunks)} for map_reduce...")
                prompt = prompt_template.format(text=chunk)
                summary = None
                for attempt in range(max_retries + 1):
                     summary = self._call_openai_api(prompt, self.max_output_tokens)
                     if summary is not None:
                         individual_summaries.append(summary)
                         break
                     elif attempt < max_retries:
                         logging.warning(f"API call failed for chunk {i+1}, attempt {attempt+1}. Retrying in {retry_delay}s...")
                         time.sleep(retry_delay)
                     else:
                         logging.error(f"Failed to summarize chunk {i+1} after {max_retries} retries.")

                time.sleep(0.5) # Smaller delay maybe sufficient for Azure

            if not individual_summaries:
                logging.error("No summaries were generated from the chunks.")
                return None

            combined_summaries = "\n\n".join(individual_summaries)
            combined_tokens = self._count_tokens(combined_summaries)
            logging.info(f"Combined intermediate summaries ({len(individual_summaries)}). Total tokens: {combined_tokens}")

            # Check if combined summaries need final summarization
            final_summary = combined_summaries
            final_prompt_template = prompt_template # Use same template for final reduction
            if combined_tokens > self._get_max_chunk_size(final_prompt_template):
                logging.info("Combined summaries exceed limit. Performing final reduction summarization.")
                final_prompt = final_prompt_template.format(text=combined_summaries)
                # Ensure the combined text itself fits within the *absolute* model limits for the final call
                # If combined_tokens > self.max_model_tokens, even this single call will fail without further chunking.
                if combined_tokens > self.max_model_tokens - self.max_output_tokens - 15:
                     logging.error(f"Combined summaries ({combined_tokens} tokens) are too large even for a single final API call. Cannot reduce further with this method.")
                     # Consider alternative: return truncated combined summaries, or implement recursive reduction.
                     return combined_summaries # Return combined but un-reduced summary

                final_summary_result = None
                for attempt in range(max_retries + 1):
                    final_summary_result = self._call_openai_api(final_prompt, self.max_output_tokens)
                    if final_summary_result is not None:
                        final_summary = final_summary_result
                        break
                    elif attempt < max_retries:
                         logging.warning(f"Final reduction API call failed, attempt {attempt+1}. Retrying in {retry_delay}s...")
                         time.sleep(retry_delay)
                    else:
                         logging.error("Failed to perform final reduction summarization.")
                         # Decide: return the un-reduced summary or None?
                         return combined_summaries # Return combined but un-reduced

                logging.info("Final summary generated.")
            else:
                logging.info("Combined summaries fit within limits. No final reduction needed.")

            return final_summary

        # --- Refine Strategy ---
        elif strategy == "refine":
            initial_prompt_template = initial_prompt or self.REFINE_INITIAL_PROMPT
            refine_prompt_template = refine_prompt or self.REFINE_STEP_PROMPT
            max_chunk_size = self._get_max_chunk_size(refine_prompt_template)
            chunks = self._split_text(text, max_chunk_size)
            current_summary = None

            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i+1}/{len(chunks)} for refine...")

                if i == 0:
                    prompt = initial_prompt_template.format(text=chunk)
                else:
                    if current_summary is None:
                        logging.error(f"Cannot refine chunk {i+1} because previous summary is missing.")
                        continue
                    prompt = refine_prompt_template.format(existing_summary=current_summary, text=chunk)

                prompt_tokens = self._count_tokens(prompt)
                if prompt_tokens > self.max_model_tokens - self.max_output_tokens - 15:
                     logging.warning(f"Refine prompt for chunk {i+1} ({prompt_tokens} tokens) is too large (max: {self.max_model_tokens}). Skipping refine step.")
                     continue

                refined_summary_part = None
                for attempt in range(max_retries + 1):
                     refined_summary_part = self._call_openai_api(prompt, self.max_output_tokens)
                     if refined_summary_part is not None:
                          current_summary = refined_summary_part
                          break
                     elif attempt < max_retries:
                          logging.warning(f"API call failed for refine chunk {i+1}, attempt {attempt+1}. Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Failed to refine chunk {i+1} after {max_retries} retries.")
                          break # Stop retrying for this chunk

                time.sleep(0.5)

            logging.info("Refine process completed.")
            return current_summary

        else:
            logging.error(f"Unknown summarization strategy: {strategy}")
            raise ValueError("Strategy must be 'map_reduce' or 'refine'")


# --- Example Usage for Azure ---
if __name__ == "__main__":
    # --- REQUIRED AZURE CONFIGURATION ---
    # Best practice: Set these as environment variables
    # export AZURE_OPENAI_ENDPOINT='YOUR_AZURE_ENDPOINT'
    # export AZURE_OPENAI_API_KEY='YOUR_AZURE_API_KEY'
    # export AZURE_OPENAI_DEPLOYMENT_NAME='YOUR_DEPLOYMENT_NAME' # e.g., 'my-gpt4-deployment'
    # export OPENAI_API_VERSION='2024-02-15-preview' # Or your required version

    # --- OPTIONAL ---
    # Provide the base model name if it cannot be inferred from the deployment name
    # export BASE_MODEL_NAME='gpt-4' # Example

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("OPENAI_API_VERSION")
    base_model = os.getenv("BASE_MODEL_NAME") # Optional, will be inferred if None

    # Check if required Azure variables are set
    if not all([azure_endpoint, azure_api_key, azure_deployment, api_version]):
        print("ERROR: Please set required Azure environment variables:")
        print(" - AZURE_OPENAI_ENDPOINT")
        print(" - AZURE_OPENAI_API_KEY")
        print(" - AZURE_OPENAI_DEPLOYMENT_NAME")
        print(" - OPENAI_API_VERSION")
    else:
        # 1. Load text
        try:
            with open("large_document.txt", "r", encoding="utf-8") as f:
                my_large_text = f.read()
            logging.info("Loaded text from large_document.txt")
        except FileNotFoundError:
            logging.warning("large_document.txt not found. Using placeholder text.")
            base_sentence = "This sentence represents content within a large document processed via Azure OpenAI. It demonstrates chunking for summarization. "
            my_large_text = base_sentence * 600 # Make it large enough to require chunking


        # 2. Initialize the Azure summarizer
        try:
            summarizer = AzureOpenAISummarizer(
                # Pass credentials explicitly (optional if env vars are set)
                # azure_endpoint=azure_endpoint,
                # azure_api_key=azure_api_key,
                # azure_deployment_name=azure_deployment,
                # api_version=api_version,
                base_model_name=base_model, # Pass if set, otherwise None to infer
                 # Optional: Explicitly set max tokens if default lookup is wrong for your deployment
                # max_model_tokens=8192,
                max_output_tokens=750,
                temperature=0.3,
                chunk_overlap_ratio=0.05
            )

            # 3. Summarize using Map Reduce
            logging.info("\n--- Starting Azure Map Reduce Summarization ---")
            final_summary_map_reduce = summarizer.summarize(my_large_text, strategy="map_reduce")

            if final_summary_map_reduce:
                print("\n--- Final Summary (Azure Map Reduce) ---")
                print(final_summary_map_reduce)
                # with open("azure_summary_map_reduce.txt", "w", encoding="utf-8") as f:
                #     f.write(final_summary_map_reduce)
            else:
                print("\n--- Azure Map Reduce Summarization Failed ---")

            # 4. Summarize using Refine
            # logging.info("\n--- Starting Azure Refine Summarization ---")
            # final_summary_refine = summarizer.summarize(my_large_text, strategy="refine")
            #
            # if final_summary_refine:
            #     print("\n--- Final Summary (Azure Refine) ---")
            #     print(final_summary_refine)
            #     # with open("azure_summary_refine.txt", "w", encoding="utf-8") as f:
            #     #     f.write(final_summary_refine)
            # else:
            #     print("\n--- Azure Refine Summarization Failed ---")

        except ValueError as e:
             print(f"Initialization Error: {e}")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
                             
