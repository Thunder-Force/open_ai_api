import openai
import tiktoken
import os
import math
import time
import logging
from typing import List, Dict, Optional, Literal

# --- Configuration for Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- The Summarizer Class ---

class OpenAISummarizer:
    """
    A class to summarize large texts using the OpenAI API by chunking
    and applying different summarization strategies.

    Attributes:
        model_name (str): The OpenAI model to use for summarization.
        max_model_tokens (int): The maximum token limit for the specified model.
        max_output_tokens (int): Reserved token space for the summary output.
        temperature (float): Sampling temperature for the API call.
        client (openai.OpenAI): The OpenAI API client instance.
        tokenizer (tiktoken.Encoding): Tokenizer for the specified model.
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
        model_name: str = "gpt-3.5-turbo",
        max_model_tokens: Optional[int] = None,
        max_output_tokens: int = 500,
        temperature: float = 0.5,
        chunk_overlap_ratio: float = 0.1, # e.g., 10% overlap
        api_key: Optional[str] = None,
    ):
        """
        Initializes the OpenAISummarizer.

        Args:
            model_name (str): OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4").
            max_model_tokens (Optional[int]): Override the default max tokens for the model.
                                              If None, attempts to fetch a default value.
            max_output_tokens (int): Max tokens reserved for the API response (summary).
            temperature (float): Sampling temperature for generation (0.0 to 1.0).
            chunk_overlap_ratio (float): Proportion of overlap between consecutive chunks (0.0 to 0.5).
            api_key (Optional[str]): OpenAI API key. If None, uses OPENAI_API_KEY env variable.
        """
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if not 0.0 <= chunk_overlap_ratio <= 0.5:
             raise ValueError("chunk_overlap_ratio must be between 0.0 and 0.5")
        self.chunk_overlap_ratio = chunk_overlap_ratio

        # Initialize OpenAI client (handles API key from env var automatically if api_key is None)
        try:
             self.client = openai.OpenAI(api_key=api_key)
        except openai.OpenAIError as e:
             logging.error(f"Failed to initialize OpenAI client: {e}")
             raise

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            logging.info(f"Tokenizer loaded for model: {self.model_name}")
        except KeyError:
            logging.warning(f"Model {self.model_name} not found in tiktoken. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Determine max tokens for the model
        # Note: These are common values, but check OpenAI docs for the most up-to-date limits
        DEFAULT_MODEL_TOKEN_LIMITS = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0125": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo-preview": 128000, # Input tokens for preview model
            "gpt-4-1106-preview": 128000,
        }
        self.max_model_tokens = max_model_tokens or DEFAULT_MODEL_TOKEN_LIMITS.get(model_name, 4096)
        logging.info(f"Using max_model_tokens: {self.max_model_tokens}")


    def _count_tokens(self, text: str) -> int:
        """Counts tokens using the instance's tokenizer."""
        return len(self.tokenizer.encode(text))

    def _get_max_chunk_size(self, prompt_template: str) -> int:
        """Calculates the maximum token size for a text chunk based on prompt and output limits."""
        # Count tokens in the template structure (without the actual text)
        prompt_structure_tokens = self._count_tokens(prompt_template.format(text=""))
        # Add tokens for refine strategy's existing summary placeholder if applicable
        if "{existing_summary}" in prompt_template:
            # Estimate a reasonable size for the existing summary placeholder part of the prompt
            prompt_structure_tokens += self._count_tokens(self.REFINE_STEP_PROMPT.format(existing_summary="", text="")) - prompt_structure_tokens

        max_chunk_size = self.max_model_tokens - prompt_structure_tokens - self.max_output_tokens - 10 # Add a small buffer
        if max_chunk_size <= 0:
            raise ValueError("max_output_tokens and prompt structure exceed max_model_tokens.")
        logging.debug(f"Calculated max_chunk_size: {max_chunk_size} tokens")
        return max_chunk_size

    def _split_text(self, text: str, max_chunk_size: int) -> List[str]:
        """
        Splits text into chunks respecting max_chunk_size with overlap.
        Attempts to split by paragraphs first, then sentences, then falls back to words.
        """
        chunks = []
        # Use tiktoken's decode/encode for more accurate splitting if needed,
        # but splitting by logical units (paragraphs, sentences) is often preferred.
        # This implementation focuses on logical splits first.

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk_parts = []
        current_chunk_tokens = 0
        overlap_tokens = int(max_chunk_size * self.chunk_overlap_ratio)
        last_part_for_overlap = ""

        for part in paragraphs: # Could be paragraphs, sentences, or words later
            part_tokens = self._count_tokens(part)

            if part_tokens > max_chunk_size:
                logging.warning(f"A single paragraph ({part_tokens} tokens) exceeds max_chunk_size ({max_chunk_size}). Splitting further.")
                # Fallback 1: Split the oversized paragraph by sentences
                sentences = part.split('. ') # Simple sentence split
                temp_parts = [s + '.' if not s.endswith('.') else s for s in sentences if s]
                # Process these smaller parts recursively or iteratively
                # For simplicity here, we'll just process them in the next loop iteration
                # A more robust way would handle this recursively or refine splitting logic.
                # We will insert these back into the loop logic implicitly by continuing
                # But a truly robust splitter might be needed (like LangChain's)
                # Let's just split the oversized part by tokens for this example:
                logging.info(f"Attempting token-based split for oversized part.")
                encoded_part = self.tokenizer.encode(part)
                sub_chunks = []
                start_idx = 0
                while start_idx < len(encoded_part):
                    end_idx = min(start_idx + max_chunk_size, len(encoded_part))
                    sub_chunk_tokens = encoded_part[start_idx:end_idx]
                    sub_chunk_text = self.tokenizer.decode(sub_chunk_tokens)
                    sub_chunks.append(sub_chunk_text)
                    start_idx += max_chunk_size - overlap_tokens # Move window with overlap
                    if start_idx >= len(encoded_part) - overlap_tokens and start_idx < len(encoded_part):
                         start_idx = len(encoded_part) # Avoid tiny last chunk due to overlap calculation

                logging.info(f"Split oversized part into {len(sub_chunks)} sub-chunks.")
                # Process these sub-chunks immediately
                for sub_chunk in sub_chunks:
                    chunks.append(sub_chunk) # Add directly as they fit
                continue # Skip the rest of the loop for the original oversized part


            # Check if adding the current part exceeds the limit
            if current_chunk_tokens + part_tokens + (1 if current_chunk_parts else 0) <= max_chunk_size: # +1 for potential '\n\n' joiner
                current_chunk_parts.append(part)
                current_chunk_tokens += part_tokens + (1 if current_chunk_parts else 0)
                last_part_for_overlap = part # Store the last added part
            else:
                # Finalize the current chunk
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))

                # Start new chunk, potentially with overlap
                current_chunk_parts = []
                current_chunk_tokens = 0
                overlap_text = ""

                # Simple overlap: Use the last part if it fits within overlap budget
                # More sophisticated: Take trailing sentences/tokens up to overlap_tokens
                last_part_tokens = self._count_tokens(last_part_for_overlap)
                if last_part_for_overlap and last_part_tokens < overlap_tokens and last_part_tokens < max_chunk_size:
                     overlap_text = last_part_for_overlap
                     logging.debug(f"Adding overlap: '{overlap_text[:50]}...' ({last_part_tokens} tokens)")


                # Start new chunk with overlap (if any) and the current part
                new_chunk_parts = [overlap_text, part] if overlap_text else [part]
                current_chunk_parts.extend(p for p in new_chunk_parts if p) # Add non-empty parts
                current_chunk_tokens = self._count_tokens("\n\n".join(current_chunk_parts))
                last_part_for_overlap = part # Update last part for next potential overlap

                # If the new part *with* overlap *still* exceeds limit (edge case)
                if current_chunk_tokens > max_chunk_size:
                     logging.warning("Single part + overlap exceeds limit. Adding part as its own chunk without overlap.")
                     # Add previous chunk without this part
                     if chunks[-1] != "\n\n".join(current_chunk_parts[:-1]): # Avoid duplicates if previous logic added it
                        # This logic path needs careful review, might be complex edge case
                        pass # Simplified: Assume previous logic handled the chunk before overflow part

                     # Start new chunk with just the current part
                     current_chunk_parts = [part]
                     current_chunk_tokens = self._count_tokens(part)
                     last_part_for_overlap = part


        # Add the last remaining chunk
        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        logging.info(f"Split text into {len(chunks)} chunks.")
        # for i, chunk in enumerate(chunks):
        #     logging.debug(f"Chunk {i+1} token count: {self._count_tokens(chunk)}")
        return chunks


    def _call_openai_api(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Makes a single call to the OpenAI ChatCompletion API."""
        logging.debug(f"Making API call. Prompt length: {self._count_tokens(prompt)} tokens. Max output: {max_tokens} tokens.")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    # You could add a system prompt here if desired
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
            completion_tokens = response.usage.completion_tokens
            prompt_tokens = response.usage.prompt_tokens
            logging.debug(f"API call successful. Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            logging.debug(f"Received summary snippet: '{summary[:100]}...'")
            return summary
        except openai.APIError as e:
            logging.error(f"OpenAI API Error: {e}")
        except openai.RateLimitError:
            logging.warning("Rate limit exceeded. Waiting and retrying...")
            time.sleep(20) # Wait 20 seconds before potential retry (implement retry logic if needed)
            # Consider implementing exponential backoff for retries
            return None # Or re-raise/retry
        except Exception as e:
            logging.error(f"An unexpected error occurred during API call: {e}")

        return None # Return None on failure

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
        Summarizes the given text using the specified strategy.

        Args:
            text (str): The large text to summarize.
            strategy (Literal["map_reduce", "refine"]): The summarization strategy.
                - "map_reduce": Summarize chunks independently, then combine and summarize the results.
                - "refine": Summarize the first chunk, then iteratively refine the summary with subsequent chunks.
            summarization_prompt (Optional[str]): Custom prompt for standard/map_reduce summarization.
                                                  Must include "{text}". Defaults to class default.
            refine_prompt (Optional[str]): Custom prompt for the refine step. Must include
                                           "{existing_summary}" and "{text}". Defaults to class default.
            initial_prompt (Optional[str]): Custom prompt for the first step of refine. Must include
                                            "{text}". Defaults to class default.
            max_retries (int): Maximum number of retries for failed API calls per chunk.
            retry_delay (int): Delay in seconds between retries.


        Returns:
            Optional[str]: The final summary, or None if summarization failed.
        """
        logging.info(f"Starting summarization with strategy: {strategy}")

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
                         break # Success
                     elif attempt < max_retries:
                         logging.warning(f"API call failed for chunk {i+1}, attempt {attempt+1}. Retrying in {retry_delay}s...")
                         time.sleep(retry_delay)
                     else:
                         logging.error(f"Failed to summarize chunk {i+1} after {max_retries} retries.")
                         # Decide: continue with missing chunk, or abort? For now, continue.

                time.sleep(1) # Small delay to help avoid rate limits

            if not individual_summaries:
                logging.error("No summaries were generated from the chunks.")
                return None

            combined_summaries = "\n\n".join(individual_summaries)
            combined_tokens = self._count_tokens(combined_summaries)
            logging.info(f"Combined intermediate summaries ({len(individual_summaries)}). Total tokens: {combined_tokens}")

            # Check if combined summaries need final summarization
            # Use a slightly smaller max_chunk_size check here as it's the final step.
            if combined_tokens > self._get_max_chunk_size(prompt_template):
                logging.info("Combined summaries exceed limit. Performing final reduction summarization.")
                # We might need to chunk the *combined summaries* if they are massive.
                # Simplified: Assume one final summarization call is sufficient.
                # A truly robust version might recursively call summarize() here.
                final_prompt = prompt_template.format(text=combined_summaries)
                final_summary = None
                for attempt in range(max_retries + 1):
                    final_summary = self._call_openai_api(final_prompt, self.max_output_tokens)
                    if final_summary is not None:
                        break
                    elif attempt < max_retries:
                         logging.warning(f"Final reduction API call failed, attempt {attempt+1}. Retrying in {retry_delay}s...")
                         time.sleep(retry_delay)
                    else:
                         logging.error("Failed to perform final reduction summarization.")
                         return None # Failed final step
                logging.info("Final summary generated.")
                return final_summary
            else:
                logging.info("Combined summaries fit within limits. No final reduction needed.")
                return combined_summaries

        elif strategy == "refine":
            initial_prompt_template = initial_prompt or self.REFINE_INITIAL_PROMPT
            refine_prompt_template = refine_prompt or self.REFINE_STEP_PROMPT

            # Calculate max_chunk_size based on the potentially larger refine prompt
            max_chunk_size = self._get_max_chunk_size(refine_prompt_template)
            chunks = self._split_text(text, max_chunk_size)
            current_summary = None

            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i+1}/{len(chunks)} for refine...")

                if i == 0:
                    # First chunk
                    prompt = initial_prompt_template.format(text=chunk)
                else:
                    # Subsequent chunks
                    if current_summary is None:
                        logging.error(f"Cannot refine chunk {i+1} because previous summary is missing.")
                        continue # Skip this chunk if previous failed
                    prompt = refine_prompt_template.format(existing_summary=current_summary, text=chunk)

                # Estimate prompt tokens - if it exceeds limit, something went wrong (e.g. previous summary too large)
                prompt_tokens = self._count_tokens(prompt)
                if prompt_tokens > self.max_model_tokens - self.max_output_tokens - 10:
                     logging.warning(f"Refine prompt for chunk {i+1} ({prompt_tokens} tokens) is too large. Skipping refine step for this chunk.")
                     # Op

  # Option: Could try summarizing the chunk independently and appending?
                     # For now, we just keep the previous summary.
                     continue


                refined_summary_part = None
                for attempt in range(max_retries + 1):
                     refined_summary_part = self._call_openai_api(prompt, self.max_output_tokens)
                     if refined_summary_part is not None:
                          current_summary = refined_summary_part # Update summary for next iteration
                          break
                     elif attempt < max_retries:
                          logging.warning(f"API call failed for refine chunk {i+1}, attempt {attempt+1}. Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Failed to refine chunk {i+1} after {max_retries} retries.")
                          # Keep the previous summary if this step fails
                          break # Stop retrying for this chunk

                time.sleep(1) # Small delay

            logging.info("Refine process completed.")
            return current_summary

        else:
            logging.error(f"Unknown summarization strategy: {strategy}")
            raise ValueError("Strategy must be 'map_reduce' or 'refine'")


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure OPENAI_API_KEY environment variable is set
    # export OPENAI_API_KEY='your-actual-api-key'
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
    else:
        # 1. Load your large text document
        try:
            with open("large_document.txt", "r", encoding="utf-8") as f:
                my_large_text = f.read()
            logging.info("Loaded text from large_document.txt")
        except FileNotFoundError:
            logging.warning("large_document.txt not found. Using placeholder text.")
            # Create a long placeholder text
            base_sentence = "This sentence serves as a placeholder for a much larger document that requires summarization using advanced techniques like chunking and API calls. "
            my_large_text = base_sentence * 500 # Repeat to make it large


        # 2. Initialize the summarizer
        # Using a model with a larger context window if available and affordable can reduce chunking needs
        summarizer = OpenAISummarizer(
            model_name="gpt-3.5-turbo", # Or try "gpt-4-turbo-preview" if you have access & budget
            # max_model_tokens=16384, # Explicitly set if needed for gpt-3.5-turbo-16k
            max_output_tokens=750,   # Allow slightly longer summaries per chunk / final
            temperature=0.3,         # Lower temp for more focused summaries
            chunk_overlap_ratio=0.05 # Smaller overlap
        )

        # 3. Summarize using a strategy
        logging.info("\n--- Starting Map Reduce Summarization ---")
        final_summary_map_reduce = summarizer.summarize(my_large_text, strategy="map_reduce")

        if final_summary_map_reduce:
            print("\n--- Final Summary (Map Reduce) ---")
            print(final_summary_map_reduce)
            # Optional: Save the summary
            # with open("summary_map_reduce.txt", "w", encoding="utf-8") as f:
            #     f.write(final_summary_map_reduce)
        else:
            print("\n--- Map Reduce Summarization Failed ---")


        logging.info("\n--- Starting Refine Summarization ---")
        final_summary_refine = summarizer.summarize(my_large_text, strategy="refine")

        if final_summary_refine:
            print("\n--- Final Summary (Refine) ---")
            print(final_summary_refine)
            # Optional: Save the summary
            # with open("summary_refine.txt", "w", encoding="utf-8") as f:
            #     f.write(final_summary_refine)
        else:
            print("\n--- Refine Summarization Failed ---")
