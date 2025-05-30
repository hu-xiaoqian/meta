from typing import Dict, List, Any
import os

import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

import vllm

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

VLLM_TENSOR_PARALLEL_SIZE = 1 
VLLM_GPU_MEMORY_UTILIZATION = 0.85 

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of image search results to retrieve
NUM_IMAGE_SEARCH_RESULTS = 20

class SingleSourceAgent(BaseAgent):

    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        super().__init__(search_pipeline)
        
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for image search functionality")
            
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        
        self.initialize_models()
        
    def initialize_models(self):

        print(f"Initializing {self.model_name} with vLLM...")
        
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Models loaded successfully")

    def get_batch_size(self) -> int:

        return AICROWD_SUBMISSION_BATCH_SIZE
    
    def retrieve_image_context(self, images: List[Image.Image]) -> List[List[dict]]:

        all_search_results = []
        
        for image in images:
            # Use image search to find similar images and their metadata
            try:
                # The search pipeline accepts images directly for image search
                results = self.search_pipeline(image, k=NUM_IMAGE_SEARCH_RESULTS)
                all_search_results.append(results)
            except Exception as e:
                print(f"Image search failed: {e}")
                all_search_results.append([])
        
        return all_search_results
    
    def prepare_inputs_with_context(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        image_search_results: List[List[dict]],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:

        inputs = []
        
        for idx, (query, image, message_history, search_results) in enumerate(
            zip(queries, images, message_histories, image_search_results)
        ):
            # Create system prompt
            SYSTEM_PROMPT = (
                "You are a meticulous image‐questioning assistant. When answering, follow these steps:\n"
                "1. Observe: List 2–3 key details you see in the image.\n"
                "2. Retrieve: If similar‐image context is available, briefly reference it (e.g. “Based on similar image [N]…”).\n"
                "3. Answer: Provide a concise, direct answer based on your observations and any retrieved context.\n"
                "4. Verify: If you lack sufficient evidence or are uncertain, respond exactly with “I don’t know”.\n"
                "Always keep your final answer short and to the point."
            )
            
            # Format retrieved context from similar images
            context = ""
            if search_results:
                context = "\nHere is information about similar images that may be relevant:\n\n"
                for i, result in enumerate(search_results):
                    # Extract relevant metadata from the search result
                    snippet = result.get('page_snippet', '')
                    title = result.get('title', '')
                    
                    if snippet or title:
                        context += f"[Similar Image {i+1}]"
                        if title:
                            context += f" {title}"
                        if snippet:
                            context += f"\n{snippet}"
                        context += "\n\n"
            
            # Structure messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
            
            # Add context if available
            if context.strip():
                messages.append({
                    "role": "user", 
                    "content": f"Consider this additional information:{context}"
                })
            
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:

        print(f"Processing batch of {len(queries)} queries with image search")
        
        # Step 1: Retrieve context from similar images
        image_search_results = self.retrieve_image_context(images)
        
        # Step 2: Prepare inputs with retrieved context
        inputs = self.prepare_inputs_with_context(
            queries, images, image_search_results, message_histories
        )
        
        # Step 3: Generate responses
        print(f"Generating responses for {len(inputs)} queries")
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        
        # Extract and return the generated responses
        responses = [output.outputs[0].text for output in outputs]
        print(f"Successfully generated {len(responses)} responses")
        return responses