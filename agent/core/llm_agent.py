# core/llm_agent.py

import logging
from transformers import pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class LLMComponentAgent:
    def __init__(self, model="mistral-7b"):
        self.model = model
        self.generator = pipeline("text-generation", model=self.model)
        logging.info(f"Initialized LLMComponentAgent with local Hugging Face model: {model}")

    def build_prompt(self, log_text: str, code_context: list) -> str:
        """
        Builds a structured prompt for the LLM using the log and related code.
        """
        logging.info("Building prompt for LLM analysis")
        prompt = "You are an expert debugging assistant.\n"
        prompt += "Given a log and relevant code, identify the issue and suggest a fix.\n\n"

        prompt += f"🪵 LOG:\n{log_text.strip()}\n\n"

        prompt += "💡 RELATED CODE CONTEXT:\n"
        for idx, chunk in enumerate(code_context, 1):
            component = chunk.get('component', 'Unknown')
            path = chunk.get('file_path', 'N/A')
            code = chunk.get('code', '')
            logging.info(f"Adding code chunk #{idx}: component={component}, path={path}")
            prompt += f"\n--- Code Chunk #{idx} ---\n"
            prompt += f"Component: {component}\nPath: {path}\nCode:\n{code.strip()}\n"

        prompt += "\n🔍 Your Task:\n"
        prompt += "- Diagnose the likely issue\n"
        prompt += "- Identify which component(s) are likely affected\n"
        prompt += "- Suggest a fix (code patch if possible)\n"
        prompt += "- Explain your reasoning\n"

        logging.info("Prompt built successfully")
        logging.info(f"Prompt content: {prompt[:500]}...")  # Show first
        return prompt

    def analyze(self, log_text: str, code_context: list) -> dict:
        """
        Sends prompt to the local LLM and returns the response.
        """
        logging.info("Starting local LLM analysis")
        prompt = self.build_prompt(log_text, code_context)
        logging.debug(f"Prompt sent to local LLM: {prompt[:200]}...")  # Show first 200 chars

        try:
            response = self.generator(prompt, max_length=512, do_sample=True)
            result = response[0]['generated_text']
            logging.info("Local LLM response received successfully")
        except Exception as e:
            logging.error(f"Error during local LLM analysis: {e}")
            result = "Error: Unable to get response from local LLM."

        return {
            "diagnosis": result,
            "prompt_used": prompt
        }