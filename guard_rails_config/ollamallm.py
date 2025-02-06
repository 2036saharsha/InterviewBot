from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain.schema import Generation
import ollama
from ollama import AsyncClient

class OllamaLLMResponse:
    def __init__(self, llm_output: str, generations: List):
        self.llm_output = llm_output
        self.generations = generations

class OllamaLLM(BaseLLM):
    temperature: float = 0.2
    streaming: bool = False

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ollamallm"
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        294
        stream = ollama.chat(
            model="llama3.1:8b",
            messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
            stream=True
        )

        for chunk in stream:
            text = chunk['message']['content']
            chunk_obj = GenerationChunk(text=text)
            yield chunk_obj
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        message = {'role': 'user', 'content': prompt}
        async for chunk in await AsyncClient().chat(model='llama3.1:8b', messages=[message], stream=True):
            text = chunk['message']['content']
            chunk_obj = GenerationChunk(text=text)
        
            if len(chunk_obj.text) == 0:
                continue
            yield chunk_obj
            if run_manager:
                await run_manager.on_llm_new_token(chunk_obj.text, chunk=chunk_obj)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append(
                [Generation(text=text, generation_info={"prompt": prompt})]
            )
        return LLMResult(
            generations=generations,
            llm_output={},
        )
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
            
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generations = []
        for prompt in prompts:
            text = await self._acall(
                prompt, stop=stop, run_manager=run_manager, **kwargs
            )
            generations.append(
                [Generation(text=text, generation_info={"prompt": prompt})]
            )
        return LLMResult(
            generations=generations,
            llm_output={}
        )
