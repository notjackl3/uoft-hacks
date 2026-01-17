"""
Backboard.io Multi-Model AI Service with Adaptive Memory
Implements intelligent model switching for different tasks
"""
from __future__ import annotations

import logging
import httpx
from typing import List, Dict, Any, Optional
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)

# Backboard.io API configuration
BACKBOARD_BASE_URL = "https://app.backboard.io/api"


class AIModel(Enum):
    """Available AI models for different tasks (optimized for speed)"""
    GPT4 = ("openai", "gpt-4o-mini")  # Faster planning with good quality
    GPT35 = ("openai", "gpt-3.5-turbo")  # Fast element matching
    CLAUDE = ("anthropic", "claude-3-7-sonnet-20250219")  # Code generation
    GEMINI = ("google", "gemini-2.5-flash-lite")  # Fastest quick tasks
    GROK = ("xai", "grok-3-mini")  # Faster quick decisions


class TaskType(Enum):
    """Types of tasks that require different AI models"""
    PLANNING = "planning"  # Complex multi-step planning
    ELEMENT_MATCHING = "element_matching"  # Fast element identification
    CODE_GENERATION = "code_generation"  # Generating selectors/code
    PATTERN_LEARNING = "pattern_learning"  # Learning user patterns
    QUICK_DECISION = "quick_decision"  # Fast yes/no decisions


class BackboardAI:
    """
    Adaptive AI service using Backboard.io's unified API
    Intelligently switches between models based on task type
    Maintains memory of user interactions for personalization
    """
    
    def __init__(self):
        self.api_key = settings.backboard_api_key
        self.base_url = BACKBOARD_BASE_URL
        self.headers = {"X-API-Key": self.api_key}
        self._user_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._model_stats: Dict[str, Dict[str, int]] = {}
        self._assistant_id: Optional[str] = None
        self._thread_cache: Dict[str, str] = {}  # user_id -> thread_id
        
    async def _get_or_create_assistant(self) -> str:
        """Get or create the Big Brother assistant"""
        if self._assistant_id:
            return self._assistant_id
            
        async with httpx.AsyncClient() as client:
            # Create assistant
            response = await client.post(
                f"{self.base_url}/assistants",
                json={
                    "name": "Big Brother Web Automation",
                    "system_prompt": (
                        "You are a precise web automation planner. "
                        "Generate step-by-step plans for navigating websites. "
                        "Always return valid JSON with a 'steps' array."
                    )
                },
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            self._assistant_id = response.json()["assistant_id"]
            logger.info(f"Created Backboard assistant: {self._assistant_id}")
            return self._assistant_id
    
    async def _get_or_create_thread(self, user_id: Optional[str]) -> str:
        """Get or create a conversation thread for this user"""
        if not user_id:
            user_id = "anonymous"
            
        if user_id in self._thread_cache:
            return self._thread_cache[user_id]
        
        assistant_id = await self._get_or_create_assistant()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/assistants/{assistant_id}/threads",
                json={},
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            thread_id = response.json()["thread_id"]
            self._thread_cache[user_id] = thread_id
            logger.info(f"Created thread {thread_id} for user {user_id}")
            return thread_id
    
    def _select_model_for_task(self, task_type: TaskType) -> tuple[str, str]:
        """
        Intelligent model selection based on task type
        Returns (provider, model_name) tuple
        This is a key feature for the Backboard.io challenge
        """
        model_mapping = {
            TaskType.PLANNING: AIModel.GPT4,  # GPT-4 for complex reasoning
            TaskType.ELEMENT_MATCHING: AIModel.GPT35,  # Fast for matching
            TaskType.CODE_GENERATION: AIModel.CLAUDE,  # Claude excels at code
            TaskType.PATTERN_LEARNING: AIModel.GEMINI,  # Gemini for analysis
            TaskType.QUICK_DECISION: AIModel.GROK,  # Grok for quick decisions
        }
        
        selected = model_mapping.get(task_type, AIModel.GPT35)
        provider, model = selected.value
        logger.info(f"Selected {provider}/{model} for task {task_type.value}")
        return provider, model
    
    async def _call_with_memory(
        self, 
        user_id: str, 
        prompt: str, 
        task_type: TaskType
    ) -> str:
        """
        Core method that calls Backboard with memory tracking
        Uses the REST API to send messages to threads
        """
        # Get or create assistant (also creates thread internally)
        assistant_id = await self._get_or_create_assistant()
        
        # Get or create thread for this user
        thread_id = await self._get_or_create_thread(user_id)
        
        # Select appropriate model for task
        llm_provider, model_name = self._select_model_for_task(task_type)
        
        # Track the interaction
        if user_id not in self._user_patterns:
            self._user_patterns[user_id] = {
                "preferences": {},
                "common_actions": [],
                "last_tasks": []
            }
        
        # Send message to thread with model selection
        # POST /threads/{thread_id}/messages (uses form-data, not JSON!)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKBOARD_BASE_URL}/threads/{thread_id}/messages",
                    data={  # Use form data, not json!
                        "content": prompt,
                        "llm_provider": llm_provider,
                        "model_name": model_name,
                        "memory": "Auto",  # Enable memory
                        "stream": "false",  # String, not boolean!
                        "send_to_llm": "true"
                    },
                    headers=self.headers,
                    timeout=30.0
                )
                
                # Log response details for debugging
                if response.status_code != 200:
                    logger.error(f"Backboard API error response: {response.text}")
                    logger.error(f"Request: llm_provider={llm_provider}, model_name={model_name}")
                
                response.raise_for_status()
                data = response.json()
            
            # Track token usage
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            self._track_usage(llm_provider, model_name, total_tokens)
            
            # Update patterns
            self._user_patterns[user_id]["last_tasks"].append({
                "type": task_type.value,
                "model": f"{llm_provider}/{model_name}",
                "timestamp": data.get("created_at", "")
            })
            
            # Extract message content from response
            content = data.get("content", "")
            logger.info(f"Backboard response content (first 200 chars): {content[:200]}")
            return content
            
        except Exception as e:
            logger.error(f"Error calling Backboard API: {str(e)}")
            raise

    
    async def generate_plan(
        self,
        user_goal: str,
        page_features: List[Dict],
        url: str,
        page_title: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Generate a plan using the appropriate model
        Uses adaptive memory to personalize based on user history
        """
        # Get user's historical patterns for this type of task
        user_context = self._get_user_context(user_id, url) if user_id else ""
        
        # Build enhanced prompt with user history
        prompt = self._build_planning_prompt(
            user_goal, page_features, url, page_title, user_context
        )
        
        try:
            # Call Backboard.io with memory enabled
            response = await self._call_with_memory(
                user_id=user_id or "anonymous",
                prompt=prompt,
                task_type=TaskType.PLANNING
            )
            
            # Track this interaction for adaptive learning
            if user_id:
                self._record_interaction(user_id, "planning", {
                    "goal": user_goal,
                    "url": url,
                    "success": True
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Backboard planning failed: {e}")
            # Fallback to simpler model
            logger.info("Falling back to GPT-3.5")
            return await self._call_with_memory(
                user_id=user_id or "anonymous",
                prompt=prompt,
                task_type=TaskType.ELEMENT_MATCHING  # Use faster model as fallback
            )
    
    async def match_element(
        self,
        target_description: str,
        available_elements: List[Dict],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fast element matching using a quicker model
        Uses learned preferences to improve accuracy
        """
        prompt = f"""Match the target description to the best element:
Target: {target_description}
Elements: {available_elements[:20]}

Return JSON: {{"index": <number>, "confidence": <0-1>}}"""
        
        response = await self._call_with_memory(
            user_id=user_id or "anonymous",
            prompt=prompt,
            task_type=TaskType.ELEMENT_MATCHING
        )
        
        return {"response": response}
    
    async def learn_pattern(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from user interactions to improve future responses
        Key feature for adaptive memory requirement
        """
        # Get historical patterns
        patterns = self._user_patterns.get(user_id, {})
        
        prompt = f"""Analyze this user interaction and identify patterns:
Current interaction: {interaction_data}
Historical patterns: {patterns.get('last_tasks', [])[-10:] if patterns else 'None yet'}

What patterns do you see? What should we optimize for this user?
Return JSON with insights."""
        
        response = await self._call_with_memory(
            user_id=user_id,
            prompt=prompt,
            task_type=TaskType.PATTERN_LEARNING
        )
        
        return {"insights": response}
    
    def _track_usage(self, provider: str, model: str, tokens: int):
        """Track model usage for statistics"""
        model_key = f"{provider}/{model}"
        if model_key not in self._model_stats:
            self._model_stats[model_key] = {"calls": 0, "tokens": 0}
        
        self._model_stats[model_key]["calls"] += 1
        self._model_stats[model_key]["tokens"] += tokens
        
        logger.info(f"Tracked usage: {model_key} - {tokens} tokens")
    
    def _get_user_context(self, user_id: str, current_url: str) -> str:
        """
        Build context from user's history for personalization
        Adaptive memory implementation
        """
        user_data = self._user_patterns.get(user_id, {})
        last_tasks = user_data.get("last_tasks", [])
        
        if not last_tasks:
            return ""
        
        # Find similar past interactions (simplified - checking last tasks)
        similar_count = len([t for t in last_tasks if current_url in str(t)])
        
        if similar_count > 0:
            return f"\nUser History: This user has {similar_count} interactions on similar pages. Total: {len(last_tasks)} tasks."
        
        return f"\nUser History: First time on this type of site. {len(last_tasks)} total interactions recorded."
    
    def _record_interaction(
        self,
        user_id: str,
        interaction_type: str,
        data: Dict[str, Any]
    ):
        """Record interaction for adaptive learning"""
        if user_id not in self._user_patterns:
            self._user_patterns[user_id] = {
                "preferences": {},
                "common_actions": [],
                "last_tasks": []
            }
        
        self._user_patterns[user_id]["last_tasks"].append({
            "type": interaction_type,
            **data
        })
        
        # Keep only last 100 interactions per user
        if len(self._user_patterns[user_id]["last_tasks"]) > 100:
            self._user_patterns[user_id]["last_tasks"] = self._user_patterns[user_id]["last_tasks"][-100:]
        
        logger.info(f"Recorded interaction for user {user_id}: {interaction_type}")
    
    def _build_planning_prompt(
        self,
        user_goal: str,
        features: List[Dict],
        url: str,
        page_title: str,
        user_context: str
    ) -> str:
        """Build enhanced prompt with user context"""
        import json
        
        features_json = json.dumps(features[:30], ensure_ascii=False)
        
        return f"""You are a web automation planner with adaptive memory.

GOAL: {user_goal}
PAGE_TITLE: {page_title}
URL: {url}
{user_context}

ELEMENTS: {features_json}

Generate a step-by-step plan. Consider the user's history when making decisions.
Output JSON only."""
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage (for debugging/optimization)"""
        return {
            "models_used": list(self._model_stats.keys()),
            "total_calls": sum(s["calls"] for s in self._model_stats.values()),
            "stats_by_model": self._model_stats,
            "users_tracked": len(self._user_patterns),
        }


# Singleton instance
backboard_ai = BackboardAI()
