import datetime
import pytz
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from shared_utils.entities.Entity import Entity
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy.QueryEntitiesStrategy import QueryEntitiesStrategy
from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy, HTTPGetRequestStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import re
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# @dataclass
# class Message:
#     type: str
#     content: str

#     def serialize(self):
#         return {
#             'type': self.type,
#             'content': self.content
#         }


class ConfigureApiModelStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Configures an API model with connection details and parameters'

    def verify_executable(self, entity, strategy_request):
        config = strategy_request.param_config
        required = ['model_type', 'env_key']
        return all(k in config for k in required)

    def apply(self, entity) -> StrategyRequestEntity:
        config = self.strategy_request.param_config
        
        # Load environment variables
        project_root = Path(__file__).resolve().parents[4]
        dotenv_path = project_root / 'docker' / '.env'
        load_dotenv(dotenv_path=dotenv_path)

        # Set model type
        model_type = config.get('model_type', 'openai')
        entity.set_attribute('model_type', model_type)
        
        # Set model name with defaults based on type
        model_name = config.get('model_name')
        if not model_name:
            if model_type == 'openai':
                model_name = 'gpt-4o-mini'
        entity.set_attribute('model_name', model_name)

        if model_type == 'openai':
            env_var = 'OPENAI_API_KEY'
        elif model_type == 'anthropic':
            env_var = 'ANTHROPIC_API_KEY'
        elif model_type == 'google_genai':
            env_var = 'GOOGLE_API_KEY'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        api_key = os.getenv(env_var)
        
        # Set API key
        entity.set_attribute('api_key', api_key)
        
        # Set config with OpenAI defaults if not provided
        default_config = {
            'max_tokens': 5000,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stream': False
        }
        
        if 'model_config' in config:
            default_config.update(config['model_config'])
        
        entity.set_attribute('config', default_config)

        self.entity_service.save_entity(entity)
            
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'model_type': 'openai',  # Optional, defaults to openai
            'model_name': 'gpt-4o-mini',  # Optional, has defaults per model_type
            'env_key': 'OPENAI_API_KEY',  # Required - environment variable key
            'model_config': {  # Optional - has defaults
                'max_tokens': 4000,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'stream': False
            }
        }
    

class CallApiModelStrategy(Strategy):
    """
    Strategy for making calls to configured API models using LangChain.
    
    This strategy implements an agent loop that:
    1. Forms context from instruction documents, entity data, and available strategies
    2. Initializes a chat model with tool binding
    3. Runs an iterative conversation loop with the model
    4. Handles tool calls including strategy execution, entity management, and user interaction
    5. Maintains visible entities and message history throughout the conversation
    
    The agent continues until either MAX_ITERS is reached or yield_to_user tool is called.
    """
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Makes a call to the configured API model using LangChain'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MAX_ITERS = 15  # Maximum number of agent loop iterations
        self.CUR_ITERS = 0   # Current iteration count

    def verify_executable(self, entity, strategy_request):
        required_attrs = ['model_type', 'model_name', 'api_key', 'serialize_entities_and_strategies']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Execute the main agent loop:
        1. Form context from instruction documents and system information
        2. Initialize chat model with tool bindings
        3. Add user input to message history
        4. Run iterative conversation loop with model
        5. Handle tool calls until yield_to_user or max iterations reached
        """
        # Form the complete context for the agent
        base_context = self.form_context(entity)
        
        # Extract user input from strategy request configuration
        config = self.strategy_request.param_config
        user_input = config.get('user_input', '')

        # Initialize the chat model with tool bindings
        chat = init_chat_model(
            model=entity.get_attribute('model_name'), 
            model_provider=entity.get_attribute('model_type'), 
            api_key=entity.get_attribute('api_key')
        )
        chat = chat.bind_tools([self.create_strategy_request, self.yield_to_user, self.update_visible_entities])
        tool_dict = {
            "create_strategy_request": self.create_strategy_request, 
            "yield_to_user": self.yield_to_user, 
            "update_visible_entities": self.update_visible_entities
        }

        # Reset iteration counters for this execution
        self.CUR_ITERS = 0

        # Add the user's input to message history to start the conversation
        self.add_to_message_history(entity, HumanMessage(content=user_input))
        
        # Main agent conversation loop
        while self.CUR_ITERS < self.MAX_ITERS:
            self.CUR_ITERS += 1

            # Build the complete context for this iteration
            # Includes base context + dynamically added visible entities
            complete_context = base_context + self._get_visible_entities_context(entity)
            
            # Create system message with complete context and invoke model
            system_message = SystemMessage(content=complete_context)
            model_input = [system_message] + entity.get_attribute('message_history')
            response = chat.invoke(model_input)
            self.add_to_message_history(entity, response)
            
            # Save entity state after each model response
            self.entity_service.save_entity(entity)
            
            # Process all tool calls from the model's response
            for tool_call in response.tool_calls:
                # Execute tool and check if we should break the loop
                should_break = self._call_tool(entity, tool_call, tool_dict)
                entity = self.entity_service.get_entity(entity.entity_id)
                if should_break:
                    break

        logger.info("Agent loop completed - control returned to user")
        self.entity_service.save_entity(entity)
        
        return self.strategy_request
    
    def form_context(self, entity: Entity) -> str:
        """
        Forms the complete context for the API model by gathering various pieces of information.
        
        The context includes:
        1. System instructions from the agent_instructions.md file
        2. Custom instruction documents tagged for this specific entity
        3. Serialized entity data (agent's own information)
        4. Available strategies directory for tool selection
        5. List of available entities in the system
        6. Current timestamp for date-aware operations
        
        Args:
            entity: The API model entity that will receive this context
            
        Returns:
            str: Complete formatted context string ready for the model
        """
        contexts = []
        processed_instruction_ids = set()

        # 1. Load and add core system instructions from markdown file
        # These are the fundamental behavior rules the agent must follow
        with open("shared_utils/entities/api_model/strategy/agent_instructions.md", "r") as f:
            instructions = f.read()
            contexts.append(
                f"{'='*50}\n"
                f"SYSTEM INSTRUCTIONS DO NOT DEVIATE FROM THESE INSTRUCTIONS\n"
                f"{'-'*50}\n"
                f"{instructions}\n"
                f"{'='*50}\n"
            )

        # 2. Find and add custom instruction documents for this specific entity
        # These are documents tagged with 'instructions_{entity_id}' = True
        # Allows the agent to create persistent instructions for itself
        instruction_doc_request = QueryEntitiesStrategy.request_constructor(entity.entity_id, [
            {
                'attribute': 'instructions_'+entity.entity_id,
                'operator': 'equals',
                'value': True
            }
        ])
        instruction_doc_request = self.executor_service.execute_request(instruction_doc_request)
        matched_ids = instruction_doc_request.ret_val['matching_entity_ids']
        contexts.append("HERE ARE ADDITIONAL INSTRUCTIONS THAT YOU HAVE PROVIDED TO YOURSELF BY SPECIFYING ATTRIBUTE 'instructions_entity_id' ON THE DOCUMENT TO TRUE WHERE entity_id is your id. You probably added this to come back to later. IF YOU WANT TO CREATE A NEW INSTRUCTION DOCUMENT FOR YOURSELF, YOU MUST SET THE ATTRIBUTE 'instructions_YOUR_ID' ON THE DOCUMENT TO TRUE")
        
        # Recursively process all instruction documents and their children
        for matched_id in matched_ids:
            self._add_instructions_recursively(matched_id, processed_instruction_ids, contexts)

        # 3. Add the agent's own entity data (excluding message history and strategy requests to avoid recursion)
        # This gives the agent self-awareness of its current state and attributes
        self_serialized = entity.serialize()
        del self_serialized['message_history']  # Remove to prevent context bloat
        del self_serialized['strategy_requests']  # Remove to prevent recursion
        
        contexts.append(
            f"{'='*50}\n"
            f"Entity Context THIS IS YOU, THE AGENT\n"
            f"{'-'*50}\n"
            f"{json.dumps(self_serialized, indent=2)}\n"
            f"{'='*50}\n"
        )

        processed_instruction_ids.add(entity.entity_id)

        # 4. Add available strategies directory
        # This provides the agent with a catalog of all executable strategies and their configurations
        strategy_directory = self.get_strategy_directory(entity)
        contexts.append(
            f"{'='*50}\n"
            f"Strategy Directory: FIND THE STRATEGY YOU NEED TO EXECUTE FROM THIS DIRECTORY \n"
            f"{'-'*50}\n"
            f"{strategy_directory}\n"
            f"{'='*50}\n"
        )

        # 5. Add available entities in the system
        # This gives the agent awareness of what entities exist and can be operated on
        available_entities = self.get_available_entities(entity)
        contexts.append(
            f"{'='*50}\n"
            f"Available Entities\n"
            f"{'-'*50}\n"
            f"{available_entities}\n"
            f"{'='*50}\n"
        )

        # 6. Add current timestamp for date-aware operations
        # Provides the agent with temporal context for scheduling and time-based decisions
        est_tz = pytz.timezone('US/Eastern')
        current_time_est = datetime.datetime.now(est_tz)
        contexts.append("HERE IS THE CURRENT DATE USE IT IF THE USER REQUESTS DATE RELEVANT INFORMATION: " + current_time_est.strftime("%A, %Y-%m-%d %H:%M:%S %Z"))

        # Store the IDs of instruction documents in the entity for reference
        # This helps track what's already in context to avoid duplication in visible entities
        entity.set_attribute('ids_in_context', list(processed_instruction_ids))

        return "\n".join(contexts)
    
     # =================== UTILITY METHODS ===================
    
    def _set_visible_entities(self, entity: 'Entity', entity_ids: List[str], method: str = 'a') -> None:
        """
        Centralized method for managing visible entities with proper deduplication.
        
        Args:
            entity: The API model entity to update
            entity_ids: List of entity IDs to add or remove
            method: 'a' for add, 'r' for remove
        """
        current_visible = entity.get_attribute('visible_entities') if entity.has_attribute('visible_entities') else []
        ids_in_context = entity.get_attribute('ids_in_context') if entity.has_attribute('ids_in_context') else []
        
        if method == 'a':
            # Add new entity IDs, avoiding duplicates and entities already in context
            new_entities = [eid for eid in entity_ids if eid not in current_visible and eid not in ids_in_context]
            updated_visible = current_visible + new_entities
        elif method == 'r':
            # Remove specified entity IDs
            updated_visible = [eid for eid in current_visible if eid not in entity_ids]
        else:
            raise ValueError(f"Invalid method: {method}. Use 'a' for add or 'r' for remove.")
        
        entity.set_attribute('visible_entities', updated_visible)
        self.entity_service.save_entity(entity)

    def _get_visible_entities_context(self, entity: 'Entity') -> str:
        """
        Generate context string for visible entities, excluding those already in the main context.
        
        Args:
            entity: The API model entity
            
        Returns:
            str: Formatted context string with visible entity data
        """
        context_parts = ["\n HERE ARE THE VISIBLE ENTITIES YOU CURRENTLY HAVE ACCESS TO: \n"]
        
        # Get current visible entities and IDs already in context
        visible_entities = list(entity.get_attribute('visible_entities') if entity.has_attribute('visible_entities') else [])
        ids_in_context = entity.get_attribute('ids_in_context') if entity.has_attribute('ids_in_context') else []
        
        # Process each visible entity
        for entity_id in visible_entities[:]:  # Use slice copy to avoid modification during iteration
            # Skip entities already included in the main context
            if entity_id in ids_in_context:
                continue
                
            try:
                # Fetch and serialize the entity
                current_entity = self.entity_service.get_entity(entity_id)
                if current_entity:
                    context_parts.append(json.dumps(current_entity.serialize()) + "\n")
                else:
                    # Remove invalid entity IDs from visible entities
                    visible_entities.remove(entity_id)
            except Exception as e:
                # Remove problematic entity IDs and log the error
                logger.warning(f"Removing invalid entity {entity_id} from visible entities: {e}")
                visible_entities.remove(entity_id)
        
        # Update visible entities if any were removed
        current_visible_count = len(entity.get_attribute('visible_entities')) if entity.has_attribute('visible_entities') else 0
        if len(visible_entities) != current_visible_count:
            entity.set_attribute('visible_entities', visible_entities)
        
        return "".join(context_parts)

    def _call_tool(self, entity: Entity, tool_call: Dict[str, Any], tool_dict: Dict[str, Any]) -> bool:
        """
        Execute a single tool call and handle the response.
        
        Args:
            entity: The API model entity
            tool_call: The tool call dictionary from the model response
            tool_dict: Dictionary mapping tool names to tool functions
            
        Returns:
            bool: True only for 'yield_to_user' (breaks tool loop), False for all others (continue processing)
        """
        tool_name = tool_call['name'].lower()
        
        try:
            if tool_name == 'yield_to_user':
                # Signal end of agent loop - return control to user
                self.CUR_ITERS = self.MAX_ITERS
                tool_message = ToolMessage(
                    content="Control Yielded Back to the user",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                self.add_to_message_history(entity, tool_message)
                return True  # Break the tool loop
                
            elif tool_name == 'update_visible_entities':
                # Update the visible entities list
                tool = tool_dict.get(tool_name)
                tool_msg = tool.invoke(tool_call)
                entity_ids, method = tool_msg.artifact
                
                # Use the centralized method for updating visible entities
                self._set_visible_entities(entity, entity_ids, method)
                
                visible_entities_list = entity.get_attribute('visible_entities') if entity.has_attribute('visible_entities') else []
                tool_message = ToolMessage(
                    content=f"Updated visible entities: {visible_entities_list}",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                self.add_to_message_history(entity, tool_message)
                return False  # Continue processing other tools
                
            elif tool_name == 'create_strategy_request':
                # Execute a strategy on another entity
                selected_tool = tool_dict.get(tool_name)
                tool_msg = selected_tool.invoke(tool_call)
                child_request = tool_msg.artifact
                
                if child_request:
                    # Execute the strategy request
                    request = self.execute_model_request(child_request, entity)
                    entity = self.entity_service.get_entity(entity.entity_id)  # Refresh entity
                    
                    # Clean up return value for response
                    ret_val = request.ret_val.copy()
                    ret_val.pop('entity', None)
                    ret_val.pop('child_entity', None)
                    
                    # Collect all affected entity IDs for visibility updates
                    affected_entity_ids = [request.target_entity_id]
                    
                    # Extract entity IDs from param_config
                    if request.param_config:
                        affected_entity_ids.extend(self.extract_entity_ids_from_obj(request.param_config))
                    
                    # Add target_entity_ids if present
                    if hasattr(request, 'target_entity_ids') and request.target_entity_ids:
                        affected_entity_ids.extend(request.target_entity_ids)
                    
                    # Remove duplicates and update visible entities
                    affected_entity_ids = list(set(affected_entity_ids))
                    self._set_visible_entities(entity, affected_entity_ids, 'a')
                    
                    # Create response message for the model
                    tool_message_content = (
                        f"Result of model executed strategy {request.strategy_name} "
                        f"with config {request.param_config} on target entity {request.target_entity_id}. "
                        f"This step is complete: Are there any further actions needed? "
                        f"If yes, complete further actions, else let the user return additional information. "
                        f"Be sure to call the yield_to_user() tool.\n\n"
                        f"\n{json.dumps(ret_val, indent=2)}\n\n"
                        f"```entities\n{json.dumps(affected_entity_ids)}\n```"
                    )
                    
                    tool_message = ToolMessage(
                        content=tool_message_content,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                    
                    self.add_to_message_history(entity, tool_message)
                return False  # Continue processing other tools
                
            else:
                # Unknown tool
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            # Handle tool execution errors
            error_message = f"Error executing tool {tool_call['name']}: {str(e)}"
            logger.error(error_message)
            tool_message = ToolMessage(
                content=error_message,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
            self.add_to_message_history(entity, tool_message)
            return False  # Continue with next tool

    def execute_model_request(self, request, entity):
        target_entity = self.entity_service.get_entity(request.target_entity_id)
        if target_entity is None:
            raise ValueError(f'Entity not found for id: {request.target_entity_id}')
        target_entity.add_child(request)
        self.entity_service.save_entity(target_entity)
        self.entity_service.save_entity(request)
        request = self.executor_service.execute_request(request)

        return request


    def get_strategy_directory(self, entity: 'Entity'):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = HTTPGetRequestStrategy.__name__
        strategy_request.target_entity_id = entity.entity_id
        strategy_request.param_config = {
            'url': 'http://127.0.0.1:8000/training_session/api/get_strategy_registry/',
            'response_attribute': 'strategy_registry'
        }
        strategy_request = self.executor_service.execute_request(strategy_request)
        registry = strategy_request.ret_val['strategy_registry']
        registry_flattened = [d for group in registry.values() for d in group]
        for strategy in registry_flattened:
            del strategy['source']
        return registry_flattened

    def get_available_entities(self, entity: 'Entity'):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = HTTPGetRequestStrategy.__name__
        strategy_request.target_entity_id = entity.entity_id
        strategy_request.param_config = {
            'url': 'http://127.0.0.1:8000/training_session/api/get_available_entities/',
            'response_attribute': 'available_entities'
        }
        strategy_request = self.executor_service.execute_request(strategy_request)
        return strategy_request.ret_val['available_entities']

    def serialize_entity_and_children(self, entity_id, return_dict=None):
        try:
            entity = self.entity_service.get_entity(entity_id)
        except Exception as e:
            return return_dict
        if not entity:
            return None
        children = entity.get_children()
        if return_dict is None:
            return_dict = {}

        if entity_id in return_dict:
            return return_dict
        return_dict[entity_id] = {
            'entity_id' : entity_id,
            'entity_type': entity.entity_name.value,
            'children_ids': children,
            'name' : entity.get_attribute('name') if entity.has_attribute('name') else None,
        }

        for child_id in children:
            return_dict = self.serialize_entity_and_children(child_id, return_dict)

        return return_dict


    def extract_entity_ids_from_obj(self, obj: Union[str, Dict[str, Any], List[Any]]) -> List[str]:
        """Extract entity IDs (UUIDs) from various object types"""
        import re
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        if isinstance(obj, str):
            return re.findall(uuid_pattern, obj)
        elif isinstance(obj, dict):
            ids = []
            for v in obj.values():
                ids.extend(self.extract_entity_ids_from_obj(v))
            return ids
        elif isinstance(obj, list):
            ids = []
            for item in obj:
                ids.extend(self.extract_entity_ids_from_obj(item))
            return ids
        return []
    
    def _add_instructions_recursively(self, entity_id: str, processed_ids: set, contexts: list):
        """
        Recursively traverses entities, adding document text to contexts.
        Uses a set of processed_ids to avoid redundant work and cycles.
        """
        if entity_id in processed_ids:
            return  # Avoid processing the same entity twice
        
        processed_ids.add(entity_id)
        
        try:
            instruction_doc = self.entity_service.get_entity(entity_id)
            if not instruction_doc:
                return

            # If the entity is a document, add its content to the context
            if instruction_doc.entity_name == EntityEnum.DOCUMENT and instruction_doc.has_attribute('text'):
                contexts.append(
                    f"ENTITY ID: {entity_id}\n"
                    f"ENTITY NAME: {instruction_doc.get_attribute('name')}\n"
                    f"ENTITY TEXT: {instruction_doc.get_attribute('text')}\n"
                    f"CHILDREN: {instruction_doc.get_children()}\n"
                    f"\n\n"
                )

            # Recurse for all children
            for child_id in instruction_doc.get_children():
                self._add_instructions_recursively(child_id, processed_ids, contexts)

        except Exception as e:
            logger.error(f"Error processing instruction entity {entity_id}: {e}")

    def add_to_message_history(self, entity: 'Entity', message: Union[HumanMessage, AIMessage, ToolMessage, SystemMessage]) -> None:
        """
        Add a message to the message history of the entity.
        """
        history = entity.get_attribute('message_history')
        history.append(message)
        self.entity_service.save_entity(entity)

    # =================== STATIC TOOL METHODS ===================

    @staticmethod
    @tool(response_format="content_and_artifact")
    def serialize_entities(entities: List[str]) -> List[dict]:
        '''
        Serialize a list of entity ids into a list of dictionaries.
        @param entities: List of entity ids to serialize
        @return: List of serialized entities
        '''
        entity_service = EntityService()
        serialized_entities = []
        for entity_id in entities:
            entity = entity_service.get_entity(entity_id)
            if entity:
                serialized_entities.append(entity.serialize())
        return "Serialized Entities", serialized_entities
    
    @staticmethod
    @tool(response_format="content_and_artifact")
    def update_visible_entities(entity_ids: List[str], method: str = 'a') -> str:
        '''
        Update the visible entities for the API model.
        @param entity_ids: List of entity ids to update
        @param method: The method to use to update the visible entities
        '''
        return "updated entitiees", (entity_ids, method)

    @staticmethod
    @tool
    def yield_to_user() -> str:
        '''
            T@tool(
            name="yield_to_user",
            description="Call this (no args) when you have finished everything "
            "you can do without more input from the human. "
            "It hands control back to the user."
        )
        '''
        return ''

    @staticmethod
    @tool(response_format="content_and_artifact")
    def create_strategy_request(strategy_name: str, param_config: dict | str, target_entity_id: str, add_to_history: bool = False, target_entity_ids: List[str] = None) -> str:
        '''
        Create a strategy request entity with the given parameters.
        @param strategy_name: The name of the strategy to execute
        @param MUST BE JSON param_config: The configuration parameters for the strategy
        @param target_entity_id: The id of the target entity for the strategy
        @param add_to_history: Whether to add the strategy request to the entity's history
        @param target_entity_ids: List of target entity ids for the strategy if we are executing the request on multiple entities
        '''
        if isinstance(param_config, str):
            param_config = json.loads(param_config)


        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = strategy_name
        strategy_request.param_config = param_config
        strategy_request.target_entity_id = target_entity_id
        strategy_request.add_to_history = False
        strategy_request.set_attribute('target_entity_ids', target_entity_ids)
        return "Created Strategy Request", strategy_request

    @staticmethod
    def get_request_config():
        return {
            'user_input': '',  # Optional additional input
            'system_prompt': '',  # Optional system prompt
            'context_prefix': 'Here is the relevant context:',  # Optional prefix for context\
            'serialize_entities_and_strategies': False # Optional flag to serialize entities and strategies
        }

class ClearChatHistoryStrategy(Strategy):
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Clears the chat history for an API model'

    def apply(self, entity) -> StrategyRequestEntity:
        entity.set_attribute('message_history', [])
        entity.set_attribute('response', [])
        entity.set_attribute('visible_entities', [])
        self.entity_service.save_entity(entity)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {}
    
