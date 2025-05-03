from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy, HTTPGetRequestStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List
from dataclasses import dataclass
import json
import re
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@dataclass
class Message:
    type: str
    content: str

    def serialize(self):
        return {
            'type': self.type,
            'content': self.content
        }


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
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv(config['env_key'])
        if not api_key:
            raise ValueError(f"API key not found in environment for key: {config['env_key']}")

        # Set model type
        model_type = config.get('model_type', 'openai')
        entity.set_attribute('model_type', model_type)
        
        # Set model name with defaults based on type
        model_name = config.get('model_name')
        if not model_name:
            if model_type == 'openai':
                model_name = 'gpt-4o-mini'
        entity.set_attribute('model_name', model_name)
        
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
    entity_type = EntityEnum.API_MODEL
    strategy_description = 'Makes a call to the configured API model using LangChain'

    def verify_executable(self, entity, strategy_request):
        required_attrs = ['model_type', 'model_name', 'api_key', 'serialize_entities_and_strategies']
        return all(entity.has_attribute(attr) for attr in required_attrs)

    def form_context(self, entity) -> str:
        """Form context from document children"""
        doc_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.DOCUMENT)

        history = entity.get_attribute('message_history')
        
        if not doc_ids:
            return ""

        contexts = []


        contexts.append("These documents may contain import instructions or other relevant information:")
        for doc_id in doc_ids:
            doc = self.entity_service.get_entity(doc_id)
            if doc and doc.has_attribute('text'):
                doc_type = doc.get_document_type() or 'unknown'
                if doc.has_attribute('path'):
                    doc_type = f"{doc_type} ({doc.get_attribute('path')})"
                if doc.has_attribute('name'):
                    doc_type = f"{doc_type} - {doc.get_attribute('name')}"
                contexts.append(
                    f"{'='*50}\n"
                    f"Document Type and Name and Path : {doc_type.upper()}\n"
                    f"{'-'*50,'DOCUMENT_BEGIN'}\n"
                    f"{doc.get_text()}\n"
                    f"{'='*50,"DOCUMENT_END"}\n"
                )

        if self.strategy_request.param_config.get('serialize_entities_and_strategies', False):
            entity_graph = self.serialize_entity_and_children(entity.entity_id)
            contexts.append(
                f"{'=' * 50}\n"
                f"Entity Graph\n"
                f"{'-' * 50}\n"
                f"{json.dumps(entity_graph, indent=2)}\n"
                f"{'=' * 50}\n"
            )
            strategy_directory = self.get_strategy_directory(entity)
            contexts.append(
                f"{'='*50}\n"
                f"Strategy Directory\n"
                f"{'-'*50}\n"
                f"{strategy_directory}\n"
                f"{'='*50}\n"
            )

            available_entities = self.get_available_entities(entity)
            contexts.append(
                f"{'='*50}\n"
                f"Available Entities\n"
                f"{'-'*50}\n"
                f"{available_entities}\n"
                f"{'='*50}\n"
            )

        for i in range(len(history)-1, -1, -1):
            message = history[i]
            contexts.append(
                f"{'='*50}\n"
                f"Message Type: {message.type.upper()}\n"
                f"{'-'*50}\n"
                f"{message.content}\n"
                f"{'='*50}\n"
            )

        return "\n".join(contexts)

    def format_response_text(self, text: str, max_line_length: int = 80) -> str:
        """Format response text with proper line breaks"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # Check if adding this word exceeds max line length
            if current_length + len(word) + 1 <= max_line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                # Start new line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        # Add final line
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)

    def apply(self, entity) -> StrategyRequestEntity:
        # Get context from documents
        context = self.form_context(entity)
        
        # Get user input and system prompt from config
        config = self.strategy_request.param_config

        if entity.has_attribute('user_input'):
            user_input = entity.get_attribute('user_input')
        else:
            user_input = config.get('user_input', '')

        system_prompt = config.get('system_prompt', '')
        context_prefix = config.get('context_prefix', 'Here is the relevant context:')
        
        # Combine context with user input
        if context:
            combined_input = (
                f"{context_prefix}\n\n"
                f"{context}\n\n"
                f"{'='*50}\n"
                f"USER QUERY:\n"
                f"{'-'*50}\n"
                f"{user_input}\n"
                f"{'='*50}"
            )
        else:
            combined_input = user_input

        # # Initialize LangChain chat model
        # chat = ChatOpenAI(
        #     model_name=entity.get_attribute('model_name'),
        #     openai_api_key=entity.get_attribute('api_key'),
        #     max_tokens=entity.get_attribute('config').get('max_tokens', 1000)
        # )
        chat = init_chat_model(entity.get_attribute('model_name'), model_provider='google_genai', api_key=entity.get_attribute('api_key'))
        chat = chat.bind_tools([self.create_strategy_request, self.tasks_complete])
        tool_dict = {"create_strategy_request": self.create_strategy_request, "tasks_complete": self.tasks_complete}

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=combined_input))

        MAX_ITERS = 25
        CUR_ITERS = 0
        # Use invoke() instead of __call__

        self.add_to_message_history(entity, Message(type='request', content=user_input))
        while CUR_ITERS < MAX_ITERS:
            CUR_ITERS += 1
            response = chat.invoke(messages)
            content = "\n".join([str(item) for item in response.content]) if isinstance(response.content,                                                                                        list) else response.content
            self.add_to_message_history(entity, Message(type='response', content=content))
            if response.content != '':
                messages.append(AIMessage(content=response.content))

            self.entity_service.save_entity(entity)
            for tool_call in response.tool_calls:
                if tool_call['name'].lower() == 'tasks_complete':
                    CUR_ITERS = MAX_ITERS
                    break
                selected_tool = tool_dict.get(tool_call['name'].lower())
                tool_msg = selected_tool.invoke(tool_call)
                child_request = tool_msg.artifact
                request = self.execute_model_request(child_request, entity)
                entity = self.entity_service.get_entity(entity.entity_id)
                request_message = Message(type='response', content=f"Result of Model Executed strategy {request.strategy_name} with config {request.param_config} on target entity {request.target_entity_id}")
                self.add_to_message_history(entity, request_message)
                messages.append(HumanMessage(content=request_message.content))




        logger.info(f"Response from model: Message Received from APi model")

        # self.parse_strategy_requests(response.content, entity)
        
        entity.set_attribute('last_context', context)

        self.strategy_request.ret_val['context_used'] = context
        self.strategy_request.ret_val['entity'] = entity
        self.entity_service.save_entity(entity)
        
        return self.strategy_request

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
        return strategy_request.ret_val['strategy_registry']

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
            'children_ids': children
        }

        for child_id in children:
            return_dict = self.serialize_entity_and_children(child_id, return_dict)

        return return_dict

    @tool()
    def serialize_entities(self,entities: List[str]) -> List[dict]:
        '''
        Serialize a list of entities. If the model needs to know about entities with
        specific ids, this method will return more information about the entities.
        '''
        serialized_entities = []
        for entity_id in entities:
            entity = self.entity_service.get_entity(entity_id)
            if entity:
                serialized_entities.append(entity.serialize())
        return serialized_entities

    @staticmethod
    @tool
    def tasks_complete() -> str:
        '''
        End the conversation with the model. This method will be called when the conversation
        is over.
        '''
        return ''

    @staticmethod
    @tool(response_format="content_and_artifact")
    def create_strategy_request(strategy_name: str, param_config: dict | str, target_entity_id: str, add_to_history: bool = False):
        '''
        Create a strategy request entity with the given parameters.
        @param strategy_name: The name of the strategy to execute
        @param MUST BE JSON param_config: The configuration parameters for the strategy
        @param target_entity_id: The id of the target entity for the strategy
        @param add_to_history: Whether to add the strategy request to the entity's history
        '''
        if isinstance(param_config, str):
            param_config = json.loads(param_config)


        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = strategy_name
        strategy_request.param_config = param_config
        strategy_request.target_entity_id = target_entity_id
        strategy_request.add_to_history = add_to_history
        return "Created Strategy Request", strategy_request

    def add_to_message_history(self, entity, message):
        """
        Add a message to the message history of the entity.
        """
        history = entity.get_attribute('message_history')
        history.append(message)
        self.entity_service.save_entity(entity)


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
        self.entity_service.save_entity(entity)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {}
    
